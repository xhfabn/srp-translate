import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import transforms as tfs

import timm
import util.util as util
from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss


class ROMAMaskDetailModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--adj_size_list', type=list, default=[2, 4, 6, 8, 12], help='different scales of perception field')
        parser.add_argument('--lambda_mlp', type=float, default=1.0, help='weight of lr for discriminator')
        parser.add_argument('--lambda_motion', type=float, default=1.0, help='weight for Temporal Consistency')
        parser.add_argument('--lambda_D_ViT', type=float, default=1.0, help='weight for discriminator')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_global', type=float, default=1.0, help='weight for Global Structural Consistency')
        parser.add_argument('--lambda_spatial', type=float, default=1.0, help='weight for Local Structural Consistency')
        parser.add_argument('--atten_layers', type=str, default='1,3,5', help='compute Cross-Similarity on which layers')
        parser.add_argument('--local_nums', type=int, default=256)
        parser.add_argument('--which_D_layer', type=int, default=-1)
        parser.add_argument('--side_length', type=int, default=7)
        parser.add_argument('--lambda_local_gan', type=float, default=0.5, help='weight for local GAN loss')
        parser.add_argument('--lambda_detail_res', type=float, default=0.1, help='weight for detail residual regularization')
        parser.add_argument('--lambda_keep', type=float, default=5.0, help='weight for non-content-region keep loss')
        parser.add_argument('--detail_scale', type=float, default=0.1, help='detail residual scaling factor')
        parser.add_argument('--mask_percent', type=float, default=0.25, help='top percent of structure-rich area used in mask')
        parser.add_argument('--local_patch_size', type=int, default=96, help='patch size for local discriminator')
        parser.add_argument('--local_patch_k', type=int, default=4, help='number of local patches sampled per image')
        parser.add_argument('--mask_blur_kernel', type=int, default=5, help='average pooling kernel for mask smoothing')
        parser.set_defaults(pool_size=0)
        parser.set_defaults(netG='resnet_9blocks_mask_detail')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'motion',
                           'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
        self.visual_names = ['real_A0', 'real_A1', 'mask_A0', 'mask_A1', 'fake_B0', 'fake_B1', 'real_B0', 'real_B1']
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]
        self.is_single_frame = False

        if self.isTrain:
            self.model_names = ['G', 'D_ViT', 'D_local']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain,
                                      opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            self.netD_local = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                                opt.normD, opt.init_type, opt.init_gain,
                                                opt.no_antialias, self.gpu_ids, opt)
            self.netPreViT = timm.create_model('vit_base_patch16_384', pretrained=True).to(self.device)
            self.resize = tfs.Resize(size=(384, 384))
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for _ in self.atten_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionL1 = nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_local = torch.optim.Adam(self.netD_local.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_ViT)
            self.optimizers.append(self.optimizer_D_local)

    def data_dependent_initialize(self, data):
        pass

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD_ViT, self.netD_local], True)
        self.optimizer_D_ViT.zero_grad()
        self.optimizer_D_local.zero_grad()
        self.loss_D = self.compute_D_loss() + self.compute_local_D_loss()
        self.loss_D.backward()
        self.optimizer_D_ViT.step()
        self.optimizer_D_local.step()

        self.set_requires_grad([self.netD_ViT, self.netD_local], False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.is_single_frame = 'A0' not in input
        if self.is_single_frame:
            self.real_A0 = input['A' if AtoB else 'B'].to(self.device)
            self.real_A1 = self.real_A0
            self.real_B0 = input['B' if AtoB else 'A'].to(self.device)
            self.real_B1 = self.real_B0
        else:
            self.real_A0 = input['A0' if AtoB else 'B0'].to(self.device)
            self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
            self.real_B0 = input['B0' if AtoB else 'A0'].to(self.device)
            self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.mask_A0 = self.build_content_mask(self.real_A0)
        self.mask_A1 = self.build_content_mask(self.real_A1)
        self.fake_B0, self.base_B0, self.residual_B0, _, self.detail_B0 = self.netG(self.real_A0, self.mask_A0)
        self.fake_B1, self.base_B1, self.residual_B1, _, self.detail_B1 = self.netG(self.real_A1, self.mask_A1)

        if self.is_single_frame:
            self.fake_B = self.fake_B0
            self.real_A = self.real_A0
            self.real_B = self.real_B0
            self.mask_A = self.mask_A0

        if self.opt.isTrain:
            self.real_A0_resize = self.resize(self.real_A0)
            self.real_A1_resize = self.resize(self.real_A1)
            real_B0 = self.resize(self.real_B0)
            real_B1 = self.resize(self.real_B1)
            self.fake_B0_resize = self.resize(self.fake_B0)
            self.fake_B1_resize = self.resize(self.fake_B1)
            self.mutil_real_A0_tokens = self.netPreViT(self.real_A0_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_A1_tokens = self.netPreViT(self.real_A1_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_B0_tokens = self.netPreViT(real_B0, self.atten_layers, get_tokens=True)
            self.mutil_real_B1_tokens = self.netPreViT(real_B1, self.atten_layers, get_tokens=True)
            self.mutil_fake_B0_tokens = self.netPreViT(self.fake_B0_resize, self.atten_layers, get_tokens=True)
            self.mutil_fake_B1_tokens = self.netPreViT(self.fake_B1_resize, self.atten_layers, get_tokens=True)
            if self.is_single_frame:
                self.mutil_real_A1_tokens = self.mutil_real_A0_tokens
                self.mutil_real_B1_tokens = self.mutil_real_B0_tokens
                self.mutil_fake_B1_tokens = self.mutil_fake_B0_tokens
                self.real_A1_resize = self.real_A0_resize
                self.fake_B1_resize = self.fake_B0_resize
                self.mask_A1 = self.mask_A0
                self.detail_B1 = self.detail_B0
                self.base_B1 = self.base_B0
                self.real_B1 = self.real_B0
                self.fake_B1 = self.fake_B0
                self.real_A1 = self.real_A0
                self.residual_B1 = self.residual_B0
                self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
                self.visual_names = ['real_A', 'mask_A', 'fake_B', 'real_B']
            else:
                self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'motion', 'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
                self.visual_names = ['real_A0', 'real_A1', 'mask_A0', 'mask_A1', 'fake_B0', 'fake_B1', 'real_B0', 'real_B1']
        elif self.is_single_frame:
            self.visual_names = ['real_A', 'mask_A', 'fake_B', 'real_B']
        else:
            self.visual_names = ['real_A0', 'real_A1', 'mask_A0', 'mask_A1', 'fake_B0', 'fake_B1', 'real_B0', 'real_B1']
            self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'motion', 'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
        if not self.is_single_frame:
            self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'motion', 'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
        else:
            self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'G_local', 'D_local_real', 'D_local_fake', 'detail_res', 'keep']
        self.loss_motion = 0.0

    def build_content_mask(self, frame):
        gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        sobel_x = frame.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = frame.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        lap = frame.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        laplacian = torch.abs(F.conv2d(gray, lap, padding=1))
        kernel = max(3, int(self.opt.mask_blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        mean = F.avg_pool2d(gray, kernel_size=kernel, stride=1, padding=kernel // 2)
        variance = F.avg_pool2d((gray - mean).pow(2), kernel_size=kernel, stride=1, padding=kernel // 2)
        response = self.normalize_map(gradient) + self.normalize_map(laplacian) + self.normalize_map(variance)
        response = self.normalize_map(response)
        flat = response.flatten(1)
        keep = max(1, int(flat.shape[1] * float(self.opt.mask_percent)))
        threshold = torch.topk(flat, keep, dim=1).values[:, -1].view(-1, 1, 1, 1)
        hard_mask = (response >= threshold).float()
        mask = response * hard_mask
        return self.normalize_map(mask)

    def normalize_map(self, value):
        value = value - value.amin(dim=(2, 3), keepdim=True)
        denom = value.amax(dim=(2, 3), keepdim=True)
        return value / (denom + 1e-6)

    def sample_local_patches(self, image, mask):
        patch_size = min(self.opt.local_patch_size, image.shape[2], image.shape[3])
        half = patch_size // 2
        mask_flat = mask[:, 0].reshape(mask.shape[0], -1)
        k = min(int(self.opt.local_patch_k), mask_flat.shape[1])
        _, indices = torch.topk(mask_flat, k=k, dim=1)
        patches = []
        for b in range(image.shape[0]):
            for idx in indices[b]:
                y = int(idx.item() // mask.shape[3])
                x = int(idx.item() % mask.shape[3])
                y0 = min(max(y - half, 0), image.shape[2] - patch_size)
                x0 = min(max(x - half, 0), image.shape[3] - patch_size)
                patches.append(image[b:b + 1, :, y0:y0 + patch_size, x0:x0 + patch_size])
        return torch.cat(patches, dim=0)

    def tokens_concat(self, origin_tokens, adjacent_size):
        adj_size = adjacent_size
        B, token_num, C = origin_tokens.shape[0], origin_tokens.shape[1], origin_tokens.shape[2]
        S = int(math.sqrt(token_num))
        if S * S != token_num:
            print('Error! Not a square!')
        token_map = origin_tokens.clone().reshape(B, S, S, C)
        cut_patch_list = []
        for i in range(0, S, adj_size):
            for j in range(0, S, adj_size):
                i_left = i
                i_right = i + adj_size + 1 if i + adj_size <= S else S + 1
                j_left = j
                j_right = j + adj_size if j + adj_size <= S else S + 1
                cut_patch = token_map[:, i_left:i_right, j_left:j_right, :]
                cut_patch = cut_patch.reshape(B, -1, C)
                cut_patch = torch.mean(cut_patch, dim=1, keepdim=True)
                cut_patch_list.append(cut_patch)
        return torch.cat(cut_patch_list, dim=1)

    def cat_results(self, origin_tokens, adj_size_list):
        res_list = [origin_tokens]
        for ad_s in adj_size_list:
            res_list.append(self.tokens_concat(origin_tokens, ad_s))
        return torch.cat(res_list, dim=1)

    def compute_D_loss(self):
        lambda_D_ViT = self.opt.lambda_D_ViT
        fake_B0_tokens = self.cat_results(self.mutil_fake_B0_tokens[self.opt.which_D_layer].detach(), self.opt.adj_size_list)
        fake_B1_tokens = self.cat_results(self.mutil_fake_B1_tokens[self.opt.which_D_layer].detach(), self.opt.adj_size_list)
        real_B0_tokens = self.cat_results(self.mutil_real_B0_tokens[self.opt.which_D_layer], self.opt.adj_size_list)
        real_B1_tokens = self.cat_results(self.mutil_real_B1_tokens[self.opt.which_D_layer], self.opt.adj_size_list)
        pre_fake0_ViT = self.netD_ViT(fake_B0_tokens)
        pre_fake1_ViT = self.netD_ViT(fake_B1_tokens)
        self.loss_D_fake_ViT = (self.criterionGAN(pre_fake0_ViT, False).mean() + self.criterionGAN(pre_fake1_ViT, False).mean()) * 0.5 * lambda_D_ViT
        pred_real0_ViT = self.netD_ViT(real_B0_tokens)
        pred_real1_ViT = self.netD_ViT(real_B1_tokens)
        self.loss_D_real_ViT = (self.criterionGAN(pred_real0_ViT, True).mean() + self.criterionGAN(pred_real1_ViT, True).mean()) * 0.5 * lambda_D_ViT
        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5
        return self.loss_D_ViT

    def compute_local_D_loss(self):
        fake_patches = torch.cat([
            self.sample_local_patches(self.fake_B0.detach(), self.mask_A0),
            self.sample_local_patches(self.fake_B1.detach(), self.mask_A1)
        ], dim=0)
        real_patches = torch.cat([
            self.sample_local_patches(self.real_B0, self.mask_A0),
            self.sample_local_patches(self.real_B1, self.mask_A1)
        ], dim=0)
        pred_fake = self.netD_local(fake_patches)
        pred_real = self.netD_local(real_patches)
        self.loss_D_local_fake = self.criterionGAN(pred_fake, False).mean() * self.opt.lambda_local_gan
        self.loss_D_local_real = self.criterionGAN(pred_real, True).mean() * self.opt.lambda_local_gan
        self.loss_D_local = 0.5 * (self.loss_D_local_fake + self.loss_D_local_real)
        return self.loss_D_local

    def compute_G_loss(self):
        if self.opt.lambda_GAN > 0.0:
            fake_B0_tokens = self.cat_results(self.mutil_fake_B0_tokens[self.opt.which_D_layer], self.opt.adj_size_list)
            fake_B1_tokens = self.cat_results(self.mutil_fake_B1_tokens[self.opt.which_D_layer], self.opt.adj_size_list)
            pred_fake0_ViT = self.netD_ViT(fake_B0_tokens)
            pred_fake1_ViT = self.netD_ViT(fake_B1_tokens)
            self.loss_G_GAN_ViT = (self.criterionGAN(pred_fake0_ViT, True) + self.criterionGAN(pred_fake1_ViT, True)) * 0.5 * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = 0.0

        if self.opt.lambda_global > 0.0 or self.opt.lambda_spatial > 0.0:
            self.loss_global, self.loss_spatial = self.calculate_attention_loss()
        else:
            self.loss_global, self.loss_spatial = 0.0, 0.0

        if self.opt.lambda_motion > 0.0 and not self.is_single_frame:
            self.loss_motion = 0.0
            for real_A0_tokens, real_A1_tokens, fake_B0_tokens, fake_B1_tokens in zip(self.mutil_real_A0_tokens, self.mutil_real_A1_tokens, self.mutil_fake_B0_tokens, self.mutil_fake_B1_tokens):
                A0_B1 = real_A0_tokens.bmm(fake_B1_tokens.permute(0, 2, 1))
                B0_A1 = fake_B0_tokens.bmm(real_A1_tokens.permute(0, 2, 1))
                cos_dis_global = F.cosine_similarity(A0_B1, B0_A1, dim=-1)
                self.loss_motion += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
        else:
            self.loss_motion = 0.0

        self.loss_G_local = self.compute_local_G_loss()
        self.loss_detail_res = self.opt.lambda_detail_res * 0.5 * (torch.mean(torch.abs(self.detail_B0)) + torch.mean(torch.abs(self.detail_B1)))
        keep0 = (1.0 - self.mask_A0) * (self.fake_B0 - self.base_B0)
        keep1 = (1.0 - self.mask_A1) * (self.fake_B1 - self.base_B1)
        self.loss_keep = self.opt.lambda_keep * 0.5 * (torch.mean(torch.abs(keep0)) + torch.mean(torch.abs(keep1)))
        self.loss_G = self.loss_G_GAN_ViT + self.loss_global + self.loss_spatial + self.loss_motion + self.loss_G_local + self.loss_detail_res + self.loss_keep
        return self.loss_G

    def compute_local_G_loss(self):
        fake_patches = torch.cat([
            self.sample_local_patches(self.fake_B0, self.mask_A0),
            self.sample_local_patches(self.fake_B1, self.mask_A1)
        ], dim=0)
        pred_fake = self.netD_local(fake_patches)
        self.loss_G_local = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_local_gan
        return self.loss_G_local

    def calculate_attention_loss(self):
        if self.opt.lambda_global > 0.0:
            loss_global = self.calculate_similarity(self.mutil_real_A0_tokens, self.mutil_fake_B0_tokens) + self.calculate_similarity(self.mutil_real_A1_tokens, self.mutil_fake_B1_tokens)
            loss_global *= 0.5
        else:
            loss_global = 0.0

        if self.opt.lambda_spatial > 0.0:
            tokens_cnt = 576
            local_nums = self.opt.local_nums
            local_id = np.random.permutation(tokens_cnt)
            local_id = local_id[:int(min(local_nums, tokens_cnt))]
            mutil_real_A0_local_tokens = self.netPreViT(self.real_A0_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length=self.opt.side_length)
            mutil_real_A1_local_tokens = self.netPreViT(self.real_A1_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length=self.opt.side_length)
            mutil_fake_B0_local_tokens = self.netPreViT(self.fake_B0_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length=self.opt.side_length)
            mutil_fake_B1_local_tokens = self.netPreViT(self.fake_B1_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length=self.opt.side_length)
            loss_spatial = self.calculate_similarity(mutil_real_A0_local_tokens, mutil_fake_B0_local_tokens) + self.calculate_similarity(mutil_real_A1_local_tokens, mutil_fake_B1_local_tokens)
            loss_spatial *= 0.5
        else:
            loss_spatial = 0.0

        return loss_global * self.opt.lambda_global, loss_spatial * self.opt.lambda_spatial

    def calculate_similarity(self, mutil_src_tokens, mutil_tgt_tokens):
        loss = 0.0
        for src_tokens, tgt_tokens in zip(mutil_src_tokens, mutil_tgt_tokens):
            src_tgt = src_tokens.bmm(tgt_tokens.permute(0, 2, 1))
            tgt_src = tgt_tokens.bmm(src_tokens.permute(0, 2, 1))
            cos_dis_global = F.cosine_similarity(src_tgt, tgt_src, dim=-1)
            loss += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
        return loss / len(self.atten_layers)

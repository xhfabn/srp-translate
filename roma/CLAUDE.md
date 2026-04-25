# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROMA (Region Similarity Matching) is a PyTorch implementation for unpaired nighttime infrared to daytime visible video translation (ACM MM'22). It is built on top of the [CycleGAN/Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation) frameworks.

## Commands

### Training

**Video mode** (two adjacent frames concatenated as input):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataroot /path/to/dataset \
  --name ROMA_name \
  --dataset_mode unaligned_double \
  --no_flip \
  --local_nums 64 \
  --display_env ROMA_env \
  --model roma \
  --side_length 7 \
  --lambda_spatial 5.0 \
  --lambda_global 5.0 \
  --lambda_motion 1.0 \
  --atten_layers 1,3,5 \
  --lr 0.00001
```

**Image mode** (single image input):
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataroot /path/to/dataset \
  --name ROMA_name \
  --dataset_mode unaligned \
  --local_nums 64 \
  --model roma \
  --side_length 7 \
  --lambda_spatial 5.0 \
  --lambda_global 5.0 \
  --atten_layers 1,3,5 \
  --lr 0.00001
```

### Testing / Inference
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataroot /path/to/test_dataset \
  --checkpoints_dir ./checkpoints \
  --name experiment_name \
  --model roma_single \
  --num_test 10000 \
  --epoch latest
```

Outputs are saved under `./results/<name>/`.

## Dataset Structure

```
dataset_root/
  trainA/   # nighttime infrared frames
  trainB/   # daytime visible frames
```

For video mode (`unaligned_double`), each sample is a concatenation of two adjacent frames along the width axis. For image mode (`unaligned`), each sample is a single image.

## Architecture

### Key Design Choices

- **Generator (`netG`)**: ResNet-based (default `resnet_9blocks`), translates domain A → domain B.
- **Discriminator (`netD_ViT`)**: `MLPDiscriminator` applied to token features from a frozen pretrained ViT (`vit_base_patch16_384` from `timm/`). The ViT is loaded with `timm.create_model(..., pretrained=True)` and is never trained.
- **Cross-Similarity**: Computed across domains using intermediate ViT token features at layers specified by `--atten_layers` (default `1,3,5`). This is the core novelty — matching structural regions cross-domain to guide the generator.

### Loss Terms

| Flag | Loss | Purpose |
|---|---|---|
| `--lambda_global` | Global Structural Consistency | ViT token similarity across full frame |
| `--lambda_spatial` | Local Structural Consistency | patch-level ViT token similarity |
| `--lambda_motion` | Temporal Consistency (video only) | consistency between adjacent frames A0/A1 |
| `--lambda_GAN` | GAN loss | standard adversarial loss |

### Options System

Options are resolved dynamically via `argparse` in `options/`. Each model class implements `modify_commandline_options(parser, is_train)` to inject its own flags. The resolution order is: `base_options.py` → `train_options.py` / `test_options.py` → model-specific options → dataset-specific options.

Key global flags: `--checkpoints_dir` (default `./checkpoints`), `--gpu_ids`, `--batch_size`, `--load_size` / `--crop_size`.

### Model Files

- `models/roma_model.py` — main training model (video, dual-frame input)
- `models/roma_single_model.py` — inference-only model (single frame)
- `models/networks.py` — generator, discriminator, and `MLPDiscriminator` definitions
- `models/patchnce.py` — PatchNCE loss (inherited from CUT)

### `timm/`

A vendored local copy of `pytorch-image-models`. The ViT (`vit_base_patch16_384`) is used as a frozen feature extractor — modifications here affect discriminator feature extraction.

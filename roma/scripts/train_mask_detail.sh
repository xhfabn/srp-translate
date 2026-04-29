#!/usr/bin/env bash
# =============================================================================
# 训练脚本：roma_mask_detail 模型
#
# 用法：
#   视频模式（默认）:
#     bash scripts/train_mask_detail.sh /path/to/dataset
#
#   图像模式:
#     bash scripts/train_mask_detail.sh /path/to/dataset image
#
# 数据集目录结构要求：
#   视频模式 (unaligned_double):
#     dataset/
#       trainA/   <- 每张图为 512x256，左半帧0 + 右半帧1 横向拼接
#       trainB/
#       testA/    (可选，没有自动找 valA)
#       testB/
#
#   图像模式 (unaligned):
#     dataset/
#       trainA/   <- 普通单帧图像，256x256
#       trainB/
#       testA/
#       testB/
# =============================================================================

set -e

DATAROOT="${1:?请提供数据集路径，例如: bash scripts/train_mask_detail.sh /data/infrared}"
MODE="${2:-video}"   # video | image，默认 video

GPU="0"
NAME="roma_mask_detail_$(date +%Y%m%d_%H%M)"
CHECKPOINTS_DIR="./checkpoints"

# 公共参数
COMMON_ARGS=(
    --dataroot        "$DATAROOT"
    --name            "$NAME"
    --checkpoints_dir "$CHECKPOINTS_DIR"
    --model           roma_mask_detail
    --netG            resnet_9blocks_mask_detail
    --no_flip
    --lr              0.00001
    --local_nums      64
    --atten_layers    1,3,5
    --side_length     7
    --lambda_global   5.0
    --lambda_spatial  5.0
    --lambda_GAN      1.0
    --lambda_local_gan  0.5
    --lambda_detail_res 0.1
    --lambda_keep     5.0
    --detail_scale    0.1
    --mask_percent    0.25
    --local_patch_size 96
    --local_patch_k   4
    --display_env     "$NAME"
    --n_epochs        100
    --n_epochs_decay  100
)

if [ "$MODE" = "image" ]; then
    # 图像模式：单帧，不用 motion loss
    echo "[INFO] 图像模式 (unaligned)"
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py \
        "${COMMON_ARGS[@]}"  \
        --dataset_mode unaligned \
        --lambda_motion 0.0
else
    # 视频模式：双帧拼接，启用 motion loss
    echo "[INFO] 视频模式 (unaligned_double)"
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py \
        "${COMMON_ARGS[@]}"  \
        --dataset_mode unaligned_double \
        --lambda_motion 1.0
fi

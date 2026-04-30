#!/usr/bin/env bash
# =============================================================================
# 并行训练 IRVI 数据集（single 图像模式 + triplet 视频模式）
#
# Checkpoint 保存路径：
#   checkpoints/roma_mask_detail/<scene>/
#
# 场景列表：
#   IRVI_single_monitor  -> /root/dataset_archives/IRVI/single/rooftop_total  (5080张)
#   IRVI_single_traffic  -> /root/dataset_archives/IRVI/single/traffic         (16999张)
#   IRVI_triplet_monitor -> /root/dataset_archives/IRVI/triplet/rooftop_total  (1686组)
#   IRVI_triplet_traffic -> /root/dataset_archives/IRVI/triplet/traffic         (5666组)
#
# GPU: RTX 3090 (49GB)，每进程约 8GB，4 个场景并行约占 32GB。
#
# 用法（在 roma 项目根目录下执行）：
#   bash scripts/train_irvi.sh
#   # 后台运行：
#   nohup bash scripts/train_irvi.sh > logs/irvi/main.log 2>&1 &
#
# 查看进度：tail -f logs/irvi/IRVI_single_traffic.log
# 查看显存：watch -n5 nvidia-smi
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/logs/irvi"
CHECKPOINTS_DIR="$REPO_DIR/checkpoints/roma_mask_detail"
mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR"

# 场景列表：(数据路径, 模式, 场景名)
DATAPATHS=(
    "/root/dataset_archives/IRVI/single/rooftop_total"
    "/root/dataset_archives/IRVI/single/traffic"
    "/root/dataset_archives/IRVI/triplet/rooftop_total"
    "/root/dataset_archives/IRVI/triplet/traffic"
)
MODES=(
    "image"
    "image"
    "video"
    "video"
)
NAMES=(
    "IRVI_single_monitor"
    "IRVI_single_traffic"
    "IRVI_triplet_monitor"
    "IRVI_triplet_traffic"
)

echo "========================================================"
echo " IRVI 并行训练 (roma_mask_detail)"
echo " 场景数: ${#DATAPATHS[@]}，GPU=0 (RTX 3090, 49GB)"
echo " Checkpoints: $CHECKPOINTS_DIR/<scene>/"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

PIDS=()

for i in "${!DATAPATHS[@]}"; do
    DATAROOT="${DATAPATHS[$i]}"
    MODE="${MODES[$i]}"
    NAME="${NAMES[$i]}"
    LOG_FILE="$LOG_DIR/${NAME}.log"

    echo "[启动] $NAME  模式=$MODE"
    echo "       数据=$DATAROOT"
    echo "       日志=$LOG_FILE"

    (
        cd "$REPO_DIR"
        CUDA_VISIBLE_DEVICES=0 python3 train.py \
            --dataroot        "$DATAROOT" \
            --name            "$NAME" \
            --checkpoints_dir "$CHECKPOINTS_DIR" \
            --model           roma_mask_detail \
            --netG            resnet_9blocks_mask_detail \
            --no_flip \
            --lr              0.00001 \
            --local_nums      64 \
            --atten_layers    1,3,5 \
            --side_length     7 \
            --lambda_global   5.0 \
            --lambda_spatial  5.0 \
            --lambda_GAN      1.0 \
            --lambda_local_gan  0.5 \
            --lambda_detail_res 0.1 \
            --lambda_keep     5.0 \
            --detail_scale    0.1 \
            --mask_percent    0.25 \
            --local_patch_size 96 \
            --local_patch_k   4 \
            --display_env     "$NAME" \
            --n_epochs        100 \
            --n_epochs_decay  100 \
            --dataset_mode    "$([ "$MODE" = "image" ] && echo unaligned || echo unaligned_double)" \
            --lambda_motion   "$([ "$MODE" = "image" ] && echo 0.0 || echo 1.0)" \
            > "$LOG_FILE" 2>&1
    ) &

    PIDS+=($!)
    sleep 3   # 错开启动，避免同时加载 ViT 权重抢显存
done

echo ""
echo "所有场景已后台启动，PID: ${PIDS[*]}"
echo "查看进度示例："
echo "  tail -f $LOG_DIR/IRVI_single_traffic.log"
echo "  watch -n5 nvidia-smi"
echo ""
echo "等待所有进程结束..."

FAIL=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    NAME="${NAMES[$i]}"
    if wait "$PID"; then
        echo "[完成] $NAME (PID=$PID)"
    else
        echo "[失败] $NAME (PID=$PID) 退出码=$?"
        FAIL=1
    fi
done

echo ""
echo "========================================================"
echo " 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
[ "$FAIL" -eq 0 ] && echo " 全部完成！" || echo " 部分场景失败，请检查日志。"
echo "========================================================"

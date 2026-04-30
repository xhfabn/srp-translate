#!/usr/bin/env bash
# =============================================================================
# 并行训练所有 single 场景，使用 roma_single 模型
# 学习率策略：
#   总训练 100 epochs
#   前 50 epochs 保持 lr=1e-5
#   后 50 epochs 线性衰减到 0
#
# Checkpoint 保存路径：
#   checkpoints/roma_single/<scene>/
#
# 场景（共5个，全部并行，约占 5x7=35GB，RTX 3090 49GB 充裕）：
#   IRVI_single_monitor     -> /root/dataset_archives/IRVI/single/rooftop_total                    (trainA=5080)
#   IRVI_single_traffic     -> /root/dataset_archives/IRVI/single/traffic                          (trainA=16999)
#   ICL_single_monitor      -> /root/dataset_archives/InfraredCity-Lite/Single/Monitor_std         (trainA=5612)
#   ICL_single_traffic_city -> /root/dataset_archives/InfraredCity-Lite/Single/Traffic/City_std    (trainA=5538)
#   ICL_single_traffic_hwy  -> /root/dataset_archives/InfraredCity-Lite/Single/Traffic/Highway_std (trainA=4412)
#
# 用法（在 roma 项目根目录下执行）：
#   bash scripts/train_all_roma_single.sh
#   # 后台运行：
#   nohup bash scripts/train_all_roma_single.sh > logs/roma_single/main.log 2>&1 &
#
# 查看进度：tail -f logs/roma_single/<name>.log
# 查看显存：watch -n5 nvidia-smi
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/roma_single"
CHECKPOINTS_DIR="$REPO_DIR/checkpoints/roma_single"
mkdir -p "$LOG_DIR" "$CHECKPOINTS_DIR"

DATAPATHS=(
    "/root/dataset_archives/IRVI/single/rooftop_total"
    "/root/dataset_archives/IRVI/single/traffic"
    "/root/dataset_archives/InfraredCity-Lite/Single/Monitor_std"
    "/root/dataset_archives/InfraredCity-Lite/Single/Traffic/City_std"
    "/root/dataset_archives/InfraredCity-Lite/Single/Traffic/Highway_std"
)
NAMES=(
    "IRVI_single_monitor"
    "IRVI_single_traffic"
    "ICL_single_monitor"
    "ICL_single_traffic_city"
    "ICL_single_traffic_hwy"
)

echo "========================================================"
echo " 全场景并行训练 (roma_single)"
echo " 场景数: ${#DATAPATHS[@]}，GPU=0 (RTX 3090, 49GB)"
echo " Checkpoints: $CHECKPOINTS_DIR/<scene>/"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

PIDS=()

for i in "${!DATAPATHS[@]}"; do
    DATAROOT="${DATAPATHS[$i]}"
    NAME="${NAMES[$i]}"
    LOG_FILE="$LOG_DIR/${NAME}.log"

    echo "[启动] $NAME"
    echo "       数据=$DATAROOT"
    echo "       日志=$LOG_FILE"

    (
        cd "$REPO_DIR"
        CUDA_VISIBLE_DEVICES=0 python3 train.py \
            --dataroot        "$DATAROOT" \
            --name            "$NAME" \
            --checkpoints_dir "$CHECKPOINTS_DIR" \
            --model           roma_single \
            --dataset_mode    unaligned \
            --no_flip \
            --lr              0.00001 \
            --local_nums      64 \
            --atten_layers    1,3,5 \
            --side_length     7 \
            --lambda_GAN      1.0 \
            --lambda_global   5.0 \
            --lambda_spatial  5.0 \
            --display_id      -1 \
            --no_html \
            --lr_policy       linear \
            --n_epochs        50 \
            --n_epochs_decay  50 \
            > "$LOG_FILE" 2>&1
    ) &

    PIDS+=($!)
    sleep 5   # 错开启动，避免同时加载 ViT 权重抢显存
done

echo ""
echo "所有场景已后台启动，PID: ${PIDS[*]}"
echo ""
echo "查看各场景进度："
for NAME in "${NAMES[@]}"; do
    echo "  tail -f $LOG_DIR/${NAME}.log"
done
echo ""
echo "查看显存：watch -n5 nvidia-smi"
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

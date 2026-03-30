#!/bin/bash

# 从早停的第二阶段检查点继续训练的脚本
# 专门用于从 autodl-tmp/dinolbh/logs0624/DINO/TUSZ/090816_two_stage/checkpoint0020.pth 继续训练
# 使用方法: bash scripts/continue_stage2_training.sh [num_gpus]

# 获取参数
NUM_GPUS=${1:-1}  # 默认使用1张GPU

# 指定的检查点路径
CHECKPOINT_PATH="/root/autodl-tmp/checkpoint0005.pth"

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请确认路径是否正确，或者检查点文件是否已被移动"
    exit 1
fi

echo "=========================================="
echo "从早停检查点继续第二阶段训练"
echo "检查点路径: $CHECKPOINT_PATH"
echo "使用GPU数量: $NUM_GPUS"
echo "=========================================="

# 创建输出目录（基于当前时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/continue_stage2_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "输出目录: $OUTPUT_DIR"
echo ""

# 根据GPU数量选择启动方式
if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡继续第二阶段训练..."
    # 多卡训练
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        --use_env \
        main.py \
        --output_dir "$OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_slice_400 \
        --tusz_txt_dir /root/autodl-tmp/dataset_slice_400 \
        --seed 42 \
        --save_log \
        --two_stage_training \
        --start_stage2_from_checkpoint "$CHECKPOINT_PATH" \
        --stage1_epochs 50 \
        --stage2_epochs 80 \
        --stage1_patience 10 \
        --stage2_patience 20 \
        --stage2_lr_factor 0.1 \
        --tusz_downsample_times 2.0 \
        --gradient_accumulation_steps 2 \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Continue Stage2 training from checkpoint0020 with multi-GPU"
else
    echo "使用单卡继续第二阶段训练..."
    # 单卡训练
    python main.py \
        --output_dir "$OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_lbhdataset \
        --tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
        --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
        --device cuda:0 \
        --seed 42 \
        --save_log \
        --two_stage_training \
        --start_stage2_from_checkpoint "$CHECKPOINT_PATH" \
        --stage1_epochs 50 \
        --stage2_epochs 80 \
        --stage1_patience 10 \
        --stage2_patience 20 \
        --stage2_lr_factor 0.1 \
        --tusz_downsample_times 2.0 \
        --gradient_accumulation_steps 2 \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Continue Stage2 training from checkpoint0020 with single-GPU"
fi

echo ""
echo "=========================================="
echo "第二阶段继续训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $OUTPUT_DIR/info.txt"
echo "训练日志: $OUTPUT_DIR/log.txt"
echo "=========================================="
echo ""
echo "使用说明:"
echo "1. 单卡训练: bash scripts/continue_stage2_training.sh"
echo "2. 多卡训练: bash scripts/continue_stage2_training.sh 2"
echo "3. 查看训练日志: tail -f $OUTPUT_DIR/info.txt"
echo "4. 监控训练进度: watch -n 10 'tail -20 $OUTPUT_DIR/info.txt'"
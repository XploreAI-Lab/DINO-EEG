#!/bin/bash

# 从指定检查点开始第二阶段训练的脚本
# 使用方法: bash scripts/DINO_stage2_from_checkpoint.sh [checkpoint_path] [num_gpus]

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 请提供检查点路径"
    echo "使用方法: bash $0 [checkpoint_path] [num_gpus]"
    echo "示例: bash $0 /autodl-tmp/dinolbh/logs0624/DINO/TUSZ/two_stage_training_memory_optimized/checkpoint0020.pth 2"
    exit 1
fi

# 获取参数
CHECKPOINT_PATH=$1
NUM_GPUS=${2:-1}  # 默认使用1张GPU

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "从检查点开始第二阶段训练"
echo "检查点路径: $CHECKPOINT_PATH"
echo "使用GPU数量: $NUM_GPUS"

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/stage2_from_checkpoint_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "输出目录: $OUTPUT_DIR"

# 根据GPU数量选择启动方式
if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡第二阶段训练"
    # 多卡训练
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        --use_env \
        main.py \
        --output_dir "$OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_lbhdataset \
        --tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
        --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
        --seed 42 \
        --save_log \
        --two_stage_training \
        --start_stage2_from_checkpoint "$CHECKPOINT_PATH" \
        --stage1_epochs 50 \
        --stage2_epochs 50 \
        --stage1_patience 10 \
        --stage2_patience 15 \
        --stage2_lr_factor 0.1 \
        --tusz_downsample_times 2.0 \
        --gradient_accumulation_steps 2 \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Stage2 training from checkpoint with multi-GPU"
else
    echo "使用单卡第二阶段训练"
    # 单卡训练
    python main.py \
        --output_dir "$OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
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
        --stage2_epochs 50 \
        --stage1_patience 10 \
        --stage2_patience 15 \
        --stage2_lr_factor 0.1 \
        --tusz_downsample_times 2.0 \
        --gradient_accumulation_steps 2 \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Stage2 training from checkpoint with single-GPU"
fi

echo ""
echo "第二阶段训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $OUTPUT_DIR/info.txt"
echo "训练日志: $OUTPUT_DIR/log.txt"

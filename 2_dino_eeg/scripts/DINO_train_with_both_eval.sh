#!/bin/bash

# 启用同时评估验证集和测试集的训练脚本
# 使用方法: bash scripts/DINO_train_with_both_eval.sh [num_gpus]

# 获取参数
NUM_GPUS=${1:-1}  # 默认使用1张GPU

echo "开始训练，每轮同时评估验证集和测试集"
echo "使用GPU数量: $NUM_GPUS"

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/training_with_both_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "输出目录: $OUTPUT_DIR"

# 根据GPU数量选择启动方式
if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡训练模式"
    # 多卡训练
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29501 \
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
        --eval_both_sets \
        --two_stage_training \
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
        --note "Training with both validation and test set evaluation (multi-GPU)"
else
    echo "使用单卡训练模式"
    # 单卡训练
    python main.py \
        --output_dir "$OUTPUT_DIR" \
        --device cuda:0 \
        -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_lbhdataset \
        --tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
        --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
        --seed 42 \
        --save_log \
        --eval_both_sets \
        --two_stage_training \
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
        --note "Training with both validation and test set evaluation (single-GPU)"
fi

echo ""
echo "训练完成!"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $OUTPUT_DIR/info.txt"
echo "训练日志: $OUTPUT_DIR/log.txt"
echo ""
echo "每轮训练都会同时评估验证集和测试集，结果保存在日志中"
echo "日志格式:"
echo "  - val_eval_bbox: 验证集mAP"
echo "  - test_eval_bbox: 测试集mAP"
echo "  - 其他指标也会分别标记为val_和test_前缀"

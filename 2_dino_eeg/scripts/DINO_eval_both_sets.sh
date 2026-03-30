#!/bin/bash

# 同时评估验证集和测试集的脚本
# 使用方法: bash scripts/DINO_eval_both_sets.sh [checkpoint_path] [num_gpus]

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 请提供模型检查点路径"
    echo "使用方法: bash $0 [checkpoint_path] [num_gpus]"
    echo "示例: bash $0 /path/to/checkpoint.pth 1"
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

echo "开始同时评估验证集和测试集"
echo "检查点路径: $CHECKPOINT_PATH"
echo "使用GPU数量: $NUM_GPUS"

# 创建评估输出目录
EVAL_OUTPUT_DIR="./evaluation_both_sets_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "评估结果将保存到: $EVAL_OUTPUT_DIR"

# 根据GPU数量选择启动方式
if [ $NUM_GPUS -gt 1 ]; then
    echo "使用多卡评估模式"
    # 多卡评估
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29502 \
        --use_env \
        main.py \
        --eval \
        --eval_both_sets \
        --resume "$CHECKPOINT_PATH" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_lbhdataset \
        --tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
        --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
        --device cuda \
        --seed 42 \
        --save_log \
        --save_results \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Evaluation on both validation and test sets (multi-GPU)"
else
    echo "使用单卡评估模式"
    # 单卡评估
    python main.py \
        --eval \
        --eval_both_sets \
        --resume "$CHECKPOINT_PATH" \
        --output_dir "$EVAL_OUTPUT_DIR" \
        -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
        --dataset tusz \
        --data_dir /root/autodl-tmp/dataset_lbhdataset \
        --tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
        --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
        --device cuda:0 \
        --seed 42 \
        --save_log \
        --save_results \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0 \
        backbone_dir=/root/autodl-tmp/ \
        --note "Evaluation on both validation and test sets (single-GPU)"
fi

echo ""
echo "评估完成!"
echo "结果文件位置:"
echo "  - 评估日志: $EVAL_OUTPUT_DIR/info.txt"
echo "  - 测试集结果: $EVAL_OUTPUT_DIR/results.bbox.json (test set)"
echo "  - 真实标签: $EVAL_OUTPUT_DIR/ground_truth.bbox.json"
echo "  - 配置文件: $EVAL_OUTPUT_DIR/config_args_all.json"
echo ""
echo "注意: 日志中会显示两个集合的对比结果:"
echo "  - test_eval_bbox: 测试集mAP"
echo "  - val_eval_bbox: 验证集mAP"

# 显示结果摘要
if [ -f "$EVAL_OUTPUT_DIR/info.txt" ]; then
    echo ""
    echo "=== 评估结果摘要 ==="
    echo "查找评估结果对比:"
    grep -A 5 "评估结果对比" "$EVAL_OUTPUT_DIR/info.txt" || echo "未找到结果对比信息"
fi

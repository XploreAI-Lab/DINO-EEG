#!/bin/bash

# 单卡评估脚本 - 简化版本
# 使用方法: bash scripts/DINO_eval_swin_tusz_single.sh [checkpoint_path]
# 示例: bash scripts/DINO_eval_swin_tusz_single.sh ./logs/checkpoint.pth

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 请提供模型检查点路径"
    echo "使用方法: bash $0 [checkpoint_path]"
    echo "示例: bash $0 ./logs/checkpoint.pth"
    exit 1
fi

# 获取参数
CHECKPOINT_PATH=$1

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "启动单卡模型评估"
echo "检查点路径: $CHECKPOINT_PATH"

# 创建评估输出目录
EVAL_OUTPUT_DIR="./evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "评估结果将保存到: $EVAL_OUTPUT_DIR"

# 单卡评估
python main.py \
    --eval \
    --resume "$CHECKPOINT_PATH" \
    --output_dir "$EVAL_OUTPUT_DIR" \
    -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
    --dataset tusz \
    --data_dir /root/autodl-tmp/TUSZ_sliced_30_data \
    --tusz_txt_dir /root/autodl-tmp/TUSZ_sliced_30_data \
    --device cuda:0 \
    --seed 42 \
    --save_log \
    --save_results \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0 \
    backbone_dir=/root/autodl-tmp/ \
    --note "Evaluation on TUSZ test set"

echo ""
echo "评估完成!"
echo "结果文件位置:"
echo "  - 评估日志: $EVAL_OUTPUT_DIR/info.txt"
echo "  - 检测结果: $EVAL_OUTPUT_DIR/results.bbox.json (eval数据集)"
echo "  - 真实标签: $EVAL_OUTPUT_DIR/ground_truth.bbox.json"
echo "  - 配置文件: $EVAL_OUTPUT_DIR/config_args_all.json"

# 显示结果摘要
if [ -f "$EVAL_OUTPUT_DIR/info.txt" ]; then
    echo ""
    echo "=== 评估结果摘要 ==="
    echo "最后20行关键指标:"
    tail -20 "$EVAL_OUTPUT_DIR/info.txt"
fi

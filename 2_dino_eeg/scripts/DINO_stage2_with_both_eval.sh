#!/bin/bash

# 从指定检查点开始第二阶段训练，并启用同时评估验证集和测试集
# 使用方法: bash scripts/DINO_stage2_with_both_eval.sh [checkpoint_path]
# bash scripts/DINO_stage2_with_both_eval.sh /autodl-tmp/dinolbh/logs0624/DINO/TUSZ/two_stage_training_memory_optimized/checkpoint0020.pth

# 检查参数
if [ $# -lt 1 ]; then
    echo "错误: 请提供检查点路径"
    echo "使用方法: bash $0 [checkpoint_path]"
    echo "示例: bash $0 /autodl-tmp/dinolbh/logs0624/DINO/TUSZ/two_stage_training_memory_optimized/checkpoint0020.pth"
    exit 1
fi

# 获取参数
CHECKPOINT_PATH=$1

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "🚀 从检查点开始第二阶段训练"
echo "📊 每轮同时评估验证集和测试集"
echo "=========================================="
echo "检查点路径: $CHECKPOINT_PATH"
echo "训练模式: 单卡训练"
echo "评估模式: 双重评估 (验证集 + 测试集)"

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/stage2_both_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 开始训练
echo "🔄 开始第二阶段训练..."
echo "   从检查点: $CHECKPOINT_PATH"
echo "   训练模式: 两阶段训练 (直接第二阶段)"
echo "   评估模式: 每轮同时评估验证集和测试集"
echo ""

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
    --save_results \
    --two_stage_training \
    --start_stage2_from_checkpoint "$CHECKPOINT_PATH" \
    --eval_both_sets \
    --stage1_epochs 50 \
    --stage2_epochs 50 \
    --stage1_patience 10 \
    --stage2_patience 31 \
    --stage2_lr_factor 0.1 \
    --tusz_downsample_times 2.0 \
    --gradient_accumulation_steps 2 \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0 \
    backbone_dir=/root/autodl-tmp/ \
    --note "Stage2 training from checkpoint with both validation and test set evaluation"

echo ""
echo "=========================================="
echo "✅ 训练完成!"
echo "=========================================="
echo "📁 输出目录: $OUTPUT_DIR"
echo "📄 日志文件: $OUTPUT_DIR/info.txt"
echo "📊 训练日志: $OUTPUT_DIR/log.txt"
echo ""

echo "🔍 训练过程中每轮都会显示:"
echo "   Epoch X: 开始验证集评估..."
echo "   Epoch X: 开始测试集评估..."
echo "   Epoch X 性能对比:"
echo "     验证集 mAP: 0.xxxx"
echo "     测试集 mAP: 0.xxxx"
echo "     差异: ±0.xxxx"
echo ""

echo "📈 日志中包含的指标:"
echo "   - val_eval_bbox: 验证集mAP"
echo "   - test_eval_bbox: 测试集mAP"
echo "   - val_loss_ce: 验证集分类损失"
echo "   - test_loss_ce: 测试集分类损失"
echo "   - val_loss_bbox: 验证集边界框损失"
echo "   - test_loss_bbox: 测试集边界框损失"
echo "   - 其他指标也会分别标记为val_和test_前缀"
echo ""

echo "💡 后续分析建议:"
echo "   1. 使用 analyze_both_eval_logs.py 分析训练日志"
echo "   2. 查看性能差异趋势，检测过拟合"
echo "   3. 基于测试集性能选择最佳模型"
echo ""

# 检查是否生成了日志文件
if [ -f "$OUTPUT_DIR/log.txt" ]; then
    echo "📊 日志文件预览 (最后5行):"
    echo "----------------------------------------"
    tail -5 "$OUTPUT_DIR/log.txt"
    echo "----------------------------------------"
else
    echo "⚠️  未找到训练日志文件"
fi

echo ""
echo "🎯 脚本执行完成!"

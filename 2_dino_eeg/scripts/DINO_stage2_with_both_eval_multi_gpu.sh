#!/bin/bash

# 从指定检查点开始第二阶段训练，并启用同时评估验证集和测试集（多卡版本）
# 使用方法: bash scripts/DINO_stage2_with_both_eval_multi_gpu.sh [checkpoint_path] [num_gpus]

# 检查参数
if [ $# -lt 2 ]; then
    echo "错误: 请提供检查点路径和GPU数量"
    echo "使用方法: bash $0 [checkpoint_path] [num_gpus]"
    echo "示例: bash $0 /autodl-tmp/dinolbh/logs0624/DINO/TUSZ/two_stage_training_memory_optimized/checkpoint0010.pth 4"
    exit 1
fi

# 获取参数
CHECKPOINT_PATH=$1
NUM_GPUS=$2

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

# 检查GPU数量
if [ $NUM_GPUS -lt 1 ]; then
    echo "错误: GPU数量必须大于0"
    exit 1
fi

echo "=========================================="
echo "🚀 从检查点开始第二阶段训练（多卡版本）"
echo "📊 每轮同时评估验证集和测试集"
echo "=========================================="
echo "检查点路径: $CHECKPOINT_PATH"
echo "使用GPU数量: $NUM_GPUS"
echo "训练模式: 多卡分布式训练"
echo "评估模式: 双重评估 (验证集 + 测试集)"

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/stage2_both_eval_multi_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 开始多卡训练
echo "🔄 开始多卡第二阶段训练..."
echo "   从检查点: $CHECKPOINT_PATH"
echo "   训练模式: 两阶段训练 (直接第二阶段)"
echo "   评估模式: 每轮同时评估验证集和测试集"
echo "   分布式训练: $NUM_GPUS 张GPU"
echo ""

# 使用torch.distributed.launch启动多卡训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
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
    --save_results \
    --two_stage_training \
    --start_stage2_from_checkpoint "$CHECKPOINT_PATH" \
    --eval_both_sets \
    --stage1_epochs 50 \
    --stage2_epochs 50 \
    --stage1_patience 10 \
    --stage2_patience 50 \
    --stage2_lr_factor 0.1 \
    --tusz_downsample_times 2.0 \
    --gradient_accumulation_steps 2 \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0 \
    backbone_dir=/root/autodl-tmp/ \
    --note "Stage2 training from checkpoint with both validation and test set evaluation (multi-GPU)"

echo ""
echo "=========================================="
echo "✅ 多卡训练完成!"
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

echo "🚀 多卡训练特性:"
echo "   - 训练加速: 使用 $NUM_GPUS 张GPU并行训练"
echo "   - 分布式评估: 验证集和测试集评估也使用多卡加速"
echo "   - 梯度同步: 自动处理多卡间的梯度同步"
echo "   - 内存优化: 每张GPU处理较小的批次，减少显存占用"
echo ""

echo "💡 后续分析建议:"
echo "   1. 使用 analyze_both_eval_logs.py 分析训练日志"
echo "   2. 查看性能差异趋势，检测过拟合"
echo "   3. 基于测试集性能选择最佳模型"
echo "   4. 对比多卡训练与单卡训练的性能差异"
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
echo "🎯 多卡脚本执行完成!"
echo ""
echo "📊 性能提升预期:"
echo "   - 训练速度: 约 ${NUM_GPUS}x 加速"
echo "   - 批次大小: 有效批次大小 = ${NUM_GPUS} × 单卡批次大小"
echo "   - 显存使用: 每张GPU显存占用减少"
echo "   - 评估速度: 验证集和测试集评估也获得加速"

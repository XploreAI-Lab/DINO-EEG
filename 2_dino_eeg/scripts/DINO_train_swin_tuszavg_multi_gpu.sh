#!/bin/bash

# 多卡训练脚本
# 使用方法: bash scripts/DINO_train_swin_tusz_multi_gpu.sh [num_gpus]

# 获取GPU数量，默认为2
NUM_GPUS=${1:-4}

echo "启动多卡训练，使用 $NUM_GPUS 张GPU"

# 使用torch.distributed.launch启动多卡训练
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --use_env \
    main.py \
    --output_dir /root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/091007_avg_two_stage \
    -c config/DINO/DINO_4scale_swin_tusz_multi_gpu.py \
    --dataset tusz \
    --data_dir /root/autodl-tmp/TUSZ_avg_stft \
    --tusz_txt_dir /root/autodl-tmp/TUSZ_avg_stft \
    --tusz_label_dir /root/autodl-tmp/TUSZ_avg_stft_2label \
    --seed 42 \
    --save_log \
    --two_stage_training \
    --stage1_epochs 50 \
    --stage2_epochs 100 \
    --stage1_patience 10 \
    --stage2_patience 15 \
    --stage2_lr_factor 0.1 \
    --tusz_downsample_times 2.0 \
    --options dn_scalar=100 embed_init_tgt=TRUE \
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
    dn_box_noise_scale=1.0 \
    backbone_dir=/root/autodl-tmp/ \
    --note "Two-stage training with multi-GPU support (memory optimized)"

echo "多卡训练完成"

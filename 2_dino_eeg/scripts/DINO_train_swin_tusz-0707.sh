python main.py \
	--output_dir /root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/0707mix \
	-c config/DINO/DINO_4scale_swin_tusz.py \
	--dataset tusz \
	--data_dir /root/autodl-tmp/dataset_lbhdataset \
	--tusz_txt_dir /root/autodl-tmp/dataset_seiztxt \
    --tusz_label_dir /root/autodl-tmp/dataset_bilabel \
    --resume "/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/0707mix/checkpoint0020.pth" \
	--device "cuda:0" \
	--seed 42 \
	--save_log \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	backbone_dir=/root/autodl-tmp/ \
	--note "your note"
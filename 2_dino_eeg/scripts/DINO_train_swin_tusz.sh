python main.py \
	--output_dir /root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/0717Seizeit3_re110 \
	-c config/DINO/DINO_4scale_swin_tusz.py \
	--dataset tusz \
	--data_dir /root/autodl-tmp/SeizeIT2_stft_sliced \
	--tusz_txt_dir /root/autodl-tmp/SeizeIT2_stft_sliced \
    --tusz_label_dir /root/autodl-tmp/SeizeIT2_3label_sliced \
    --resume "/root/autodl-tmp/dinolbh/logs0624/DINO/TUSZ/0717Seizeit3_re35/checkpoint0110.pth" \
	--device "cuda:0" \
	--seed 42 \
	--save_log \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	backbone_dir=/root/autodl-tmp/ \
	--note "your note"
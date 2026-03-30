device=$1
python main.py \
	--output_dir /path/to/your/logs/DINO/TUSZ/foo \
	-c config/DINO/DINO_4scale_swin_tusz.py \
	--dataset tusz \
	--data_dir /path/to/your/tusz/STFT_Data_Dir \
	--tusz_txt_dir /path/to/your/tusz/TXT_file_Dir \
	--pretrain_model_path /path/to/your/checkpoints/Swin-MS4-bi-seed42-sns-crop-ft/checkpoint_best_regular.pth \
	--device $device \
	--seed 42 \
	--save_log \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	backbone_dir=/path/to/your/swin_large_patch4_window12_384_22k.pth \
	--note "your note"
device=$1
python main.py \
	--output_dir /path/to/your/logs/DINO/CHBMIT/foo \
	-c config/DINO/DINO_4scale_swin_chbmit.py \
	--dataset chbmit \
	--data_dir /path/to/your/chbmit/STFT_Data_Dir \
	--chbmit_txt_dir /path/to/your/chbmit/TXT_file_Dir \
	--device $device \
	--seed 42 \
	--save_log \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 \
	backbone_dir=/path/to/your/swin_large_patch4_window12_384_22k.pth \
	--note "your note"
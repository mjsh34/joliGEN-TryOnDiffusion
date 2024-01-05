#python3 train.py \
#	--G_netG unet_mha_ref_attn \
#	--data_dataset_mode self_supervised_labeled_mask_online_ref \
#	--model_type palette \
#	--alg_palette_conditioning " " \
#      	--alg_palette_cond_image_creation y_t \

python3 train.py \
	--dataroot ../Datasets/processed/VITON_HD_refbb \
	--checkpoints_dir ./checkpoints/tryondiffusion \
       	--name tryondiffusion_viton \
	--config_json examples/example_ddpm_unetref_viton.json \
	#--data_online_creation_load_size_A #768 1024

# Transform model shots to ghost mannequin shots
# _rev datasets have contents of imgs/ and ref/ swapped from normally preprocessed VITON-HD datasets.

python3 train.py \
	--dataroot ../Datasets/processed/VITON_HD_nof_rev \
	--checkpoints_dir ./checkpoints/model2ghost_nof \
       	--name model2ghost_nof \
	--output_display_env model2ghost_nof \
	--config_json examples/example_ddpm_unetref_viton.json \
	--data_online_creation_load_size_A 768 834 # train on full resolution w/o cropping (https://github.com/jolibrain/joliGEN/issues/568#issuecomment-1772212006).
	# We are using the modified VITON-HD dataset with part of the model's head removed from image.

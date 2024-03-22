python test_data.py \
	--mixing 0 \
	--batchSize 1 \
	--nThreads 1 \
	--name comod-ffhq-512 \
	--dataset_mode cubimage \
	--image_dir /data/kb/tanyuanyong/TransFG-master/data/CUB_200_2011 \
  --output_dir ./output/CUB_200_2011_MASK_0.7 \
	--crop_size 512 \
	--z_dim 512 \
	--model comod \
	--netG comodgan \
  --which_epoch co-mod-gan-ffhq-9-025000 \
  --min_hole 0.0 \
  --max_hole 0.7 \
  	${EXTRA} \

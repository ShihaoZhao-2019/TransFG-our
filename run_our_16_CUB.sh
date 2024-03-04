CUDA_VISIBLE_DEVICES=5,6,7 nohup  python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=14748 train_L0_k685_2l_con_premlp_fixFG.py --dataset CUB_200_2011 --split overlap --num_steps 40000 --fp16 --fp16_opt_level O0 --name our_16 --train_batch_size 1 --eval_batch_size 3 --eval_every 1000 --learning_rate 3e-2 --warmup_steps 4000 --model_type=ViT-B_16 --pretrained_dir /data/kb/tanyuanyong/TransFG-master/data/vit_model/ViT-B_16.npz >our_16_CUB.file 2>&1 &

tail -F our_16_CUB.file

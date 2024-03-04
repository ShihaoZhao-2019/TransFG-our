CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 nohup  python3 -m torch.distributed.launch --nproc_per_node=7 --master_port=10748 train_L0_k685_2l_con_premlp_fixFG.py --dataset CUB_200_2011 --split overlap --num_steps 40000 --fp16 --fp16_opt_level O0 --name our_32 --train_batch_size 2 --eval_batch_size 4 --eval_every 1000 --learning_rate 3e-2 --warmup_steps 4000 --model_type=ViT-B_32 --pretrained_dir /data/kb/tanyuanyong/TransFG-master/data/vit_model/ViT-B_32.npz >our_32_CUB.file 2>&1 &

tail -F our_32_CUB.file

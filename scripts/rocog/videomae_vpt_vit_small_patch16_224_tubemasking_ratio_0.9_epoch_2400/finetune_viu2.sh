# APT on SSv2
OUTPUT_DIR='experiments/APT/ROCOG/syn_air_to_real_air/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/adam_mome9e-1_wd1e-5_lr5e-2_pl200_ps0_pe11_drop10'
DATA_PATH='datasets/rocogv2/splits/syn_air_to_real_air'
MODEL_PATH='experiments/pretrain/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint.pth'

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=8 \
    run_class_apt.py \
    --model vit_small_patch16_224 \
    --transfer_type prompt \
    --prompt_start 0 \
    --prompt_end 11 \
    --prompt_num_tokens 200 \
    --prompt_dropout 0.1 \
    --data_set ROCOG \
    --nb_classes 7 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --batch_size_val 16 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 0.05 \
    --weight_decay 0.00001 \
    --epochs 50 \
    --warmup_epochs 10 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --pin_mem \
    --enable_deepspeed \
    --prompt_reparam \
    --is_aa \
    --aa rand-m4-n2-mstd0.2-inc1
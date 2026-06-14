

export NUM_GPUS=2
export global_bs=16
export micro_bs=2
export loss_type=kd_diff_sum
export reveal_mode=gt
export gt_topk=1
export mrp_steps=3
export loss_weight_mode=uniform
export mrp_init_std=0.2
export LR=1e-3
export freeze_backbone=True
export freeze_lm_head=True
export override=False
export dataset=ultrachat_200k


SIZE=8
export MODEL_PATH="checkpoints/sdar-${SIZE}b-mrp-3lyr"
export MODEL_NAME="mrp_sdar-${SIZE}b-chat-b16-3lyr"
mrp_steps=1 \
bash scripts/train_sdar_mrp.sh

export MODEL_PATH="checkpoints/sdar-30b-a3b-mrp-3lyr"
export MODEL_NAME="mrp_sdar-30b-a3b-chat-b16-3lyr"
mrp_steps=1 \
bash scripts/train_sdar_mrp.sh
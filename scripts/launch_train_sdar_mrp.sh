#!/bin/bash

set -euo pipefail

# Resume the incomplete K=3 GT reveal runs for 4B and 8B.
# Keep these knobs aligned with the existing checkpoint directory names so
# LLaMAFactory resumes the intended checkpoints when override=False.

export NUM_GPUS=${NUM_GPUS:-8}
export global_bs=${global_bs:-16}
export micro_bs=${micro_bs:-2}
export dataset=sftdatasetv3
export input_prep_mode=bd_packed
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
export overwrite_cache=false
export preprocessing_num_workers=${preprocessing_num_workers:-8}

run_resume() {
    local size="$1"
    local ckpt="$2"

    export MODEL_PATH="checkpoints/sdar-${size}b-mrp-3lyr"
    export MODEL_NAME="mrp_sdar-${size}b-chat-b16-3lyr"
    export resume_from_checkpoint="${ckpt}"

    nohup-queue run scripts/train_sdar_mrp.sh
}

run_resume 4 "checkpoints/mrp_sdar-4b-chat-b16-3lyr/ds_sftdatasetv3_seq_4096_bs_16_ep_1_freezebb_True_freezelm_True/loss_kd_diff_sum_T_1_0_r_False_init_std_0_2_reveal_gt_1_K_3_lwm_uniform_prep_bd_packed_lr_1e-3_cosine_with_min_lr/checkpoint-62250"
run_resume 8 "checkpoints/mrp_sdar-8b-chat-b16-3lyr/ds_sftdatasetv3_seq_4096_bs_16_ep_1_freezebb_True_freezelm_True/loss_kd_diff_sum_T_1_0_r_False_init_std_0_2_reveal_gt_1_K_3_lwm_uniform_prep_bd_packed_lr_1e-3_cosine_with_min_lr/checkpoint-55000"

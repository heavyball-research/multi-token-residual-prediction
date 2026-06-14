#!/bin/bash
# Launch v1.5 training for the 1.7B backbone.
#
# v1.5 architecture differences from the original recipe:
#   - MRP module takes target's layer[L-D-1] output as input (not h_A).
#     Set via config.json: "mrp_input_layer_offset": -4 for D=3.
#   - Drops the explicit residual addition (lm_head(h_C) directly, not h_A+h_C).
#     Set via config.json: "mrp_drop_residual": true.
#   - MRP layers are initialized as copies of target.layer[L-D:L] at training
#     start. Enabled below via mrp_init_from_target_layers=True.
#   - Standard mrp_init_std=0.02 (no longer need the std=0.2 cold-start hack
#     since init-from-target gives a usable starting point).
#
# Combined effect: at step 0, MRP reproduces target's last D layers' output
# exactly, so prediction(init) = ℓ_A. The optimizer only has to learn the
# small deviation needed to predict ℓ_{t-1} (B).
#
# Limitation: v1.5 supports K=1 cleanly. Multi-step (K>=2) iteration is open;
# this script uses mrp_steps=1.
#
# Loss type: must be a non-sum loss (kd_diff, kd, or ce). Sum losses
# (kd_diff_sum) assume the h_A + h_C residual which v1.5 drops.

set -euo pipefail

export NUM_GPUS=${NUM_GPUS:-2}
export global_bs=${global_bs:-16}
# Bumped from 2 → 4: 80GB H100 has plenty of headroom at seq=4096 with the
# backbone frozen, and the larger micro batch drops grad_accum from 4 → 2,
# halving per-step fixed overhead (ZeRO reduce-scatter, logger, sync).
export micro_bs=${micro_bs:-4}

# Disable activation checkpointing. With frozen backbone + freeze_lm_head, only
# MRP (3 layers) backprops, and at micro_bs=4/seq=4096 we are nowhere near OOM
# on 80GB. The extra forward gc adds is pure overhead here. DETERMINISTIC=1
# (the train script's default) is kept intentionally for reproducibility.
export gradient_checkpointing=${gradient_checkpointing:-false}

# Dataset: ultrachat 200k as requested.
export dataset=ultrachat_200k

export input_prep_mode=bd_packed
# kd_diff (forward KL of target's logits vs MRP's logits, no residual sum).
# Sum-loss variants are NOT compatible with mrp_drop_residual=true.
export loss_type=kd_diff
export kd_temperature=1.0
export kd_reverse_kl=False

export reveal_mode=gt_random
export gt_random_topk="1,2,3,4"  # sample reveal count from {1,2,3,4} per block per step
export gt_topk=1                 # unused when reveal_mode=gt_random, kept for arg-passthrough
export mrp_steps=1
export loss_weight_mode=uniform

# v1.5: init MRP from target's last D layers; use standard init for anything else.
export mrp_init_std=0.02
export mrp_init_from_target_layers=True
export init_eh_proj=False

export LR=${LR:-1e-4}
export freeze_backbone=True
export freeze_lm_head=True
export override=False
export overwrite_cache=false
export preprocessing_num_workers=${preprocessing_num_workers:-8}

# Point at the assembled v1.5 checkpoint directory (modeling files + safetensors
# from the baseline 1.7B target). MRP weights will be re-initialized from
# target's last D layers via init_mrp_from_target_layers below.
export MODEL_PATH="checkpoints/sdar-1_7b-mrp-v1_5-3lyr"
export MODEL_NAME="mrp_sdar-1_7b-v15-3lyr"

# Run training (and eval after, per train_sdar_mrp.sh's do_eval gate).
nohup-queue run scripts/train_sdar_mrp.sh

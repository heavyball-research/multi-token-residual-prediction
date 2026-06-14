#!/bin/bash

set -euo pipefail

# Train SDAR-MRP 1.7B (3-layer MRP head) on ultrachat_200k with the ce_sum
# loss: cross-entropy of the summed logits(A + C) against the GROUND-TRUTH
# labels. Exploits lm_head linearity (lm_head(h_A + h_C) = logits_A + logits_C)
# and needs NO second target forward (no B) — only h_A + labels.
#
# Sum loss => train_sdar_mrp.sh auto-selects mrp_logit_mix_mode=logits_sum
# (the matched inference mode).

export NUM_GPUS=${NUM_GPUS:-4}
export global_bs=${global_bs:-16}
export micro_bs=${micro_bs:-4}
export dataset=ultrachat_200k
export input_prep_mode=bd_packed
export loss_type=ce_sum
export reveal_mode=gt
export gt_topk=1
export mrp_steps=1
export mrp_init_std=0.2
export LR=1e-3
export freeze_backbone=True
export freeze_lm_head=True
export override=False
export overwrite_cache=false
export preprocessing_num_workers=${preprocessing_num_workers:-8}

export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}
export MODEL_NAME=${MODEL_NAME:-"mrp_sdar-1_7b-chat-b16-3lyr"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[launch_train_ce_sum] queuing ${MODEL_NAME} (${MODEL_PATH}) on ${dataset} " \
     "loss_type=${loss_type} NUM_GPUS=${NUM_GPUS}"
nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/train_sdar_mrp.sh"

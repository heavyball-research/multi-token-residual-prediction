#!/bin/bash

set -euo pipefail

# Threshold sweep for the v1.5 1.7B checkpoint at MRP_STEPS=1.
#
# Runs one eval per target confidence threshold against the completed
# epoch-1 ckpt of the LR=1e-5 v1.5 run. Each eval uses dynamic
# low-confidence decoding on the target (the threshold knob actually does
# something), and static low_confidence_static for the MRP path.
#
# The threshold controls how confident the target must be on a position
# before that position is committed. Lower threshold ⇒ more aggressive
# target commitment per step; threshold=1.0 ⇒ effectively never commit
# off the schedule (only 1 token per denoising step, pure schedule order).

# One log per (ckpt, MRP_STEPS, threshold) — keep CLAUDE.md eval-parsing
# convention so summary tooling still works.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# v1.5 + MRP_STEPS=1 inference config.
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-none}
export MRP_REMASKING_STRATEGY=${MRP_REMASKING_STRATEGY:-low_confidence_static}
export MRP_NUM_TRANSFER_PER_ITER=${MRP_NUM_TRANSFER_PER_ITER:-1}

# Dynamic threshold on the target side — this is the knob being swept.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}

# Completed v1.5 1.7B ckpt (LR=1e-5, epoch=1, step=3854).
CKPT_ROOT="checkpoints/mrp_sdar-1_7b-v15-3lyr/ds_ultrachat_200k_seq_4096_bs_16_ep_1_freezebb_True_freezelm_True"
RUN_DIR="loss_kd_diff_T_1_0_r_False_init_std_0_02_reveal_gt_random_1-4_prep_bd_packed_lr_1e-5_cosine_with_min_lr"
export MODEL_PATH=${MODEL_PATH:-"${CKPT_ROOT}/${RUN_DIR}/checkpoint-3854"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_sdar_mrp_v15_mrp1_threshold] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

THRESHOLDS=(
    "1.00"
    "0.95"
    "0.90"
    "0.85"
    "0.80"
)


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for THRESHOLD in "${THRESHOLDS[@]}"; do
    export THRESHOLD
    echo "[launch_eval_sdar_mrp_v15_mrp1_threshold] queuing ckpt=${MODEL_PATH##*/} MRP_STEPS=${MRP_STEPS} THRESHOLD=${THRESHOLD}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

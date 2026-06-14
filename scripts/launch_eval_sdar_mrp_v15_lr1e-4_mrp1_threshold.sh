#!/bin/bash

set -euo pipefail

# Threshold sweep for the v1.5 1.7B LR=1e-4 run — intermediate checkpoint.
#
# The LR=1e-4 run is still training; save_total_limit=1 means only the most
# recent `checkpoint-N` dir survives at any time. This script auto-resolves
# whichever ckpt-N is currently on disk and runs the same threshold sweep we
# do for the completed LR=1e-5 endpoint. Useful for comparing the two runs
# while the LR=1e-4 one is still in flight.
#
# Override MODEL_PATH to pin a specific ckpt:
#   MODEL_PATH=checkpoints/.../lr_1e-4_.../checkpoint-2750 bash scripts/launch_eval_sdar_mrp_v15_lr1e-4_mrp1_threshold.sh

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

# Dynamic threshold on the target side — knob being swept.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}

# LR=1e-4 v1.5 1.7B run.
CKPT_ROOT="checkpoints/mrp_sdar-1_7b-v15-3lyr/ds_ultrachat_200k_seq_4096_bs_16_ep_1_freezebb_True_freezelm_True"
RUN_DIR="loss_kd_diff_T_1_0_r_False_init_std_0_02_reveal_gt_random_1-4_prep_bd_packed_lr_1e-4_cosine_with_min_lr"
RUN_PATH="${CKPT_ROOT}/${RUN_DIR}"

# If the caller pinned MODEL_PATH already, use it. Otherwise pick the
# highest-numbered checkpoint-* dir that's currently on disk. With
# save_total_limit=1, this is whatever the trainer most recently saved.
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "${RUN_PATH}" ]]; then
        echo "[launch_eval_v15_lr1e-4] run dir does not exist: ${RUN_PATH}" >&2
        exit 1
    fi
    LATEST_CKPT=$(ls -d "${RUN_PATH}"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2, $0}' \
        | sort -n -k1,1 \
        | awk 'END {print $2}')
    if [[ -z "${LATEST_CKPT}" ]]; then
        echo "[launch_eval_v15_lr1e-4] no checkpoint-* dir under ${RUN_PATH}" >&2
        exit 1
    fi
    export MODEL_PATH="${LATEST_CKPT}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_v15_lr1e-4] missing checkpoint dir: ${MODEL_PATH}" >&2
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

echo "[launch_eval_v15_lr1e-4] evaluating ${MODEL_PATH}"
for THRESHOLD in "${THRESHOLDS[@]}"; do
    export THRESHOLD
    echo "[launch_eval_v15_lr1e-4] queuing MRP_STEPS=${MRP_STEPS} THRESHOLD=${THRESHOLD}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

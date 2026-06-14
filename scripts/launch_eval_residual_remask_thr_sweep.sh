#!/bin/bash

set -euo pipefail

# residual_remask THRESHOLD x RESIDUAL_DELTA grid for sdar-1_7b-mrp-3lyr.
#
# Mode: target unmasks under THRESHOLD (low_confidence_dynamic); MRP's
# logit-space residual remasks a committed token when
# |C[pos, committed_id]| > RESIDUAL_DELTA. MRP never commits.
#
# Grid: THRESHOLD in {0.85, 0.9, 0.95} x RESIDUAL_DELTA in {0.5, 1.0, 2.0, 4.0}
# = 12 jobs, each queued via `nohup-queue run`. One log per
# (ckpt, MRP_STEPS, THRESHOLD, RESIDUAL_DELTA) per CLAUDE.md eval convention.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Aggressive target unmask with dynamic remasking; the residual veto is the brake.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}

# residual_remask configuration. MRP runs as the vetoer only.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-residual_remask}
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_residual_remask_thr_sweep] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

THRESHOLD_LIST=(0.85 0.9 0.95)
DELTA_LIST=(0.5 1.0 2.0 4.0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for T in "${THRESHOLD_LIST[@]}"; do
    export THRESHOLD="${T}"
    for D in "${DELTA_LIST[@]}"; do
        export RESIDUAL_DELTA="${D}"
        echo "[launch_eval_residual_remask_thr_sweep] queuing ckpt=${MODEL_PATH##*/} " \
             "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
             "THRESHOLD=${THRESHOLD} RESIDUAL_DELTA=${RESIDUAL_DELTA}"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
    done
done

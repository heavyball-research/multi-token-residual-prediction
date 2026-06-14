#!/bin/bash

set -euo pipefail

# residual_remask scale sweep: sdar-{4b,8b}-mrp-3lyr x THRESHOLD x delta=2.0.
#
# Mode: target unmasks under THRESHOLD (low_confidence_dynamic); MRP's
# logit-space residual remasks a committed token when
# |C[pos, committed_id]| > RESIDUAL_DELTA (fixed 2.0 here). MRP never commits.
#
# Grid: MODEL in {sdar-4b-mrp-3lyr, sdar-8b-mrp-3lyr}
#       x THRESHOLD in {0.8, 0.85, 0.9, 0.95}  (delta fixed = 2.0)
# = 8 jobs, queued via `nohup-queue run`.
#
# NOTE: BLOCK_LENGTH is forced to 16 — the 4b ckpt ships config.block_size=4
# (stale metadata; the weights are the b=16 target model), so we pass 16
# explicitly to match the 1.7b/8b block layout.
#
# NOTE: delta=2.0 was tuned on 1.7b. Per the residual-outlier profiling the
# 4b/8b residual magnitudes are larger, so delta=2.0 likely vetoes a larger
# fraction at scale — interpret the cost numbers with that in mind.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-residual_remask}
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}
export RESIDUAL_DELTA=${RESIDUAL_DELTA:-2.0}

MODEL_LIST=(checkpoints/sdar-4b-mrp-3lyr checkpoints/sdar-8b-mrp-3lyr)
THRESHOLD_LIST=(0.8 0.85 0.9 0.95)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MP in "${MODEL_LIST[@]}"; do
    if [[ ! -d "${MP}" ]]; then
        echo "[launch_eval_residual_remask_4b8b_d2] missing checkpoint dir: ${MP}" >&2
        exit 1
    fi
    export MODEL_PATH="${MP}"
    for T in "${THRESHOLD_LIST[@]}"; do
        export THRESHOLD="${T}"
        echo "[launch_eval_residual_remask_4b8b_d2] queuing ckpt=${MODEL_PATH##*/} " \
             "MRP_VERIFY_MODE=${MRP_VERIFY_MODE} THRESHOLD=${THRESHOLD} RESIDUAL_DELTA=${RESIDUAL_DELTA}"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
    done
done

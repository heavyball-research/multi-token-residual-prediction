#!/bin/bash

set -euo pipefail

# residual_remask low-THRESHOLD sweep for sdar-1_7b-mrp-3lyr at delta=2.0.
#
# Mode: target unmasks AGGRESSIVELY under a low THRESHOLD (low_confidence_dynamic);
# MRP's logit-space residual remasks a committed token when
# |C[pos, committed_id]| > RESIDUAL_DELTA (fixed 2.0). MRP never commits — it is
# the vetoer only. The residual veto is the brake on aggressive early commits.
#
# This is the deterministic (magnitude-selection) veto. Its matched-count random
# control is launch_eval_residual_remask_scale_d2_thr_low_random.sh.
#
# Grid: THRESHOLD in {0.6, 0.65, 0.7, 0.75}  (delta fixed = 2.0)
# = 4 jobs, queued via `nohup-queue run`. One log per
# (ckpt, MRP_STEPS, THRESHOLD, RESIDUAL_DELTA) per CLAUDE.md eval convention.
#
# Low thresholds stress the hypothesis: the target commits more (and more
# prematurely), giving the residual veto something to recover — unlike the
# conservative 0.8+ thresholds where accuracy was flat.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Aggressive target unmask with dynamic remasking; the residual veto is the brake.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}

# residual_remask configuration. MRP runs as the vetoer only (deterministic).
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-residual_remask}
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}
export RESIDUAL_DELTA=${RESIDUAL_DELTA:-2.0}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_residual_remask_1_7b_d2_thr_low] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

THRESHOLD_LIST=(0.6 0.65 0.7 0.75)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for T in "${THRESHOLD_LIST[@]}"; do
    export THRESHOLD="${T}"
    echo "[launch_eval_residual_remask_1_7b_d2_thr_low] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_VERIFY_MODE=${MRP_VERIFY_MODE} THRESHOLD=${THRESHOLD} RESIDUAL_DELTA=${RESIDUAL_DELTA}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

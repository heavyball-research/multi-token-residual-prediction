#!/bin/bash

set -euo pipefail

# residual_remask small-delta sweep for sdar-1_7b-mrp-3lyr at THRESHOLD=0.8.
#
# Mode: target unmasks under THRESHOLD (low_confidence_dynamic); MRP's
# logit-space residual remasks a committed token when
# |C[pos, committed_id]| > RESIDUAL_DELTA. MRP never commits.
#
# This sweep probes the AGGRESSIVE-veto regime: RESIDUAL_DELTA in
# {0.05, 0.1, 0.2, 0.3, 0.4} (much tighter than the {0.5..4.0} grid). Per the
# veto-scalar distribution (median |C| ~3.5), these δ veto the vast majority of
# commits, so the decode degrades toward ~1 commit/step — this measures whether
# very tight vetoing buys any accuracy at the (large) throughput cost.
#
# One log per (ckpt, MRP_STEPS, THRESHOLD, RESIDUAL_DELTA) per CLAUDE.md.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Aggressive target unmask with dynamic remasking; the residual veto is the brake.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
export THRESHOLD=${THRESHOLD:-0.8}

# residual_remask configuration. MRP runs as the vetoer only.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-residual_remask}
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_residual_remask_small_delta] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

DELTA_LIST=(0.05 0.1 0.2 0.3 0.4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for D in "${DELTA_LIST[@]}"; do
    export RESIDUAL_DELTA="${D}"
    echo "[launch_eval_residual_remask_small_delta] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
         "THRESHOLD=${THRESHOLD} RESIDUAL_DELTA=${RESIDUAL_DELTA}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

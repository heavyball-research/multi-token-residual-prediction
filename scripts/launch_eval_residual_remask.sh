#!/bin/bash

set -euo pipefail

# residual_remask sweep for the published sdar-1_7b-mrp-3lyr ckpt.
#
# Mode: the target unmasks AGGRESSIVELY under a low confidence THRESHOLD, then
# MRP's logit-space residual vetoes per token — a committed token is remasked
# when |C[pos, committed_id]| > RESIDUAL_DELTA (C = MRP's raw prediction). MRP
# never commits; it only remasks unstable early commits so they resolve later
# with more context. MRP_STEPS=1 (one MRP veto forward per denoising step).
#
# We sweep RESIDUAL_DELTA (the only knob) at a fixed low target threshold.
# One log per (ckpt, MRP_STEPS, RESIDUAL_DELTA) per CLAUDE.md eval convention.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Aggressive target unmask: low confidence threshold + dynamic remasking so
# the target commits many positions per step; the residual veto is the brake.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
export THRESHOLD=${THRESHOLD:-0.5}

# residual_remask configuration. MRP runs as the vetoer only.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-residual_remask}
export MRP_STEPS=${MRP_STEPS:-1}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_residual_remask] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

# Sweep the veto threshold. 1.0 is the default starting point; smaller = more
# aggressive remasking, larger = looser veto (closer to plain dynamic decode).
DELTA_LIST=(0.5 1.0 2.0 4.0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for D in "${DELTA_LIST[@]}"; do
    export RESIDUAL_DELTA="${D}"
    echo "[launch_eval_residual_remask] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
         "THRESHOLD=${THRESHOLD} RESIDUAL_DELTA=${RESIDUAL_DELTA}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

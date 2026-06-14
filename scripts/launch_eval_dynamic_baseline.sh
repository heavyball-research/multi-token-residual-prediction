#!/bin/bash

set -euo pipefail

# Target-only dynamic-decode BASELINE for sdar-1_7b-mrp-3lyr.
#
# No MRP at all (MRP_STEPS=0, MRP_VERIFY_MODE=none): plain low_confidence_dynamic
# decoding at each THRESHOLD. This is the control for the residual_remask sweeps
# — same accuracy axis, but tokens/forward with NO MRP forward tax, so it
# isolates what the residual veto actually costs vs. buys.
#
# Sweep: THRESHOLD in {0.8, 0.85, 0.9, 0.95} = 4 jobs, queued via
# `nohup-queue run`. One log per (ckpt, THRESHOLD) per CLAUDE.md convention.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Plain dynamic target decoding; no MRP.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-none}
export MRP_STEPS=${MRP_STEPS:-0}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_dynamic_baseline] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

THRESHOLD_LIST=(0.8 0.85 0.9 0.95)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for T in "${THRESHOLD_LIST[@]}"; do
    export THRESHOLD="${T}"
    echo "[launch_eval_dynamic_baseline] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_STEPS=0 MRP_VERIFY_MODE=none REMASKING_STRATEGY=${REMASKING_STRATEGY} " \
         "THRESHOLD=${THRESHOLD}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

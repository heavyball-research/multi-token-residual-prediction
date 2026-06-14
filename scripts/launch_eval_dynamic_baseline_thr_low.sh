#!/bin/bash

set -euo pipefail

# Target-only dynamic-decode BASELINE for sdar-1_7b-mrp-3lyr at LOW thresholds.
#
# No MRP at all (MRP_STEPS=0, MRP_VERIFY_MODE=none): plain low_confidence_dynamic
# decoding at each THRESHOLD. This is the missing control for the low-threshold
# residual_remask sweeps (launch_eval_residual_remask_1_7b_d2_thr_low.sh and
# launch_eval_residual_remask_scale_d2_thr_low_random.sh) — same accuracy axis
# and the SAME thresholds, but with NO MRP forward tax, so it isolates what the
# residual veto (deterministic or random) actually buys over plain aggressive
# dynamic decoding at the thresholds where early-commit instability shows up.
#
# Sweep: THRESHOLD in {0.6, 0.65, 0.7, 0.75} = 4 jobs, queued via
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
    echo "[launch_eval_dynamic_baseline_thr_low] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

THRESHOLD_LIST=(0.6 0.65 0.7 0.75)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for T in "${THRESHOLD_LIST[@]}"; do
    export THRESHOLD="${T}"
    echo "[launch_eval_dynamic_baseline_thr_low] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_STEPS=0 MRP_VERIFY_MODE=none REMASKING_STRATEGY=${REMASKING_STRATEGY} " \
         "THRESHOLD=${THRESHOLD}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

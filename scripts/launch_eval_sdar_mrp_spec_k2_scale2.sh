#!/bin/bash

set -euo pipefail

# Spec-decoding sweep for the sdar-k2 (1.7B / 4B / 8B) MRP checkpoints with
# the new 'logits_sum' accumulator scaled to 2x — i.e. A + 2C for both the
# logit accumulator and the matching hidden-state accumulator fed into the
# next MRP forward (MRP_LOGIT_SUM_SCALE=2.0).
#
# Target uses static low_confidence_static remasking (fixed schedule, no
# threshold knob); verify mode is 'spec' (panel (d) parallel verification).
# K = MRP_STEPS is swept across {1, 2, 3}.
#
# One log per (ckpt, MRP_STEPS, REMASKING_STRATEGY) per CLAUDE.md eval-
# parsing convention.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Static target denoising — same schedule for every K so accept-rate
# differences are purely a function of draft quality at K.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_static}

# Spec-decoding configuration. K is set per-iteration below.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-spec}
# 'logits_sum' enables the cross-iter accumulator (A + sum_scale * C per
# iter); sum_scale of 2.0 produces A + 2C, matching the new ablation.
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-logits_sum}
export MRP_LOGIT_SUM_SCALE=${MRP_LOGIT_SUM_SCALE:-2.0}
export MRP_REMASKING_STRATEGY=${MRP_REMASKING_STRATEGY:-low_confidence_static}
export MRP_NUM_TRANSFER_PER_ITER=${MRP_NUM_TRANSFER_PER_ITER:-1}

CKPT_LIST=(
    "checkpoints/sdar-1_7b-mrp-3lyr-k2"
    "checkpoints/sdar-4b-mrp-3lyr-k2"
    "checkpoints/sdar-8b-mrp-3lyr-k2"
)
MRP_STEPS_LIST=(1 2 3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Queue every (ckpt, K) as a single-GPU job via nohup-queue. The queue claims
# GPUs (--num-gpus) and runs jobs in the background, so on the 2x H100 box the
# 9 jobs self-distribute across both cards as they free up. Each job logs to
# logs/eval_opencompass_hf/<timestamp>.log per the CLAUDE.md convention.
for CKPT in "${CKPT_LIST[@]}"; do
    if [[ ! -d "${CKPT}" ]]; then
        echo "[launch_eval_sdar_mrp_spec_k2_scale2] missing checkpoint dir: ${CKPT}" >&2
        exit 1
    fi
    export MODEL_PATH="${CKPT}"
    for K in "${MRP_STEPS_LIST[@]}"; do
        export MRP_STEPS="${K}"
        echo "[launch_eval_sdar_mrp_spec_k2_scale2] queuing ckpt=${MODEL_PATH##*/} " \
             "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
             "MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE} " \
             "MRP_LOGIT_SUM_SCALE=${MRP_LOGIT_SUM_SCALE} " \
             "REMASKING_STRATEGY=${REMASKING_STRATEGY}"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
    done
done

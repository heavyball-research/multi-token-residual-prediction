#!/bin/bash

set -euo pipefail

# DIRECT (no-verify) scale=2 logits_sum sweep for the sdar-k2 (1.7B / 4B / 8B)
# MRP checkpoints.
#
# Unlike the *_spec_k2_scale2 sweep (MRP_VERIFY_MODE=spec, which is lossless by
# construction because the target verifies every commit), this sweep runs
# MRP_VERIFY_MODE=none: the logits_sum-aggregated MRP logits
# (prev_logits + MRP_LOGIT_SUM_SCALE * mrp_logits) are sampled and COMMITTED
# DIRECTLY. Here MRP_LOGIT_SUM_SCALE actually changes the generated output, so
# this is the sweep that measures scale=2's true effect on accuracy.
#
# Target denoising uses low_confidence_static; K = MRP_STEPS swept over {1,2,3}.
#
# One log per (ckpt, MRP_STEPS) via nohup-queue, to
# logs/eval_opencompass_hf/<timestamp>.log per CLAUDE.md convention. The abbr
# omits the _v_spec tag (verify=none) so these don't collide with the spec runs.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}

# Static target denoising — same schedule for every K.
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_static}

# Direct-commit MRP: logits_sum accumulator scaled 2x, NO verification.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-none}
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

# Queue every (ckpt, K) via nohup-queue. The queue claims a GPU per job and
# self-distributes across the box's GPUs.
for CKPT in "${CKPT_LIST[@]}"; do
    if [[ ! -d "${CKPT}" ]]; then
        echo "[launch_eval_sdar_mrp_direct_k2_scale2] missing checkpoint dir: ${CKPT}" >&2
        exit 1
    fi
    export MODEL_PATH="${CKPT}"
    for K in "${MRP_STEPS_LIST[@]}"; do
        export MRP_STEPS="${K}"
        echo "[launch_eval_sdar_mrp_direct_k2_scale2] queuing ckpt=${MODEL_PATH##*/} " \
             "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
             "MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE} " \
             "MRP_LOGIT_SUM_SCALE=${MRP_LOGIT_SUM_SCALE} " \
             "REMASKING_STRATEGY=${REMASKING_STRATEGY}"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
    done
done

echo "[launch_eval_sdar_mrp_direct_k2_scale2] queued all 9 runs"

#!/bin/bash

set -euo pipefail

# Vanilla spec-decoding sweep for the published sdar-1_7b-mrp-3lyr ckpt.
# Target uses static low_confidence_static remasking (fixed schedule, no
# threshold knob); verify mode is 'spec' (panel (d) parallel verification).
# K = MRP_STEPS is swept across {1, 2, 3, 4} — one eval per K.
#
# One log per (ckpt, MRP_STEPS, REMASKING_STRATEGY) per CLAUDE.md
# eval-parsing convention.

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
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}
export MRP_REMASKING_STRATEGY=${MRP_REMASKING_STRATEGY:-low_confidence_static}
export MRP_NUM_TRANSFER_PER_ITER=${MRP_NUM_TRANSFER_PER_ITER:-1}

# Published sdar-mrp 1.7B checkpoint.
export MODEL_PATH=${MODEL_PATH:-"checkpoints/sdar-1_7b-mrp-3lyr"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[launch_eval_sdar_mrp_spec] missing checkpoint dir: ${MODEL_PATH}" >&2
    exit 1
fi

MRP_STEPS_LIST=(1 2 3 4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for K in "${MRP_STEPS_LIST[@]}"; do
    export MRP_STEPS="${K}"
    echo "[launch_eval_sdar_mrp_spec] queuing ckpt=${MODEL_PATH##*/} " \
         "MRP_STEPS=${MRP_STEPS} MRP_VERIFY_MODE=${MRP_VERIFY_MODE} " \
         "REMASKING_STRATEGY=${REMASKING_STRATEGY}"
    nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
done

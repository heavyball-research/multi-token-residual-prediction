#!/bin/bash

set -euo pipefail

# Low-threshold sweep for sdar-{4b,8b}-mrp-3lyr: residual_remask (delta=2.0)
# AND the matching target-only baselines, over THRESHOLD in {0.75,0.7,0.65,0.6}.
#
# This extends the earlier {0.8,0.85,0.9,0.95} grid into the aggressive-unmask
# regime, where plain dynamic decode commits more (lower threshold) and the
# residual veto has more unstable early commits to act on. delta fixed at 2.0.
#
# Per (model, threshold) we queue TWO jobs:
#   - residual_remask: MRP_VERIFY_MODE=residual_remask, MRP_STEPS=1, RESIDUAL_DELTA=2.0
#   - baseline:        MRP_VERIFY_MODE=none, MRP_STEPS=0   (no MRP forward)
# => 2 models x 4 thresholds x 2 modes = 16 jobs, via `nohup-queue run`.
#
# BLOCK_LENGTH forced to 16 (4b config.block_size is now repaired to 16; 8b is 16).

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export PROFILE_TPF=${PROFILE_TPF:-1}
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-low_confidence_dynamic}
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-none}

MODEL_LIST=(checkpoints/sdar-4b-mrp-3lyr checkpoints/sdar-8b-mrp-3lyr)
THRESHOLD_LIST=(0.75 0.7 0.65 0.6)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MP in "${MODEL_LIST[@]}"; do
    if [[ ! -d "${MP}" ]]; then
        echo "[launch_eval_4b8b_lowthr] missing checkpoint dir: ${MP}" >&2
        exit 1
    fi
    export MODEL_PATH="${MP}"
    for T in "${THRESHOLD_LIST[@]}"; do
        export THRESHOLD="${T}"

        # --- residual_remask, delta=2.0 ---
        export MRP_VERIFY_MODE=residual_remask
        export MRP_STEPS=1
        export RESIDUAL_DELTA=2.0
        echo "[launch_eval_4b8b_lowthr] queuing ${MP##*/} residual_remask THRESHOLD=${T} RESIDUAL_DELTA=2.0"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"

        # --- target-only baseline ---
        export MRP_VERIFY_MODE=none
        export MRP_STEPS=0
        echo "[launch_eval_4b8b_lowthr] queuing ${MP##*/} baseline       THRESHOLD=${T}"
        nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/eval_opencompass_hf.sh"
    done
done

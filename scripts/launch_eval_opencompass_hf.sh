#!/bin/bash

set -euo pipefail

# Evaluate local reveal_conf_threshold checkpoints with matching dynamic
# low-confidence decoding thresholds. Each checkpoint was trained with one
# confidence threshold; eval mirrors that threshold in decoding.

export NUM_GPUS=${NUM_GPUS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export REUSE=${REUSE:-false}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}

export MRP_LOGIT_MIX_MODE="logits_sum"
export REMASKING_STRATEGY="low_confidence_dynamic"
export MRP_VERIFY_MODE="none"
export MRP_REMASKING_STRATEGY="low_confidence_static"
export MRP_NUM_TRANSFER_PER_ITER=1

CKPT_ROOT="checkpoints/mrp_sdar-1_7b-chat-b16-3lyr/ds_ultrachat_200k_seq_4096_bs_16_ep_1_freezebb_True_freezelm_True"
LOSS_PREFIX="loss_kd_diff_sum_T_1_0_r_False_init_std_0_2_reveal_conf_threshold"
LOSS_SUFFIX="K_3_lwm_uniform_prep_infer_matched_lr_1e-3_cosine_with_min_lr"

CONF_THRESHOLDS=(
    "0.95"
    "0.96"
    "0.97"
    "0.98"
    "0.99"
)

for CONF_THRESHOLD in "${CONF_THRESHOLDS[@]}"; do
    threshold_token=${CONF_THRESHOLD//./_}
    model_dir="${CKPT_ROOT}/${LOSS_PREFIX}_${threshold_token}_${LOSS_SUFFIX}"

    if [[ ! -d "${model_dir}" ]]; then
        echo "[launch_eval_opencompass_hf] missing checkpoint dir: ${model_dir}" >&2
        exit 1
    fi

    export MODEL_PATH="${model_dir}"
    export THRESHOLD="${CONF_THRESHOLD}"

    MRP_STEPS=0 nohup-queue run ./scripts/eval_opencompass_hf.sh
    # MRP_STEPS=2 nohup-queue run ./scripts/eval_opencompass_hf.sh
done

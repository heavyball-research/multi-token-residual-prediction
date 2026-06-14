#!/bin/bash

set -euo pipefail

# Launch random-position GT reveal training with random per-block budgets.
# reveal_mode=gt_random samples one count from gt_random_topk per active
# block, then reveals that many uniformly random positions with GT labels.
#
# Override SIZES or any exported training knob from the shell as needed, e.g.:
#   SIZES="1_7 4" dataset=sftdatasetv3 bash scripts/launch_train_gt_random.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NUM_GPUS=${NUM_GPUS:-8}
export global_bs=${global_bs:-16}
export micro_bs=${micro_bs:-2}
export dataset=${dataset:-ultrachat_200k}
export input_prep_mode=${input_prep_mode:-bd_packed}
export loss_type=${loss_type:-kd_diff_sum}
export reveal_mode=gt_random
export gt_random_topk=${gt_random_topk:-1,2,3,4}
export mrp_steps=${mrp_steps:-1}
export loss_weight_mode=${loss_weight_mode:-uniform}
export mrp_init_std=${mrp_init_std:-0.2}
export LR=${LR:-1e-3}
export freeze_backbone=${freeze_backbone:-True}
export freeze_lm_head=${freeze_lm_head:-True}
export override=${override:-False}
export overwrite_cache=${overwrite_cache:-false}
export preprocessing_num_workers=${preprocessing_num_workers:-8}

SIZES=${SIZES:-"1_7"}
DENSE_MODEL_CODE_DIR="${REPO_ROOT}/src/models/SDAR-1.7B-MRP"
HF_MODULE_CACHE_DIR="${HF_MODULE_CACHE_DIR:-${HOME}/.cache/huggingface/modules/transformers_modules}"

sync_dense_model_code() {
  local model_path="$1"
  local cache_name
  cache_name="$(basename "${model_path}")"

  if [[ ! -d "${model_path}" ]]; then
    echo "[launch_train_gt_random] WARNING: MODEL_PATH does not exist yet: ${model_path}" >&2
    return 0
  fi

  cp "${DENSE_MODEL_CODE_DIR}"/*.py "${model_path}/"
  if [[ -d "${HF_MODULE_CACHE_DIR}/${cache_name}" ]]; then
    cp "${DENSE_MODEL_CODE_DIR}"/*.py "${HF_MODULE_CACHE_DIR}/${cache_name}/"
  fi
}

for size in ${SIZES}; do
  export MODEL_PATH="checkpoints/sdar-${size}b-mrp-3lyr"
  export MODEL_NAME="mrp_sdar-${size}b-chat-b16-3lyr"
  sync_dense_model_code "${MODEL_PATH}"

  echo "[$(date)] Launching ${MODEL_NAME} reveal_mode=${reveal_mode} gt_random_topk=${gt_random_topk} mrp_steps=${mrp_steps}"
  nohup-queue run --num-gpus "${NUM_GPUS}" "${SCRIPT_DIR}/train_sdar_mrp.sh"
done

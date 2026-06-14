#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate opencompass
export PYTHONNOUSERSITE=1
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUR_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src/eval/opencompass:${PYTHONPATH:-}"

# Case-insensitive boolean check
is_true() {
  local value="${1,,}"  # Convert to lowercase
  [[ "$value" == "true" || "$value" == "yes" || "$value" == "1" ]]
}

# Pick N least-utilized GPUs (by memory.used) and emit a comma-separated index
# list suitable for CUDA_VISIBLE_DEVICES. Requires nvidia-smi.
get_free_gpus() {
  local n=${1:-1}
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n \
    | head -n "${n}" \
    | awk -F',' '{gsub(/ /,"",$1); print $1}' \
    | paste -sd, -
}

export EVAL_DIR="${REPO_ROOT}/src/eval/opencompass"

export NUM_GPUS=${NUM_GPUS:-1}
# If the user did not pin GPUs explicitly, auto-pick the NUM_GPUS least-busy ones.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(get_free_gpus "${NUM_GPUS}")
fi
echo "Running evaluation on GPU ${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"

# Generation kwargs (override via env or inline: MRP_STEPS=3 bash ...)
# MODEL_PATH can be a local checkpoint dir or a HuggingFace Hub identifier.
export MODEL_PATH=${MODEL_PATH:-"heavyball/sdar-4b-mrp-3lyr"}
export GEN_LENGTH=${GEN_LENGTH:-4096}
export BLOCK_LENGTH=${BLOCK_LENGTH:-16}
export DENOISING_STEPS=${DENOISING_STEPS:-16}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TOP_K=${TOP_K:-1}
export TOP_P=${TOP_P:-1.0}
export THRESHOLD=${THRESHOLD:-0.85}
export REMASKING_STRATEGY=${REMASKING_STRATEGY:-'low_confidence_static'}
export MRP_STEPS=${MRP_STEPS:-0}
# Pre-sampling MRP logit aggregation: none | logits_sum
export MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE:-'logits_sum'}
# Inference mode selector:
#   none            — direct MRP decoding after the target step.
#   spec            — vanilla within-block speculative decoding.
#   residual_remask — MRP-as-vetoer: the target unmasks aggressively (set a low
#                     THRESHOLD), then a committed token is remasked when its MRP
#                     logit-space residual |C[pos, committed_id]| exceeds
#                     RESIDUAL_DELTA. MRP never commits. Requires MRP_STEPS>0.
export MRP_VERIFY_MODE=${MRP_VERIFY_MODE:-'none'}
# Per-token veto threshold for MRP_VERIFY_MODE=residual_remask. Default 1.0.
export RESIDUAL_DELTA=${RESIDUAL_DELTA:-1.0}
# Scale on the per-iter MRP correction term in logits_sum (A + scale*C). Default 1.0.
export MRP_LOGIT_SUM_SCALE=${MRP_LOGIT_SUM_SCALE:-1.0}
# MRP iters can use a different remasking strategy / per-iter transfer count
# than the target denoising steps. Defaults mirror the model-side defaults:
# low_confidence_static with 1 token per MRP iter.
export MRP_REMASKING_STRATEGY=${MRP_REMASKING_STRATEGY:-'low_confidence_static'}
export MRP_NUM_TRANSFER_PER_ITER=${MRP_NUM_TRANSFER_PER_ITER:-1}

# RNG seed applied by the eval config (torch.manual_seed). Fixed default so
# repeated evals of the same ckpt are reproducible; override per-run by setting
# SEED=... in the environment.
export SEED=${SEED:-0}

export PROFILE_TPF=${PROFILE_TPF:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
# Optional deterministic subset (first-N examples per dataset). Empty/0 = full set.
export SUBSET_SIZE=${SUBSET_SIZE:-}
export NUM_FEW_SHOT=${NUM_FEW_SHOT:-0}
export DATASET_STR=${task:-'gsm8k'}
export GSM8K_SETTING=${GSM8K_SETTING:-'0shot_cot'}

# datasets=(
#     # 'mmlu'
#     'gsm8k'
#     # 'humaneval'
#     # 'mbpp'
#     # 'math'
#     # 'mathbench'
#     # 'ifeval'
#     # 'triviaqa'
# )

# export DATASET_STR=$(IFS=, ; echo "${datasets[*]}")
echo "Evaluating on datasets: ${DATASET_STR}"

run_eval() {
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "Using generation kwargs: GEN_LENGTH=${GEN_LENGTH}, BLOCK_LENGTH=${BLOCK_LENGTH}, DENOISING_STEPS=${DENOISING_STEPS}, TEMPRATURE=${TEMPERATURE}, TOP_K=${TOP_K}, TOP_P=${TOP_P}"
    echo "  THRESHOLD=${THRESHOLD}, REMASKING_STRATEGY=${REMASKING_STRATEGY}, MRP_STEPS=${MRP_STEPS}"
    echo "  MRP_LOGIT_MIX_MODE=${MRP_LOGIT_MIX_MODE}, MRP_VERIFY_MODE=${MRP_VERIFY_MODE}, MRP_LOGIT_SUM_SCALE=${MRP_LOGIT_SUM_SCALE}"
    echo "  RESIDUAL_DELTA=${RESIDUAL_DELTA}, MRP_REMASKING_STRATEGY=${MRP_REMASKING_STRATEGY}, MRP_NUM_TRANSFER_PER_ITER=${MRP_NUM_TRANSFER_PER_ITER}"
    echo "  SEED=${SEED}, PROFILE_TPF=${PROFILE_TPF}, BATCH_SIZE=${BATCH_SIZE}, SUBSET_SIZE=${SUBSET_SIZE:-(full)}"
    REUSE_FLAG=""
    if is_true "${REUSE:-true}"; then
        REUSE_FLAG="--reuse"
    fi
    python ${EVAL_DIR}/run.py ${EVAL_DIR}/configs/eval_sdar_hf.py ${REUSE_FLAG}
}

run_eval

#!/bin/bash

set -euo pipefail

# Make sure all torchrun ranks use the same Python/torch/triton.
source "$(conda info --base)/etc/profile.d/conda.sh"
# The traceback shows training is running under this env (py3.11). Adjust if you use a different one.
conda activate llamafactory_sdar
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CUR_DIR}/.." && pwd)"

# Propagate any prototype edits under src/models/ into the checkpoint dirs
# before training. Skip with SKIP_MODELING_SYNC=1 if you know files are fresh.
if [[ "${SKIP_MODELING_SYNC:-0}" != "1" ]]; then
    bash "${CUR_DIR}/sync_modeling_files.sh"
fi

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

export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton
export TORCHINDUCTOR_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/torchinductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Opt-in deterministic training. Set DETERMINISTIC=1 to make blue/orange/etc.
# produce bit-identical head-0 weights across reveal_mode branches. Slower by
# ~10-30% depending on ops. The actual torch.use_deterministic_algorithms call
# lives in src/.../llamafactory/launcher.py (must run before any torch op).
export DETERMINISTIC=${DETERMINISTIC:-1}
if [[ "${DETERMINISTIC}" == "1" ]] || [[ "${DETERMINISTIC,,}" == "true" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=":4096:8"
    echo "[train_sdar_mrp] DETERMINISTIC=1: CUBLAS_WORKSPACE_CONFIG=:4096:8"
fi

PROJ_DIR="${REPO_ROOT}/src/training/sdar"
NUM_GPUS=${NUM_GPUS:-4}
# If the user did not pin GPUs explicitly, auto-pick the NUM_GPUS least-busy ones.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(get_free_gpus "${NUM_GPUS}")
fi
echo "Using GPUs ${CUDA_VISIBLE_DEVICES} for training (NUM_GPUS=${NUM_GPUS})"

MODEL_PATH=${MODEL_PATH:-"heavyball/sdar-1.7b-mrp-3lyr"}
MODEL_NAME=${MODEL_NAME:-"mrp_sdar-1_7b-chat-3lyr"}
MODEL_CACHE_NAME=${MODEL_CACHE_NAME:-$(basename "${MODEL_PATH}")}

LR=${LR:-1e-3}
lr_scheduler_type=cosine_with_min_lr
num_epochs=1
global_bs=${global_bs:-16}
micro_bs=${micro_bs:-1}
grad_accum=$((global_bs / micro_bs / NUM_GPUS))
seq_len=4096
block_length=16
# dataset=eagle_chat
freeze_backbone=${freeze_backbone:-True}
freeze_lm_head=${freeze_lm_head:-True}
loss_type=${loss_type:-kd_diff_sum}
input_prep_mode=${input_prep_mode:-bd_packed}
kd_temperature=${kd_temperature:-1.0}
kd_reverse_kl=${kd_reverse_kl:-False}
mrp_init_std=${mrp_init_std:-0.2}
init_eh_proj=${init_eh_proj:-False}
mrp_init_from_target_layers=${mrp_init_from_target_layers:-False}
mrp_proj_lr=${mrp_proj_lr:-}

reveal_mode=${reveal_mode:-gt}
gt_topk=${gt_topk:-1}
gt_random_topk=${gt_random_topk:-1}

mrp_steps=${mrp_steps:-1}
loss_weight_mode=${loss_weight_mode:-uniform}

sync_local_dense_model_code() {
  local dense_src="${REPO_ROOT}/src/models/SDAR-1.7B-MRP"
  local hf_cache_dir="${HF_MODULE_CACHE_DIR:-${HOME}/.cache/huggingface/modules/transformers_modules}"

  if [[ ! -d "${dense_src}" || ! -d "${MODEL_PATH}" || ! -f "${MODEL_PATH}/modeling_sdar_mrp.py" ]]; then
    return 0
  fi

  cp "${dense_src}"/*.py "${MODEL_PATH}/"
  if [[ -d "${hf_cache_dir}/${MODEL_CACHE_NAME}" ]]; then
    cp "${dense_src}"/*.py "${hf_cache_dir}/${MODEL_CACHE_NAME}/"
  fi
}

# Local checkpoint directories carry their own trust_remote_code Python files.
# Keep them in sync so newly added training options are visible to AutoModel.
sync_local_dense_model_code

# Multi-step MRP training reuses the MRP head's weights across K forward paths,
# so in a single backward each MRP parameter has K autograd edges. DeepSpeed
# ZeRO-1/2 fire a per-parameter grad-reduce hook on each edge and assert
# "parameter N has already been reduced" on the second firing. Workarounds:
#   - z0 (pure DDP under the DS engine): one all-reduce per param at end of
#     backward, AFTER autograd sums all K edge contributions into .grad.
#     Fastest option and works because only the MRP head is trained (tiny).
#   - z3: partition-owned grad accumulation, correct but slow.
# Default to z0 when K>=2 (backbone and lm_head are typically frozen); let the
# user override explicitly (e.g., deepspeed=z3 if the trainable slice is too big
# for DDP). If the user already pinned `deepspeed`, respect it.
if [[ "${mrp_steps}" -ge 2 && -z "${deepspeed:-}" ]]; then
  deepspeed=z0
  echo "[train_sdar_mrp] mrp_steps=${mrp_steps} -> forcing deepspeed=z0 (pure DDP; ZeRO-2 incompatible with shared-weight multi-edge backward). Override with deepspeed=z3 if the trainable slice needs sharding."
fi
dataset=${dataset:-sftdatasetv3}
max_samples=${MAX_SAMPLES:-}
preprocessing_num_workers=${preprocessing_num_workers:-32}
preprocessing_batch_size=${preprocessing_batch_size:-10000}
overwrite_cache=${overwrite_cache:-false}
resume_from_checkpoint=${resume_from_checkpoint:-}
disable_shuffling=${disable_shuffling:-False}
do_eval=${do_eval:-True}
# dataset=open_r1_math
# dataset=sftdatasetv3


# IMPORTANT: LLaMAFactory will *load* tokenized data from disk if tokenized_path exists,
# and then it will ignore cutoff_len/neat_packing/etc. To make seq_len changes effective,
# use a tokenized_path that depends on seq_len (or remove tokenized_path entirely).
TOKENIZED_PATH=${PROJ_DIR}/tokenized_cache/${dataset}/seq_${seq_len}
if [[ -n "${max_samples}" ]]; then
  TOKENIZED_PATH=${TOKENIZED_PATH}/max_samples_${max_samples}
fi

export TORCH_COMPILE_DISABLE=0
# export TORCHINDUCTOR_DISABLE=1

get_random_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])"
}

# Determine mrp_logit_mix_mode based on loss_type.
# ('sum' losses train A+C≈B, so logits_sum at inference is the matched mode.)
determine_mrp_logit_mix_mode() {
  if [[ "${loss_type}" == *"sum"* ]]; then
    echo "logits_sum"
  else
    echo "none"
  fi
}

RUN_ID_DIR="./wandb/run_ids"
mkdir -p "${RUN_ID_DIR}"

# Persist wandb run id by run_name so reruns resume the same W&B run.
# W&B uses WANDB_RUN_ID + WANDB_RESUME to decide whether to resume.
# SAFE_RUN_NAME=$(echo "${run_name}" | tr '/[:space:]' '__')

export WANDB_RESUME="allow"
# Set WANDB_API_KEY in your environment (e.g. `export WANDB_API_KEY=...` in
# ~/.bashrc) before running. If it is absent, disable W&B cleanly.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="disabled"
  REPORT_TO="none"
  echo "[train_sdar_mrp] WANDB_API_KEY not set; disabling W&B logging."
elif [[ "${WANDB_MODE:-}" == "disabled" ]]; then
  REPORT_TO="none"
else
  REPORT_TO="wandb"
fi
export WANDB_PROJECT=${WANDB_PROJECT:-"mrp_sdar"}

run_exp() {
  # Build loss suffix and output_dir
  loss_suffix="loss_${loss_type}"
  case "${loss_type}" in
    ce)          ;;
    ce_sum)      ;;
    kd_diff)     loss_suffix+="_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    kd_diff_sum) loss_suffix+="_T_${kd_temperature}_r_${kd_reverse_kl}" ;;
    mse)         ;;
    mse_sum)     ;;
  esac

  # Append init std to loss suffix
  loss_suffix+="_init_std_${mrp_init_std}"

  # Append structured init and proj lr if set
  if is_true "${init_eh_proj}"; then
    loss_suffix+="_ehproj"
  fi
  if [[ -n "${mrp_proj_lr}" ]]; then
    loss_suffix+="_plr_${mrp_proj_lr}"
  fi

  # Append reveal_mode to loss suffix. Each mode gets its own parameter
  # appended as "_{value}" so run_name / output_dir stay unique across
  # (mode, knob) combinations.
  loss_suffix+="_reveal_${reveal_mode}"
  case "${reveal_mode}" in
    gt)              loss_suffix+="_${gt_topk}" ;;
    gt_random)
      gt_random_suffix=$(
        sorted=$(
          printf '%s' "${gt_random_topk}" \
            | tr -d '[] ' \
            | tr ',' '\n' \
            | sed '/^$/d' \
            | sort -n
        )
        first=$(echo "$sorted" | head -1)
        last=$(echo "$sorted"  | tail -1)
        count=$(echo "$sorted" | wc -l | tr -d ' ')
        expected=$(( last - first + 1 ))
        # Compact to "first-last" iff values are exactly first,first+1,...,last
        if [[ "$count" -eq "$expected" && "$first" -eq 1 && "$count" -gt 1 ]]; then
          echo "${first}-${last}"
        else
          echo "$sorted" | paste -sd '_' -
        fi
      )
      loss_suffix+="_${gt_random_suffix}"
      ;;
  esac

  # Append mrp_steps to the suffix when K >= 2 so K=1 runs keep their historical
  # paths and K>=2 runs get unique checkpoint dirs per K.
  if [[ "${mrp_steps}" -ge 2 ]]; then
    loss_suffix+="_K_${mrp_steps}_lwm_${loss_weight_mode}"
  fi

  _l1=checkpoints/${MODEL_NAME}/ds_${dataset}_seq_${seq_len}_bs_${global_bs}_ep_${num_epochs}_freezebb_${freeze_backbone}_freezelm_${freeze_lm_head}
  _l2=${loss_suffix}_prep_${input_prep_mode}_lr_${LR}_${lr_scheduler_type}
  output_dir=${_l1}/${_l2}
  output_dir=${output_dir//./_}

  run_name=${MODEL_NAME}_ds_${dataset}_seq_${seq_len}_bs_${global_bs}_ep_${num_epochs}_freezebb_${freeze_backbone}_freezelm_${freeze_lm_head}_${_l2}
  run_name=${run_name//./_}

  RUN_ID_FILE="${RUN_ID_DIR}/${run_name}"

  if [[ -s "${RUN_ID_FILE}" ]]; then
    export WANDB_RUN_ID=$(cat "${RUN_ID_FILE}")
  else
    export WANDB_RUN_ID=$(python - <<'PY'
import uuid
try:
    import wandb
    print(wandb.util.generate_id())
except Exception:
    print(uuid.uuid4().hex[:8])
PY
)
    tmpfile="${RUN_ID_FILE}.tmp"
    printf "%s" "${WANDB_RUN_ID}" > "${tmpfile}"
    mv "${tmpfile}" "${RUN_ID_FILE}"
  fi

  echo "[wandb] mode=${WANDB_MODE:-online} report_to=${REPORT_TO}"
  echo "[wandb] run_name=${run_name}"
  echo "[wandb] run_id=${WANDB_RUN_ID} (${WANDB_RESUME})"

  torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node ${NUM_GPUS} \
    --master_port $(get_random_port) \
    ${PROJ_DIR}/llama_factory_sdar/src/llamafactory/launcher.py \
    ${PROJ_DIR}/llama_factory_sdar/examples/train_full_sdar/sdar_4b/sdar_4b_math_cot_full.yaml \
    model_name_or_path=${MODEL_PATH} \
    trust_remote_code=true \
    deepspeed=${PROJ_DIR}/llama_factory_sdar/examples/deepspeed/ds_${deepspeed:-z2}_config.json \
    dataset_dir=${PROJ_DIR}/llama_factory_sdar/data \
    dataset=${dataset} \
    tokenized_path=${TOKENIZED_PATH} \
    ${max_samples:+max_samples=${max_samples}} \
    overwrite_cache=${overwrite_cache} \
    output_dir=${output_dir} \
    overwrite_output_dir=${override:-False} \
    learning_rate=${LR} \
    lr_scheduler_type=${lr_scheduler_type} \
    num_train_epochs=${num_epochs} \
    per_device_train_batch_size=${micro_bs} \
    gradient_accumulation_steps=${grad_accum} \
    cutoff_len=${seq_len} \
    block_length=${block_length} \
    gradient_checkpointing=${gradient_checkpointing:-true} \
    logging_steps=${logging_steps:-10} \
    save_steps=${save_steps:-250} \
    save_total_limit=1 \
    report_to=${REPORT_TO} \
    run_name=${run_name} \
    input_prep_mode=${input_prep_mode} \
    freeze_backbone=${freeze_backbone} \
    freeze_lm_head=${freeze_lm_head} \
    loss_type=${loss_type} \
    kd_temperature=${kd_temperature} \
    kd_reverse_kl=${kd_reverse_kl} \
    mrp_init_std=${mrp_init_std} \
    init_eh_proj=${init_eh_proj} \
    mrp_init_from_target_layers=${mrp_init_from_target_layers} \
    ${mrp_proj_lr:+mrp_proj_lr=${mrp_proj_lr}} \
    reveal_mode=${reveal_mode} \
    gt_topk=${gt_topk} \
    gt_random_topk=${gt_random_topk} \
    mrp_steps=${mrp_steps} \
    loss_weight_mode=${loss_weight_mode} \
    preprocessing_num_workers=${preprocessing_num_workers} \
    preprocessing_batch_size=${preprocessing_batch_size} \
    ${resume_from_checkpoint:+resume_from_checkpoint=${resume_from_checkpoint}} \
    disable_shuffling=${disable_shuffling}

  # Post-training evaluation
  if is_true "${do_eval}"; then
    echo "[$(date)] Starting evaluation..."
    mrp_logit_mix_mode=$(determine_mrp_logit_mix_mode)
    export MODEL_PATH="${output_dir}"
    export MRP_LOGIT_MIX_MODE="${mrp_logit_mix_mode}"
    export MRP_VERIFY_MODE="none"
    export REUSE=false
    # Use nohup-queue so eval runs in a fresh shell without the training's
    # multi-GPU CUDA_VISIBLE_DEVICES (which would otherwise defeat NUM_GPUS=1
    # and trigger device_map='auto' sharding → cross-device error in generate).
    NUM_GPUS=1 MRP_STEPS=1 nohup-queue run "${CUR_DIR}/eval_opencompass_hf.sh"
    NUM_GPUS=1 MRP_STEPS=2 nohup-queue run "${CUR_DIR}/eval_opencompass_hf.sh"
  fi
}

run_exp
exit $?

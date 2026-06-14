import os
import pdb
import torch
import warnings
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import BD3withChatTemplate, SDARMRPwithChatTemplate

gsm8k_setting = os.getenv("GSM8K_SETTING", "0shot_cot").lower()

with read_base():
    # datasets setting
    from ..opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    # math
    from ..opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import (
        gsm8k_datasets as gsm8k_4shot_datasets,
    )
    from ..opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_a58960 import (
        gsm8k_datasets as gsm8k_0shot_cot_datasets,
    )
    from ..opencompass.configs.datasets.gsm8k.gsm8k_0shot_nocot_gen_6cbf22 import (
        gsm8k_datasets as gsm8k_0shot_nocot_datasets,
    )
    from ..opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import math_datasets
    from ..opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from ..opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 import sanitized_mbpp_datasets
    from ..opencompass.configs.datasets.triviaqa.triviaqa_wiki_1shot_gen_20a989 import triviaqa_datasets
    from ..opencompass.configs.datasets.gpqa.gpqa_0shot_nocot_gen_772ea0 import gpqa_datasets
    from ..opencompass.configs.datasets.ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets

    from ..opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import (
        mathbench_datasets,
    )
    # Instruction Following
    from ..opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import (
        ifeval_datasets,
    )
    # summarizer
    from ..opencompass.configs.summarizers.internlm2_keyset import summarizer
    from ..opencompass.configs.summarizers.groups.mathbench_v1_2024 import (
        mathbench_2024_summary_groups,
    )
    from ..opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups

# summarizer
summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)

if gsm8k_setting == "4shot":
    gsm8k_datasets = gsm8k_4shot_datasets
elif gsm8k_setting == "0shot_cot":
    gsm8k_datasets = gsm8k_0shot_cot_datasets
elif gsm8k_setting == "0shot_nocot":
    gsm8k_datasets = gsm8k_0shot_nocot_datasets
else:
    raise ValueError(
        f"Unsupported GSM8K_SETTING={gsm8k_setting!r}. "
        "Expected one of: '4shot', '0shot_cot', '0shot_nocot'."
    )

summary_groups.append(
    {
        'name': 'Mathbench',
        'subsets': ['mathbench-a (average)', 'mathbench-t (average)'],
    },
)

# Summarizer
summarizer = dict(
    dataset_abbrs=[
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        "Math Calculation",
        ["gsm8k", "accuracy"],
        ['Mathbench', 'naive_average'],
        ['math_prm800k_500', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu', 'naive_average'],
        ['triviaqa_wiki_1shot', 'score'],
        ['GPQA_diamond', 'accuracy'],
        ['ARC-c', 'accuracy'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        '',
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
    ],
    summary_groups=summary_groups,
)

datasets_dict = {
    "mmlu": mmlu_datasets,
    "gsm8k": gsm8k_datasets,
    "humaneval": humaneval_datasets,
    "mbpp": sanitized_mbpp_datasets,
    "math": math_datasets,
    "mathbench": mathbench_datasets,
    "ifeval": ifeval_datasets,
    "triviaqa": triviaqa_datasets,
    "gpqa": gpqa_datasets,
    "arc_c": ARC_c_datasets,
}

datasets = []
dataset_str = os.getenv("DATASET_STR", "all")
if dataset_str == "all" or "gsm8k" in dataset_str.split(','):
    warnings.warn(
        "GSM8K in this config is controlled by `GSM8K_SETTING`, "
        f"currently set to `{gsm8k_setting}`. "
        "Supported values: `4shot`, `0shot_cot`, `0shot_nocot`.",
        UserWarning,
    )
if dataset_str == "all":
    datasets = [each for k, v in datasets_dict.items() for each in v]
else:
    datasets = [each for k, v in datasets_dict.items() if k in dataset_str.split(',') for each in v]

# import pdb; pdb.set_trace()
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = int(os.getenv("BATCH_SIZE", 1))

# Optional deterministic subset for fast speedup/profiling sweeps. When
# SUBSET_SIZE is set (>0), restrict each dataset's test split to its first N
# examples via the DatasetReader's `test_range` slice (string form => first-N,
# reproducible). Leave unset/0 for the full set.
_subset_raw = os.getenv("SUBSET_SIZE", "").strip()
subset_size = int(_subset_raw) if _subset_raw else 0
if subset_size > 0:
    for dataset in datasets:
        dataset['reader_cfg']['test_range'] = f"[:{subset_size}]"

# model
# model_configs = [
#     ("SDAR-4B-Chat-b4-thr0_95", "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/SDAR-4B-Chat", 4, 0.95, 1),
#     ("SDAR-8B-Chat-b4-thr0_95", "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/SDAR-8B-Chat", 4, 0.95, 1),
#     ("", 
#         "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/mrp_sdar-4b-chat_ds_ultrachat_200k/lr_1e-4_cosine_with_min_lr_epoch_1_bs_8_seq_len_3072_unmask_True",
#         4, 0.95, 1
#     ),
#     ("", 
#         "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/mrp_sdar-4b-chat_ds_ultrachat_200k/lr_1e-5_cosine_with_min_lr_epoch_1_bs_8_seq_len_3072_unmask_True",
#         4, 0.95, 1
#     ),
#     ("", 
#         "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/mrp_sdar-4b-chat_ds_ultrachat_200k/lr_3e-4_cosine_with_min_lr_epoch_1_bs_8_seq_len_3072_unmask_True",
#         4, 0.95, 1
#     ),
#     ("", 
#         "/mnt/weka/home/yucheng/yufeng/dflash-sdar/checkpoints/mrp_sdar-4b-chat_ds_ultrachat_200k/lr_3e-5_cosine_with_min_lr_epoch_1_bs_8_seq_len_3072_unmask_True",
#         4, 0.95, 1
#     ),
#     # ("SDAR-1.7B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-1.7B-Chat", 4, 0.95, 1),
#     # ("SDAR-1.7B-Chat-b4-thr1_00", "xxx/Models/SDAR/SDAR-1.7B-Chat", 4, 1.0, 1),
#     # ("SDAR-4B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 0.95, 1),
#     # ("SDAR-4B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 1.0, 1),
#     # ("SDAR-8B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 0.95, 1),
#     # ("SDAR-8B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 1.0, 1),
#     # ("SDAR-30B-A3B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 0.95, 1),
#     # ("SDAR-30B-A3B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 1.0, 1)
# ]



models = []

path = os.environ['MODEL_PATH']

# Optional RNG seed for reproducible eval (SEED env var). The actual
# torch.manual_seed call happens in SDARMRPwithChatTemplate.__init__ since
# mmengine lazy-imports torch at config-parse time and can't evaluate it here.
# We only read SEED here to fold it into the output `abbr` so runs with
# different seeds don't collide on disk.
_seed_raw = os.getenv('SEED', '')

gen_length = int(os.getenv('GEN_LENGTH', '4096'))
block_length = int(os.getenv('BLOCK_LENGTH', '16'))
denoising_steps = int(os.getenv('DENOISING_STEPS', block_length))
temperature = float(os.getenv('TEMPERATURE', '1.0'))
topk = int(os.getenv('TOP_K', '1'))
topp = float(os.getenv('TOP_P', '1.0'))
threshold = float(os.getenv('THRESHOLD', '0.85'))
remasking_strategy = os.getenv('REMASKING_STRATEGY', 'low_confidence_static')
mrp_steps = int(os.getenv('MRP_STEPS', '0'))
# Pre-sampling MRP logit aggregation. One of: none | logits_sum
mrp_logit_mix_mode = os.getenv('MRP_LOGIT_MIX_MODE', 'none')
# Inference mode selector. One of:
#   none            — direct MRP decoding after the target step.
#   spec            — vanilla within-block speculative decoding.
#   residual_remask — aggressive target unmask + MRP residual veto.
mrp_verify_mode = os.getenv('MRP_VERIFY_MODE', 'none')
# Scale on the per-iter MRP correction term inside 'logits_sum' mode (both
# the logits accumulator and the hidden-state accumulator). Default 1.0 =
# original A+C; set to 2.0 for A+2C. Ignored by other mix modes.
mrp_logit_sum_scale = float(os.getenv('MRP_LOGIT_SUM_SCALE', '1.0'))
# Per-token veto threshold for MRP_VERIFY_MODE='residual_remask': the target
# unmasks aggressively (low THRESHOLD), then a committed token is remasked when
# |C[pos, committed_id]| > RESIDUAL_DELTA, where C is MRP's raw logit-space
# residual. Larger delta = looser veto. Default 1.0.
residual_delta = float(os.getenv('RESIDUAL_DELTA', '1.0'))
# MRP iters may use a different remasking strategy / per-iter transfer count
# than the target denoising steps. Defaults match the model-side defaults.
mrp_remasking_strategy = os.getenv('MRP_REMASKING_STRATEGY', 'low_confidence_static')
mrp_num_transfer_per_iter = int(os.getenv('MRP_NUM_TRANSFER_PER_ITER', '1'))

num_gpus = int(os.getenv("NUM_GPUS", 1))

if "mrp" in path.lower():
    cls = SDARMRPwithChatTemplate
else:
    cls = BD3withChatTemplate
    
# Walk up the checkpoint path until we find a segment that contains 'sdar',
# joining all intermediate segments with '_'. This handles multi-level
# checkpoint dirs such as
#   checkpoints/mrp_sdar-1_7b-chat-6lyr/ds_<...>/loss_<...>
# where the 'sdar' marker lives two or more levels above the leaf.
_parts = [p for p in path.rstrip("/").split("/") if p]
_end = len(_parts) - 1
_start = _end
while _start > 0 and 'sdar' not in _parts[_start].lower():
    _start -= 1
model_name = "_".join(p.lower() for p in _parts[_start:_end + 1])

# Short tag for the remasking strategy (encoded in the eval output dir so
# runs with different strategies do not collide on disk). Mirrors the four
# branches in the inference loop:
#   low_confidence_static  -> static
#   low_confidence_dynamic -> dynamic
#   entropy_bounded        -> entropy
#   sequential             -> seq
_REMASKING_SHORT = {
    'low_confidence_static': 'static',
    'low_confidence_dynamic': 'dynamic',
    'entropy_bounded': 'entropy',
    'sequential': 'seq',
}
_rm_short = _REMASKING_SHORT.get(remasking_strategy.lower(), remasking_strategy.lower())

abbr = (
    f"{model_name}/L_{gen_length}_bl_{block_length}_dsteps_{denoising_steps}_thr_{threshold}"
    f"_rm_{_rm_short}_T_{temperature}_topk_{topk}_topp_{topp}_mrp_steps_{mrp_steps}"
)
# Aggregation and verify are now independent; encode each when non-default.
if mrp_logit_mix_mode != 'none':
    if mrp_logit_mix_mode == 'logits_sum':
        abbr = f"{abbr}_mix_{mrp_logit_mix_mode}"
        if mrp_logit_sum_scale != 1.0:
            abbr = f"{abbr}_s{mrp_logit_sum_scale}"
    else:
        raise Exception(f"Unknown mrp_logit_mix_mode: {mrp_logit_mix_mode!r}")
if mrp_verify_mode != 'none':
    if mrp_verify_mode == 'spec':
        # Vanilla speculative decoding (panel (d)): K = mrp_steps; no extra knob.
        abbr = f"{abbr}_v_spec"
    elif mrp_verify_mode == 'residual_remask':
        # Aggressive target unmask + MRP residual veto; delta is the knob.
        abbr = f"{abbr}_v_rremask_d_{residual_delta}"
    else:
        raise Exception(f"Unknown mrp_verify_mode: {mrp_verify_mode!r}")

# Encode MRP remasking strategy / per-iter transfer count when MRP iters are
# active, so runs with different MRP-side settings do not collide on disk.
if mrp_steps > 0:
    _mrp_rm_short = _REMASKING_SHORT.get(
        mrp_remasking_strategy.lower(), mrp_remasking_strategy.lower()
    )
    abbr = f"{abbr}_mrp_rm_{_mrp_rm_short}_n_{mrp_num_transfer_per_iter}"

# Seed goes into abbr so runs with different seeds do not collide on disk.
if _seed_raw.strip():
    abbr = f"{abbr}_seed{int(_seed_raw)}"

# Subset size goes into abbr so partial-set profiling runs do not collide with
# full-set accuracy runs on disk.
if subset_size > 0:
    abbr = f"{abbr}_sub{subset_size}"

models.append(
    dict(
    type=cls,
    abbr=abbr,
    path=path,
    run_cfg=dict(num_gpus=num_gpus),
    generation_kwargs=dict(
        mask_id=151669,
        gen_length=int(gen_length),
        block_length=block_length,
        denoising_steps=denoising_steps,
        temperature=temperature,
        top_k=topk,
        top_p=topp,
        # cfg_scale=0.0,
        remasking=remasking_strategy,
        threshold=threshold,
        mrp_steps=mrp_steps,
        mrp_logit_mix_mode=mrp_logit_mix_mode,
        mrp_verify_mode=mrp_verify_mode,
        residual_delta=residual_delta,
        mrp_logit_sum_scale=mrp_logit_sum_scale,
        mrp_remasking_strategy=mrp_remasking_strategy,
        mrp_num_transfer_per_iter=mrp_num_transfer_per_iter,
    ),
    model_kwargs=dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
)
)

GPUS = int(os.getenv("NUM_GPUS", 1))
infer = dict(
    # 同时启动num_workers个任务并行
    partitioner=dict(
        # type=NumWorkerPartitioner,
        # num_worker=GPUS,  # 划分完成后的任务数 / 预期能有的 worker 数
        # force_rebuild=True
        type=NaivePartitioner,
        n=len(datasets),
        # num_worker=len(datasets),
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=GPUS,  # 最大并行运行进程数
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=5
    )
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask, dump_details=True)),
)

work_dir = f'./outputs/opencompass/{abbr}'

del os, torch, pdb, warnings

import argparse
import pdb
import re
import time
import torch
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


NUM_SAMPLES = 8
BATCH_SIZE = 1
assert NUM_SAMPLES % BATCH_SIZE == 0, f"num_samples ({NUM_SAMPLES}) must be divisible by batch_size ({BATCH_SIZE})"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(42)

gsm8k = load_dataset("gsm8k", "main", split="test")
questions = [gsm8k[i]["question"] for i in range(NUM_SAMPLES)]
gt_raw_answers = [gsm8k[i]["answer"] for i in range(NUM_SAMPLES)]


def gsm8k_dataset_postprocess(text: str) -> str:
    """Extract ground truth answer (text after '#### ')."""
    return text.split('#### ')[1].replace(',', '')


def gsm8k_postprocess(text: str) -> str:
    """Extract predicted answer from model output (mirrors opencompass logic)."""
    text = text.split('Question:')[0]
    boxed = re.search(r'\\boxed\s*\{([^}]*)\}', text)
    if boxed:
        inner = boxed.group(1).replace(',', '').replace('$', '').strip()
        nums = re.findall(r'\-?\d+\.?\d*', inner)
        if nums:
            return nums[0]
    ans_matches = re.findall(
        r'[Tt]he\s+answer\s+is\s*:?\s*[\$]?\s*(\-?\d[\d,]*\.?\d*)',
        text,
    )
    if ans_matches:
        return ans_matches[-1].replace(',', '')
    text = text.replace(',', '').replace('$', '')
    numbers = re.findall(r'\-?\d+\.?\d*', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]


def is_correct(pred: str, refer: str) -> bool:
    try:
        if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
            return True
    except Exception:
        pass
    return False


gt_answers = [gsm8k_dataset_postprocess(a) for a in gt_raw_answers]

# per-config results: config_label -> list of parsed predicted answers
config_results = {}
config_tps = {}

model_path = "heavyball/sdar-1.7b-mrp-3lyr"

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"model loaded from {model_path}")
print(f"loaded {len(questions)} GSM8K questions, batch_size={BATCH_SIZE}")

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for question in questions
]
for idx, question in enumerate(questions):
    print(f"question[{idx}]: {question}")

# Pre-tokenize all batches
batches = []
for start in range(0, NUM_SAMPLES, BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_inputs = tokenizer(prompts[start:end], padding=True, return_tensors="pt").to(model.device)
    batch_prompt_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
    batches.append((batch_inputs, batch_prompt_lengths))

def generate_with_profile(input_ids, attention_mask, prompt_lengths_batch, global_offset=0, **kwargs):
    start_time = time.time()
    outputs = model.generate(input_ids, attention_mask=attention_mask, **kwargs)
    elapsed_time = time.time() - start_time
    total_gen_tokens = 0
    parsed_answers = []

    for idx in range(outputs.shape[0]):
        prompt_length = int(prompt_lengths_batch[idx])
        answer_ids = outputs[idx, prompt_length:]
        valid_answer_ids = answer_ids[answer_ids != model.config.mask_token_id]
        answer = tokenizer.decode(valid_answer_ids, skip_special_tokens=True)
        answer_text = answer.replace("\n", "\\n")
        print(f"answer[{global_offset + idx}]: {answer_text}")
        total_gen_tokens += valid_answer_ids.numel()
        parsed_answers.append(gsm8k_postprocess(answer))

    tps = total_gen_tokens / elapsed_time
    print(f"Tokens generated: {total_gen_tokens}, Time taken: {elapsed_time:.2f} seconds, Tokens per second: {tps:.2f}")
    print("-" * 50)
    return parsed_answers, total_gen_tokens, elapsed_time


def run_config_all_batches(cfg_label, **gen_kwargs):
    """Run generate_with_profile over all batches and return accumulated parsed answers."""
    all_parsed = []
    total_tokens = 0
    total_time = 0.0
    for batch_idx, (batch_inputs, batch_prompt_lengths) in enumerate(tqdm(batches, desc="batches")):
        batch_parsed, batch_tokens, batch_time = generate_with_profile(
            batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            prompt_lengths_batch=batch_prompt_lengths,
            global_offset=batch_idx * BATCH_SIZE,
            **gen_kwargs,
        )
        all_parsed.extend(batch_parsed)
        total_tokens += batch_tokens
        total_time += batch_time
    overall_tps = total_tokens / total_time if total_time > 0 else 0.0
    print(f"[{cfg_label}] Overall: {total_tokens} tokens in {total_time:.2f}s = {overall_tps:.2f} tok/s")
    return all_parsed, overall_tps

# pdb.set_trace()

max_length = 1024
temperature = 1.0
top_k = 1
top_p = 1.0
# New split API (all knobs passed as top-level generate() kwargs):
#   mrp_logit_mix_mode    : pre-sampling MRP logit aggregation
#       (none | logits_add | logits_sum | logits_ema | logits_mix)
#   mrp_verify_mode       : post-sampling verification
#       (none | threshold | edit)
#   mrp_verify_alpha      : used when mrp_logit_mix_mode in
#                           {logits_ema, logits_mix}
#   mrp_verify_threshold  : used when mrp_verify_mode == 'threshold'
#   edit_threshold        : used when mrp_verify_mode == 'edit'
mrp_logit_mix_mode = 'logits_sum'
mrp_verify_mode = 'none'
mrp_verify_alpha = 0.5
mrp_verify_threshold = 0.5
edit_threshold = 0.9
remasking_strategy = "low_confidence_dynamic"
confidence_threshold = 0.95

# import pdb; pdb.set_trace()

print(tokenizer.eos_token_id)
# import pdb; pdb.set_trace()
print(model.config.mask_token_id)
print(tokenizer.decode(model.config.mask_token_id))

for mrp_steps in [0, 1, 2, 3]:
    cfg_label = f"mrp_steps={mrp_steps}"
    print(f"\n{'='*50} {cfg_label} {'='*50}")
    parsed, tps = run_config_all_batches(
        cfg_label,
        max_length=max_length,
        temperature=temperature,
        block_length=16,
        denoising_steps=16,
        top_k=top_k,
        top_p=top_p,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        stopping_criteria_idx=[tokenizer.eos_token_id],
        mask_id=model.config.mask_token_id,
        mrp_steps=mrp_steps,
        mrp_logit_mix_mode=mrp_logit_mix_mode,
        mrp_verify_mode=mrp_verify_mode,
        mrp_verify_alpha=mrp_verify_alpha,
        mrp_verify_threshold=mrp_verify_threshold,
        edit_threshold=edit_threshold,
    )
    config_results[cfg_label] = parsed
    config_tps[cfg_label] = tps

# --- Summary table ---
configs = list(config_results.keys())
col_w = 12

header = f"{'Q':>3}  {'GT':>{col_w}}" + "".join(f"  {c:>{col_w}}" for c in configs)
print("\n" + "=" * len(header))
print("GSM8K ANSWER SUMMARY")
print("=" * len(header))
print(header)
print("-" * len(header))
for i in range(NUM_SAMPLES):
    gt = gt_answers[i]
    row = f"{i:>3}  {gt:>{col_w}}"
    for cfg in configs:
        pred = config_results[cfg][i]
        marker = "✓" if is_correct(pred, gt) else "✗"
        cell = f"{marker}{pred}"
        row += f"  {cell:>{col_w}}"
    print(row)
print("-" * len(header))
# Accuracy row
acc_row = f"{'ACC':>3}  {'':>{col_w}}"
for cfg in configs:
    preds = config_results[cfg]
    n_correct = sum(is_correct(preds[i], gt_answers[i]) for i in range(NUM_SAMPLES))
    acc_str = f"{n_correct}/{NUM_SAMPLES}"
    acc_row += f"  {acc_str:>{col_w}}"
print(acc_row)
# TPS row
tps_row = f"{'TPS':>3}  {'':>{col_w}}"
for cfg in configs:
    tps_row += f"  {config_tps[cfg]:>{col_w}.1f}"
print(tps_row)
print("=" * len(header))
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def log_trainable_modules(model: "PreTrainedModel") -> None:
    r"""Print each parameter's name, shape, and trainability.

    For repeated backbone layers (e.g. model.layers.0, model.layers.1, ...),
    only the first layer is printed; the rest are collapsed into a single note.
    All other parameters are printed individually.  Only runs on rank 0.
    """
    if int(os.getenv("LOCAL_RANK", "0")) != 0:
        return

    import re

    lines = ["\nParameter trainability:"]
    skipped_layer_pattern: "str | None" = None
    skipped_count = 0

    def _flush_skipped():
        nonlocal skipped_count, skipped_layer_pattern
        if skipped_count > 0:
            lines.append(f"  ... ({skipped_count} more layers with the same structure)")
        skipped_count = 0
        skipped_layer_pattern = None

    for name, param in model.named_parameters():

        # Detect repeated layer blocks: any path segment that is a digit index
        # e.g. model.layers.3.self_attn.q_proj.weight -> prefix = "model.layers", idx = 3
        shape = list(getattr(param, "ds_shape", param.shape))
        layer_match = re.match(r"^(.*\.layers)\.(\d+)\.", name)
        if layer_match:
            prefix = layer_match.group(1)
            idx = int(layer_match.group(2))
            if idx == 0:
                _flush_skipped()
                skipped_layer_pattern = prefix
                lines.append(f"  {name} | {shape} | trainable={param.requires_grad}")
            elif skipped_layer_pattern == prefix:
                skipped_count += 1
                continue
            else:
                _flush_skipped()
                lines.append(f"  {name} | {shape} | trainable={param.requires_grad}")
        else:
            _flush_skipped()
            lines.append(f"  {name} | {shape} | trainable={param.requires_grad}")

    _flush_skipped()
    print("\n".join(lines))


def _configure_sdar_model(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> "PreTrainedModel":
    """Apply all SDAR-specific training configuration from finetuning_args to the model.

    This is the single place where finetuning_args controls SDAR model behaviour.
    Call this once after the model is loaded and set to train() mode.
    """
    input_prep_mode = getattr(finetuning_args, "input_prep_mode", "bd_packed")
    if hasattr(model, "set_input_prep_mode"):
        model.set_input_prep_mode(input_prep_mode)
        logger.info_rank0(f"Set training input prep mode to '{input_prep_mode}'.")
    else:
        logger.warning_once(
            f"Model has no set_input_prep_mode() method, input_prep_mode='{input_prep_mode}' ignored."
        )

    loss_type = getattr(finetuning_args, "loss_type", "ce")
    if hasattr(model, "set_loss_type"):
        model.set_loss_type(
            loss_type=loss_type,
            kd_temperature=getattr(finetuning_args, "kd_temperature", 1.0),
            kd_reverse_kl=getattr(finetuning_args, "kd_reverse_kl", False),
        )
        logger.info_rank0(f"Set MRP loss type to '{loss_type}'.")
    else:
        logger.warning_once(f"Model has no set_loss_type() method, loss_type='{loss_type}' ignored.")

    loss_anneal_schedule = getattr(finetuning_args, "loss_anneal_schedule", "none")
    loss_anneal_warmup_steps = getattr(finetuning_args, "loss_anneal_warmup_steps", 0)
    if loss_anneal_schedule != "none":
        if hasattr(model, "set_loss_anneal_beta"):
            model._loss_anneal_schedule = loss_anneal_schedule
            model._loss_anneal_warmup_steps = loss_anneal_warmup_steps
            model.set_loss_anneal_beta(0.0)  # start uniform
            logger.info_rank0(f"Loss annealing enabled: schedule='{loss_anneal_schedule}', warmup={loss_anneal_warmup_steps}")
        else:
            logger.warning_once(f"Model has no set_loss_anneal_beta() method, loss_anneal_schedule ignored.")

    if getattr(finetuning_args, "profile_mrp_metrics", False):
        if hasattr(model, "set_mrp_profile_metrics"):
            model.set_mrp_profile_metrics(True)
            logger.info_rank0("MRP profiling metrics enabled (logged under 'mrp/' prefix).")
        else:
            logger.warning_once("Model has no set_mrp_profile_metrics() method, profile_mrp_metrics ignored.")

    mrp_init_std = getattr(finetuning_args, "mrp_init_std", 0.02)
    if hasattr(model, "set_mrp_init_std"):
        model.set_mrp_init_std(mrp_init_std)
        logger.info_rank0(f"Set MRP head init std to {mrp_init_std!r}.")
    else:
        logger.warning_once(
            f"Model has no set_mrp_init_std() method, mrp_init_std={mrp_init_std!r} ignored."
        )

    if getattr(finetuning_args, "init_eh_proj", False):
        if hasattr(model, "init_eh_proj_structured"):
            model.init_eh_proj_structured()
            logger.info_rank0("Applied structured init to eh_proj (W_h=identity, W_e=zero).")
        else:
            logger.warning_once("Model has no init_eh_proj_structured() method, init_eh_proj ignored.")

    # v1.5 architecture: init MRP layers from target's last D layers.
    # Combined with mrp_input_layer_offset and mrp_drop_residual config flags,
    # makes MRP reproduce target's final hidden state at init.
    if getattr(finetuning_args, "mrp_init_from_target_layers", False):
        if hasattr(model, "init_mrp_from_target_layers"):
            model.init_mrp_from_target_layers()
            logger.info_rank0("Applied v1.5 init: MRP layers initialized from target's last D layers.")
        else:
            logger.warning_once(
                "Model has no init_mrp_from_target_layers() method, "
                "mrp_init_from_target_layers ignored."
            )

    reveal_mode = getattr(finetuning_args, "reveal_mode", "gt")
    if hasattr(model, "set_reveal_mode"):
        model.set_reveal_mode(reveal_mode)
        logger.info_rank0(f"Set reveal mode to '{reveal_mode}'.")
    else:
        logger.warning_once(
            f"Model has no set_reveal_mode() method, reveal_mode='{reveal_mode}' ignored."
        )

    gt_topk = getattr(finetuning_args, "gt_topk", 1)
    if hasattr(model, "set_gt_topk"):
        model.set_gt_topk(gt_topk)
        logger.info_rank0(f"Set gt_topk to {gt_topk}.")
    else:
        logger.warning_once(
            f"Model has no set_gt_topk() method, gt_topk={gt_topk} ignored."
        )

    gt_random_topk = getattr(finetuning_args, "gt_random_topk", "1")
    if hasattr(model, "set_gt_random_topk"):
        model.set_gt_random_topk(gt_random_topk)
        logger.info_rank0(f"Set gt_random_topk to {gt_random_topk!r}.")
    else:
        logger.warning_once(
            f"Model has no set_gt_random_topk() method, gt_random_topk={gt_random_topk!r} ignored."
        )

    loss_weight_mode = getattr(finetuning_args, "loss_weight_mode", "uniform")
    if hasattr(model, "set_loss_weight_mode"):
        model.set_loss_weight_mode(loss_weight_mode)
        logger.info_rank0(f"Set loss_weight_mode to '{loss_weight_mode}'.")
    else:
        logger.warning_once(
            f"Model has no set_loss_weight_mode() method, loss_weight_mode='{loss_weight_mode}' ignored."
        )

    mrp_steps = getattr(finetuning_args, "mrp_steps", 1)
    if hasattr(model, "set_mrp_steps"):
        model.set_mrp_steps(mrp_steps)
        logger.info_rank0(f"Set mrp_steps (training) to {mrp_steps}.")
    else:
        logger.warning_once(
            f"Model has no set_mrp_steps() method, mrp_steps={mrp_steps} ignored."
        )

    return model


def _freeze_sdar_modules(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    r"""Freeze SDAR backbone and/or lm_head for MRP training.

    freeze_backbone freezes the entire base SDARModel (embed_tokens, layers, norm, rotary_emb).
    Since mrp_module.model.embed_tokens is the same object as model.model.embed_tokens, this
    also covers the MRP module's embed_tokens.

    freeze_lm_head freezes only lm_head. Since mrp_module.lm_head is the same object as
    model.lm_head, this also covers the MRP module's lm_head.

    Must be called after model.train() so that eval() on frozen parts takes effect.
    """
    if getattr(finetuning_args, "freeze_backbone", False):
        backbone = getattr(model, "model", None)
        if backbone is not None:
            backbone.requires_grad_(False)
            backbone.eval()
        logger.info_rank0("Froze SDAR backbone (model.model).")

    if getattr(finetuning_args, "freeze_lm_head", False):
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            lm_head.requires_grad_(False)
        logger.info_rank0("Froze SDAR lm_head.")


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                    model = model.thinker  # use part of Omni model

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()
        if getattr(model.config, "model_type", None) == "sdar" and is_trainable:
            _freeze_sdar_modules(model, finetuning_args)

    if is_trainable and getattr(model.config, "model_type", None) == "sdar":
        model = _configure_sdar_model(model, finetuning_args)

    if is_trainable:
        log_trainable_modules(model)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model

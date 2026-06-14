# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _compute_anneal_beta(schedule: str, step: int, total_steps: int, warmup: int) -> float:
    """Compute loss annealing beta exponent from schedule and current step."""
    if total_steps <= 0:
        return 1.0
    if schedule == "linear":
        return step / total_steps
    elif schedule == "cosine":
        import math
        return 0.5 * (1 - math.cos(math.pi * step / total_steps))
    elif schedule == "step":
        return 0.0 if step < warmup else 1.0
    elif schedule == "delayed_linear":
        if step < warmup:
            return 0.0
        return (step - warmup) / max(total_steps - warmup, 1)
    return 1.0


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        # Update loss annealing beta before the forward pass
        unwrapped = self.accelerator.unwrap_model(model)
        schedule = getattr(unwrapped, "_loss_anneal_schedule", "none")
        if schedule != "none":
            beta = _compute_anneal_beta(
                schedule,
                self.state.global_step,
                self.state.max_steps,
                getattr(unwrapped, "_loss_anneal_warmup_steps", 0),
            )
            unwrapped.set_loss_anneal_beta(beta)

        # Gate MRP profiling so the (expensive) extra denoised forward + metric
        # compute only runs on steps that will actually be logged. Without this,
        # `_profile_mrp_metrics` stays True every step and roughly doubles the
        # target forward cost for a value that is only read every `logging_steps`.
        # Source of truth: finetuning_args. Reading from the (possibly DeepSpeed-
        # wrapped) model is fragile — for MoE+ZeRO, `unwrap_model` did not surface
        # the attribute, which left this False and caused an UnboundLocalError below.
        if getattr(self, "_mrp_profile_requested", None) is None:
            self._mrp_profile_requested = bool(getattr(self.finetuning_args, "profile_mrp_metrics", True))
        should_profile = False
        if self._mrp_profile_requested:
            logging_steps = max(int(getattr(self.args, "logging_steps", 1) or 1), 1)
            # global_step is incremented *after* training_step on the final
            # micro-batch of a grad-accum group, so (global_step + 1) % N == 0
            # marks the step whose metrics will be merged into the next log().
            should_profile = ((self.state.global_step + 1) % logging_steps == 0)
            unwrapped._profile_mrp_metrics = should_profile

        loss = super().training_step(model, inputs, *args, **kwargs)
        # self._log_first_gradient_stats(model)

        # Compute per-head grad norms at the gradient-accumulation boundary so
        # they reflect the fully-accumulated gradient, not a partial micro-batch.
        # DeepSpeed engines expose is_gradient_accumulation_boundary(); fall back
        # to True (compute every step) for non-DeepSpeed setups.
        is_boundary = getattr(model, "is_gradient_accumulation_boundary",
                              lambda: True)()
        if should_profile and is_boundary:
            mrp_modules = getattr(unwrapped, "mrp_modules", None)
            if mrp_modules is not None:
                grad_metrics: dict = {}
                for _h, _head in enumerate(mrp_modules):
                    _grads = [p.grad.detach().float()
                              for p in _head.parameters()
                              if p.grad is not None]
                    if _grads:
                        _norm = torch.sqrt(
                            sum(g.norm() ** 2 for g in _grads)
                        ).item()
                        grad_metrics[f"mrp/grad_norm_head_{_h}"] = _norm
                if grad_metrics:
                    existing = getattr(unwrapped, "_mrp_step_metrics", {}) or {}
                    existing.update(grad_metrics)
                    unwrapped._mrp_step_metrics = existing

        # Stash MRP metrics so they get merged into the next normal log() call.
        if self.is_world_process_zero():
            if not hasattr(self, '_unwrapped_model'):
                self._unwrapped_model = unwrapped
            metrics = getattr(unwrapped, "_mrp_step_metrics", None)
            if metrics:
                self._pending_mrp_metrics = metrics
        return loss

    @override
    def log(self, logs: dict, *args, **kwargs) -> None:
        """Merge stashed MRP profiling metrics into every normal trainer log call."""
        pending = getattr(self, "_pending_mrp_metrics", None)
        if pending:
            self._pending_mrp_metrics = {}

            # Separate per-head keys (mrp/*_head_N) from aggregate keys.
            per_head = {k: v for k, v in pending.items() if "_head_" in k}
            aggregate = {k: v for k, v in pending.items() if "_head_" not in k}

            # Merge aggregate MRP metrics into the main log dict (goes to W&B + console).
            logs = {**logs, **aggregate}

            # Send per-head metrics directly to W&B (skip the noisy console dict).
            if per_head:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(per_head, step=self.state.global_step)
                except Exception:
                    pass

            # Print one clean dict per head to the log file.
            head_ids = sorted({int(k.split("_head_")[-1]) for k in per_head})
            for h in head_ids:
                suffix = f"_head_{h}"
                head_dict = {"head": h}
                loss_key = f"mrp/loss{suffix}"
                gnorm_key = f"mrp/grad_norm{suffix}"
                if loss_key in per_head:
                    head_dict["loss"] = round(per_head[loss_key], 6)
                if gnorm_key in per_head:
                    head_dict["grad_norm"] = round(per_head[gnorm_key], 6)
                for k, v in per_head.items():
                    if k.endswith(suffix) and k not in (loss_key, gnorm_key):
                        short = k[: -len(suffix)].removeprefix("mrp/")
                        head_dict[short] = round(v, 4) if isinstance(v, float) else v
                print(f"\n{head_dict}")

            # Replace the trainer's (grad-accum-inflated) loss/grad_norm with
            # clean per-head averages derived from the model's own metrics.
            _head_losses = sorted(
                [(k, v) for k, v in per_head.items() if k.startswith("mrp/loss_head_")],
                key=lambda kv: kv[0],
            )
            if _head_losses:
                logs["avg_loss"] = round(
                    sum(v for _, v in _head_losses) / len(_head_losses), 6
                )
                logs.pop("loss", None)   # discard trainer's inflated value
            elif "loss" in logs:
                logs["avg_loss"] = logs.pop("loss")

            _head_gnorms = sorted(
                [(k, v) for k, v in per_head.items()
                 if k.startswith("mrp/grad_norm_head_")],
                key=lambda kv: kv[0],
            )
            if _head_gnorms:
                logs["avg_grad_norm"] = round(
                    sum(v for _, v in _head_gnorms) / len(_head_gnorms), 6
                )
                logs.pop("grad_norm", None)
            elif "grad_norm" in logs:
                logs["avg_grad_norm"] = logs.pop("grad_norm")

            # Reorder: learning_rate, avg_loss, avg_grad_norm first for easy scanning.
            _front = {}
            for _k in ("learning_rate", "avg_loss", "avg_grad_norm"):
                if _k in logs:
                    _front[_k] = logs.pop(_k)
            logs = {**_front, **logs}

        super().log(logs, *args, **kwargs)

    def _log_first_gradient_stats(self, model: "torch.nn.Module") -> None:
        if getattr(self, "_logged_first_gradient_stats", False):
            return

        if self.state.global_step != 0 or not self.is_world_process_zero():
            return

        logged = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach().float()
            logger.info_rank0(
                "[first_backward_grad] %s shape=%s norm=%.6e mean=%.6e absmax=%.6e",
                name,
                tuple(param.shape),
                grad.norm().item(),
                grad.mean().item(),
                grad.abs().max().item(),
            )
            logged += 1
            if logged >= 8:
                break

        if logged == 0:
            logger.info_rank0("[first_backward_grad] No gradients found to log on rank 0.")

        self._logged_first_gradient_stats = True

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

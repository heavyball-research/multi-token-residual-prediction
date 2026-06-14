# -*- coding: utf-8 -*-
"""
Loss function abstractions for MRP training.

Supports:
- "ce": Cross-entropy loss (default) — train MRP to predict ground-truth labels
- "ce_sum": CE on lm_head(h_A + h_C) → labels — no B needed, natural pair for logits_sum inference
- "kd_diff": KD between MRP logits (C) and denoised-target logits (B)
- "kd_diff_sum": KD between summed logits (A + C) and denoised-target logits (B)
- "mse": MSE between MRP logits and target logits (mean over vocab dim per token)
- "mse_sum": MSE between logits(A+C) and logits(B), i.e. make A+C ≈ B in logit space
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_linear_diffusion_cross_entropy import FusedLinearDiffusionCrossEntropyLoss


class MRPLossFunction(ABC):
    """Abstract base class for MRP loss functions.

    All loss functions take:
        hidden_states: [N, H] — MRP output hidden states (flattened)
        labels:        [N]    — ground-truth token ids
        lm_head:       nn.Linear — shared lm_head
        p_mask:        [N]    — diffusion probability mask
        answer_len:    int    — number of non-prompt tokens (for normalization)
        target_hidden_states: [N, H] — target model hidden states (optional, for KD)
    """

    # Whether this loss needs a second target forward on denoised input.
    needs_denoised_target: bool = False
    # Whether this loss trains A+C ≈ B (sum losses) vs C ≈ B or C ≈ labels.
    # Controls which profile metrics are logged: bac-based vs bc-based.
    is_sum_loss: bool = False

    @abstractmethod
    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute and return the scalar loss."""
        ...


class CELoss(MRPLossFunction):
    """Cross-entropy loss: train MRP to predict ground-truth labels."""

    _logged = False

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not CELoss._logged:
            print("[LossFunction] Using CELoss (cross-entropy)")
            CELoss._logged = True
        loss_fct = FusedLinearDiffusionCrossEntropyLoss(reduction='sum')
        loss = loss_fct(
            x=hidden_states,
            target=labels,
            weight=lm_head.weight,
            bias=lm_head.bias,
            p_mask=p_mask,
        )
        return loss / answer_len


class CESumLoss(MRPLossFunction):
    """CE loss on summed hidden states: CE(lm_head(h_A + h_C) → labels).

    Exploits linearity of lm_head: lm_head(h_A + h_C) = logits_A + logits_C.
    Trains A+C to predict ground-truth labels without needing a second target
    forward (no B required). Direct CE counterpart to kd_diff_sum/mse_sum, and
    the natural training signal for logits_sum inference mode.
    """

    is_sum_loss = True
    _logged = False

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not CESumLoss._logged:
            print("[LossFunction] Using CESumLoss (ce_sum: CE on lm_head(h_A + h_C) → labels)")
            CESumLoss._logged = True
        assert target_hidden_states is not None, (
            "CESumLoss requires target_hidden_states (first target forward)"
        )
        loss_fct = FusedLinearDiffusionCrossEntropyLoss(reduction='sum')
        loss = loss_fct(
            x=hidden_states + target_hidden_states,
            target=labels,
            weight=lm_head.weight,
            bias=lm_head.bias,
            p_mask=p_mask,
        )
        return loss / answer_len


def _normalize_per_token(
    per_token: torch.Tensor,
    labels: torch.LongTensor,
    p_mask: torch.Tensor,
    answer_len: int,
) -> torch.Tensor:
    """Apply ignore mask and p_mask weighting, then normalize by answer_len."""
    ignore_mask = (labels != -100)
    return (per_token / p_mask * ignore_mask).sum() / answer_len


def _compute_kd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.LongTensor,
    p_mask: torch.Tensor,
    answer_len: int,
    temperature: float,
    reverse_kl: bool = False,
) -> torch.Tensor:
    """Shared KL-divergence computation from pre-computed logits.

    Args:
        student_logits: [N, V] student model logits.
        teacher_logits: [N, V] teacher model logits (detached).
        labels: [N] ground-truth ids (used only for ignore mask via -100).
        p_mask: [N] diffusion probability mask.
        answer_len: number of non-prompt tokens (for normalization).
        temperature: softmax temperature.
        reverse_kl: If False (default), compute forward KL: KL(teacher || student).
            If True, compute reverse KL: KL(student || teacher).
    """
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)

    if reverse_kl:
        # Reverse KL: KL(student || teacher) — mode-seeking
        # KL(s || t) = sum(s * (log_s - log_t))
        student_probs = student_log_probs.exp()
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    else:
        # Forward KL: KL(teacher || student) — mean-seeking
        # KL(t || s) = sum(t * (log_t - log_s))
        teacher_probs = teacher_log_probs.exp()
        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

    kl_per_token = kl_per_token * (T * T)

    ignore_mask = (labels != -100)
    kl_per_token = kl_per_token / p_mask * ignore_mask
    return kl_per_token.sum() / answer_len


class KDDiffLoss(MRPLossFunction):
    """KD between MRP logits (C) and denoised-target logits (B).

    Requires a second target-model forward on the denoised input to produce B.
    Teacher: B = lm_head(denoised_target_hidden_states)  (no grad)
    Student: C = lm_head(hidden_states)                  (MRP output)
    """

    needs_denoised_target = True
    _logged = False

    def __init__(self, temperature: float = 1.0, reverse_kl: bool = False):
        self.temperature = temperature
        self.reverse_kl = reverse_kl

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not KDDiffLoss._logged:
            print("[LossFunction] Using KDDiffLoss (temperature=%.2f, reverse_kl=%s, needs_denoised_target=True)" % (self.temperature, self.reverse_kl))
            KDDiffLoss._logged = True
        assert denoised_target_hidden_states is not None, (
            "KDDiffLoss requires denoised_target_hidden_states "
            "(second target forward on denoised input)"
        )
        student_logits = lm_head(hidden_states)  # C
        with torch.no_grad():
            teacher_logits = lm_head(denoised_target_hidden_states)  # B
        return _compute_kd_from_logits(
            student_logits, teacher_logits, labels, p_mask, answer_len, self.temperature,
            reverse_kl=self.reverse_kl,
        )


class MSELoss(MRPLossFunction):
    """MSE loss between MRP logits and target logits.

    Computes per-token MSE averaged over the vocab dimension, then sums over
    valid tokens and normalizes by answer_len. Requires target_hidden_states.
    """

    _logged = False

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not MSELoss._logged:
            print("[LossFunction] Using MSELoss (mse, mean over vocab/logit dim per token)")
            MSELoss._logged = True
        assert target_hidden_states is not None, (
            "MSELoss requires target_hidden_states from the target model"
        )
        student_logits = lm_head(hidden_states).float()
        with torch.no_grad():
            teacher_logits = lm_head(target_hidden_states).float()
        mse_per_token = ((student_logits - teacher_logits) ** 2).mean(dim=-1)  # [N]
        return _normalize_per_token(mse_per_token, labels, p_mask, answer_len)


class MSESumLoss(MRPLossFunction):
    """MSE loss that makes logits(A + C) ≈ logits(B).

    Teacher: B = lm_head(denoised_target_hidden_states)                (no grad)
    Student: A + C = lm_head(target_hidden_states + hidden_states)

    Requires a second target-model forward on the denoised input to produce B.
    """

    needs_denoised_target = True
    is_sum_loss = True
    _logged = False

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not MSESumLoss._logged:
            print("[LossFunction] Using MSESumLoss (mse_sum, logits(A+C) vs logits(B), needs_denoised_target=True)")
            MSESumLoss._logged = True
        assert target_hidden_states is not None, (
            "MSESumLoss requires target_hidden_states (first target forward)"
        )
        assert denoised_target_hidden_states is not None, (
            "MSESumLoss requires denoised_target_hidden_states "
            "(second target forward on denoised input)"
        )
        student_logits = lm_head(target_hidden_states + hidden_states).float()  # logits(A + C)
        with torch.no_grad():
            teacher_logits = lm_head(denoised_target_hidden_states).float()  # logits(B)
        mse_per_token = ((student_logits - teacher_logits) ** 2).mean(dim=-1)  # [N]
        return _normalize_per_token(mse_per_token, labels, p_mask, answer_len)


class KDDiffSumLoss(MRPLossFunction):
    """KD between summed logits (A + C) and denoised-target logits (B).

    Teacher: B = lm_head(denoised_target_hidden_states)                (no grad)
    Student: A + C = lm_head(target_hidden_states) + lm_head(hidden_states)

    This directly sums without any weighting: student = lm_head(h_A + h_C).
    """

    needs_denoised_target = True
    is_sum_loss = True
    _logged = False

    def __init__(self, temperature: float = 1.0, reverse_kl: bool = False):
        self.temperature = temperature
        self.reverse_kl = reverse_kl

    def compute(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        lm_head: nn.Linear,
        p_mask: torch.Tensor,
        answer_len: int,
        target_hidden_states: Optional[torch.Tensor] = None,
        denoised_target_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not KDDiffSumLoss._logged:
            print("[LossFunction] Using KDDiffSumLoss (temperature=%.2f, reverse_kl=%s, needs_denoised_target=True)" % (self.temperature, self.reverse_kl))
            KDDiffSumLoss._logged = True
        assert target_hidden_states is not None, (
            "KDDiffSumLoss requires target_hidden_states (first target forward)"
        )
        assert denoised_target_hidden_states is not None, (
            "KDDiffSumLoss requires denoised_target_hidden_states "
            "(second target forward on denoised input)"
        )
        # Student: A + C  (via hidden-state addition, exploiting linearity of lm_head)
        student_logits = lm_head(target_hidden_states + hidden_states)

        with torch.no_grad():
            teacher_logits = lm_head(denoised_target_hidden_states)  # B
        return _compute_kd_from_logits(
            student_logits, teacher_logits, labels, p_mask, answer_len, self.temperature,
            reverse_kl=self.reverse_kl,
        )


def build_loss_function(
    loss_type: str = "ce",
    kd_temperature: float = 1.0,
    kd_reverse_kl: bool = False,
) -> MRPLossFunction:
    """Factory function to create a loss function by name.

    Args:
        loss_type: One of "ce", "ce_sum", "kd_diff", "kd_diff_sum", "mse", "mse_sum".
        kd_temperature: Temperature for KD softmax (used by KD-based losses).
        kd_reverse_kl: If True, use reverse KL (mode-seeking) instead of forward KL
            (mean-seeking) for all KD-based losses.
    """
    print(
        "[LossFunction] build_loss_function called with loss_type=%r, "
        "kd_temperature=%.2f, kd_reverse_kl=%s"
        % (loss_type, kd_temperature, kd_reverse_kl)
    )
    if loss_type == "ce":
        return CELoss()
    elif loss_type == "ce_sum":
        return CESumLoss()
    elif loss_type == "kd_diff":
        return KDDiffLoss(temperature=kd_temperature, reverse_kl=kd_reverse_kl)
    elif loss_type == "kd_diff_sum":
        return KDDiffSumLoss(temperature=kd_temperature, reverse_kl=kd_reverse_kl)
    elif loss_type == "mse":
        return MSELoss()
    elif loss_type == "mse_sum":
        return MSESumLoss()
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Supported: 'ce', 'ce_sum', 'kd_diff', 'kd_diff_sum', 'mse', 'mse_sum'"
        )


@torch.no_grad()
def compute_mrp_profile_metrics(
    hidden_states: torch.Tensor,
    labels: torch.LongTensor,
    lm_head: nn.Linear,
    p_mask: torch.Tensor,
    answer_len: int,
    target_hidden_states: Optional[torch.Tensor] = None,
    denoised_target_hidden_states: Optional[torch.Tensor] = None,
    is_sum_loss: bool = False,
) -> Dict[str, float]:
    """Compute diagnostic metrics for MRP training monitoring.

    Metrics logged under the "mrp/" prefix (always):
      c_hidden_norm   — mean ||h_C|| per token (C magnitude in hidden space)
      a_hidden_norm   — mean ||h_A|| per token (target model baseline)
      c_a_ratio       — c_hidden_norm / a_hidden_norm (relative scale of correction)
      c_logit_norm    — mean ||lm_head(h_C)|| per token (C magnitude in logit space)
      a_logit_norm    — mean ||lm_head(h_A)|| per token
      n_tokens        — number of supervised tokens in this batch

    Metrics logged when B is available — depend on ``is_sum_loss``:
      kl_ba           — mean KL(B||A) (training signal strength; always logged)
      ba_hidden_norm  — ||h_B - h_A|| (signal magnitude; always logged)
      ba_logit_norm   — ||logits_B - logits_A|| (signal magnitude; always logged)

      For sum losses (is_sum_loss=True, e.g. ce_sum, kd_diff_sum, mse_sum):
        ce_ac           — CE(A+C → labels): should decrease as A+C ≈ B is learned
        kl_bac          — KL(B || A+C): residual after applying C; should shrink
        bac_hidden_norm — ||h_B - (h_A + h_C)||: distance in hidden space
        bac_logit_norm  — ||logits_B - (logits_A + logits_C)||: distance in logit space

      For non-sum losses (is_sum_loss=False, e.g. ce, kd_diff, mse):
        ce_a            — CE(A → labels): target model baseline
        ce_c            — CE(C → labels): should decrease as C learns to predict
        kl_bc           — KL(B || C): how well C approximates B
        bc_hidden_norm  — ||h_B - h_C||: distance in hidden space
        bc_logit_norm   — ||logits_B - logits_C||: distance in logit space

    Args:
        hidden_states:               h_C [N, H] — MRP head output hidden states.
        labels:                      [N]  — ground-truth token ids (-100 = ignore).
        lm_head:                     shared linear head.
        p_mask:                      [N]  — diffusion masking probability per token.
        answer_len:                  number of non-prompt tokens (for normalisation ref).
        target_hidden_states:        h_A [N, H] — target model hidden states (noisy pass).
        denoised_target_hidden_states: h_B [N, H] — target hidden states (denoised pass).
        is_sum_loss:                 if True, log A+C-vs-B metrics; otherwise log C-vs-B metrics.
    """
    metrics: Dict[str, float] = {}
    h_C = hidden_states.float()
    valid = (labels != -100)
    n_tokens = int(valid.sum().item())
    metrics["mrp/n_tokens"] = float(n_tokens)

    if n_tokens == 0:
        return metrics

    # ---- C magnitude ----
    c_norms = h_C[valid].norm(dim=-1)
    metrics["mrp/c_hidden_norm"] = float(c_norms.mean().item())

    # ---- A magnitude + ratio ----
    if target_hidden_states is not None:
        h_A = target_hidden_states.float()
        a_norms = h_A[valid].norm(dim=-1)
        metrics["mrp/a_hidden_norm"] = float(a_norms.mean().item())
        metrics["mrp/c_a_ratio"] = float(
            (c_norms / (a_norms + 1e-8)).mean().item()
        )

        # lm_head weight is in model dtype (e.g. bfloat16); cast inputs to match,
        # then upcast outputs to float32 for numerically stable CE/KL.
        w_dtype = lm_head.weight.dtype
        logits_A = lm_head(h_A[valid].to(w_dtype)).float()
        logits_C = lm_head(h_C[valid].to(w_dtype)).float()
        metrics["mrp/a_logit_norm"] = float(logits_A.norm(dim=-1).mean().item())
        metrics["mrp/c_logit_norm"] = float(logits_C.norm(dim=-1).mean().item())

        labels_valid = labels[valid]

        if is_sum_loss:
            # Sum-loss metrics: the prediction is A+C, so measure A+C vs B.
            metrics["mrp/ce_ac"] = float(
                F.cross_entropy(logits_A + logits_C, labels_valid, ignore_index=-100).item()
            )
        else:
            # Non-sum-loss metrics: the prediction is C alone, so measure C vs labels/B.
            metrics["mrp/ce_a"] = float(
                F.cross_entropy(logits_A, labels_valid, ignore_index=-100).item()
            )
            metrics["mrp/ce_c"] = float(
                F.cross_entropy(logits_C, labels_valid, ignore_index=-100).item()
            )

        # ---- B-dependent metrics ----
        if denoised_target_hidden_states is not None:
            h_B = denoised_target_hidden_states.float()
            logits_B = lm_head(h_B[valid].to(w_dtype)).float()
            prob_B = F.softmax(logits_B, dim=-1)

            # KL(B||A) — always logged as signal-strength baseline.
            logp_A = F.log_softmax(logits_A, dim=-1)
            kl_ba = F.kl_div(logp_A, prob_B, reduction="none").sum(dim=-1)
            metrics["mrp/kl_ba"] = float(kl_ba.mean().item())

            # ||B - A|| — always logged as signal-magnitude baseline.
            metrics["mrp/ba_hidden_norm"] = float(
                (h_B[valid] - h_A[valid]).norm(dim=-1).mean().item()
            )
            metrics["mrp/ba_logit_norm"] = float(
                (logits_B - logits_A).norm(dim=-1).mean().item()
            )

            if is_sum_loss:
                # How well does A+C approximate B?
                logp_AC = F.log_softmax(logits_A + logits_C, dim=-1)
                kl_bac = F.kl_div(logp_AC, prob_B, reduction="none").sum(dim=-1)
                metrics["mrp/kl_bac"] = float(kl_bac.mean().item())
                metrics["mrp/bac_hidden_norm"] = float(
                    (h_B[valid] - (h_A[valid] + h_C[valid])).norm(dim=-1).mean().item()
                )
                metrics["mrp/bac_logit_norm"] = float(
                    (logits_B - (logits_A + logits_C)).norm(dim=-1).mean().item()
                )
            else:
                # How well does C approximate B?
                logp_C = F.log_softmax(logits_C, dim=-1)
                kl_bc = F.kl_div(logp_C, prob_B, reduction="none").sum(dim=-1)
                metrics["mrp/kl_bc"] = float(kl_bc.mean().item())
                metrics["mrp/bc_hidden_norm"] = float(
                    (h_B[valid] - h_C[valid]).norm(dim=-1).mean().item()
                )
                metrics["mrp/bc_logit_norm"] = float(
                    (logits_B - logits_C).norm(dim=-1).mean().item()
                )

    return metrics

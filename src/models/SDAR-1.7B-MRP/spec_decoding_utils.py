"""Utilities for vanilla within-block speculative-decoding of SDAR-MRP.

Kept in a separate module so the verification attention-mask builder can be
imported and unit-tested without pulling in transformers / einops.
"""

from __future__ import annotations

import torch


def build_spec_verification_mask(prefix_len, num_candidates, block_length, device):
    """Block-diagonal attention mask for within-block spec-decoding verify.

    The verification target forward sees, past the clean prefix (whose KV is
    in ``past_key_values``), a packed sequence of ``num_candidates`` copies
    of the same block at progressively-more-revealed denoising states::

        [ C(0) | C(1) | C(2) | ... | C(K) ]

    where ``C(k)`` is the block with ``k`` MRP-drafted tokens revealed
    (others still masked). Each ``C(k)`` shares the same ``position_ids``
    (the absolute positions of the anchor block under denoising).

    Connectivity:
      * Every new token attends to the full clean prefix.
      * Within each candidate, attention is bidirectional (self block).
      * Candidates do NOT attend to each other — they are different
        denoising states of the same block, not a causal chain.

    Returns:
        bool tensor of shape ``[1, (K+1)*BL, prefix_len + (K+1)*BL]``;
        True = may attend. (Caller passes ``num_candidates = K+1``.)
    """
    K1 = int(num_candidates)  # K+1 candidates total
    BL = int(block_length)
    L_new = K1 * BL
    L_total = int(prefix_len) + L_new
    mask = torch.zeros((1, L_new, L_total), dtype=torch.bool, device=device)
    if prefix_len > 0:
        mask[:, :, :prefix_len] = True
    # Block-diagonal self-attention per candidate.
    for k in range(K1):
        row_lo = k * BL
        row_hi = (k + 1) * BL
        col_lo = prefix_len + row_lo
        col_hi = prefix_len + row_hi
        mask[:, row_lo:row_hi, col_lo:col_hi] = True
    return mask


def residual_remask_transfer(transfer_index, committed_ids, mrp_residual_logits, delta):
    """Veto target commits whose MRP residual on the committed token is large.

    Implements the ``residual_remask`` inference mode's per-token gate: the
    target proposes commits (``transfer_index``); MRP supplies its raw
    logit-space residual ``C`` (its prediction, i.e. the term that would be
    added in ``A + C``). For each proposed commit at position ``p`` with token
    ``t = committed_ids[p]``, we read ``|C[p, t]|`` — how much MRP would move
    the chosen token's logit. A large value means the prediction is unstable
    (it would shift a lot once more context is revealed), so we remask it:
    drop it from ``transfer_index`` and let it resolve in a later step.

    Args:
        transfer_index: ``[B, L]`` bool — positions the target proposes to
            commit this step. ``None`` / all-False is returned unchanged.
        committed_ids: ``[B, L]`` long — token id the target would commit at
            each position (its sampled / argmax token).
        mrp_residual_logits: ``[B, L, V]`` float — MRP head's raw residual
            ``C`` (logit space), NOT mixed with the target logits.
        delta: float threshold on ``|C[p, committed_id]|``.

    Returns:
        ``[B, L]`` bool — the filtered commit mask. **Progress guard:** any
        sample that proposed at least one commit keeps at least one — the
        smallest-``|residual|`` proposed position — even if every proposal
        exceeds ``delta``, so a denoising step never stalls.
    """
    if transfer_index is None or not bool(transfer_index.any()):
        return transfer_index

    # |C| at the committed token, per position.
    c_tok = mrp_residual_logits.gather(
        -1, committed_ids.long().unsqueeze(-1)
    ).squeeze(-1)                                          # [B, L]
    mag = c_tok.abs()

    survive = transfer_index & (mag <= float(delta))

    # Progress guard: per-sample, if a sample proposed commits but all were
    # vetoed, keep its single most-stable (smallest |C|) proposed position.
    had_any = transfer_index.any(dim=1)                    # [B]
    none_left = had_any & (~survive.any(dim=1))            # [B]
    if bool(none_left.any()):
        big = torch.where(
            transfer_index, mag, torch.full_like(mag, float('inf'))
        )
        keep_pos = big.argmin(dim=1)                       # [B]
        rows = none_left.nonzero(as_tuple=True)[0]
        survive[rows, keep_pos[rows]] = True

    return survive

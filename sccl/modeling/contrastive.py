from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ContrastiveSamplingConfig:
    """Sampling hyperparameters used by the SCCL contrastive objectives."""

    top_k: int
    end_k: int
    temperature: float


def select_pos_neg_logits_from_queue(
    liner_q: torch.Tensor,         # [bs, D], L2-normalized query projections
    label_q: torch.Tensor,         # [bs]
    feature_queue: torch.Tensor,   # [K, D], L2-normalized queue features
    label_queue: torch.Tensor,     # [K], -1 denotes an uninitialized entry
    cfg: ContrastiveSamplingConfig,
) -> Optional[torch.Tensor]:
    """
    Select positive and negative similarities from a MoCo queue.

    For each query, queue entries with the same diagnosis are positives and
    entries with a different diagnosis are negatives. Following the original
    SCCL implementation, the selected positives include both the most similar
    positives and the least similar (hard) positives, whereas negatives are
    ordered from most to least similar. The first column of the returned tensor
    is the positive logit and all remaining columns are negative logits.

    Uninitialized queue entries must have label ``-1`` and are ignored. During
    early training or toy runs, the numbers of available positives and
    negatives may be smaller than ``top_k`` or ``end_k``; the requested counts
    are therefore truncated dynamically without duplicating positive entries.

    Returns:
        Tensor of shape ``[bs * n_selected_positive, 1 + n_negative]``, or
        ``None`` when no valid contrastive pair can be constructed.
    """
    if liner_q.ndim != 2 or feature_queue.ndim != 2:
        raise ValueError("liner_q and feature_queue must both be 2D tensors.")
    if label_q.ndim != 1 or label_queue.ndim != 1:
        raise ValueError("label_q and label_queue must both be 1D tensors.")
    if liner_q.shape[0] != label_q.shape[0]:
        raise ValueError("The query feature and query label batch sizes differ.")
    if feature_queue.shape[0] != label_queue.shape[0]:
        raise ValueError("The queue feature and queue label sizes differ.")
    if liner_q.shape[1] != feature_queue.shape[1]:
        raise ValueError("Query and queue feature dimensions differ.")
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if cfg.top_k < 0 or cfg.end_k < 0 or cfg.top_k + cfg.end_k == 0:
        raise ValueError("top_k and end_k must be non-negative and not both zero.")

    # Ignore queue slots that have not yet been populated. This remains safe
    # when MoCoQueue.get() already returns only valid entries.
    valid_queue = label_queue >= 0
    if not bool(valid_queue.any()):
        return None

    feature_queue = feature_queue[valid_queue]
    label_queue = label_queue[valid_queue].long()
    label_q = label_q.view(-1).long()

    bs = int(label_q.shape[0])
    queue_size = int(label_queue.shape[0])
    if bs == 0 or queue_size == 0:
        return None

    # Cosine similarities reduce to dot products because caller-provided
    # features are L2-normalized.
    cos_sim = torch.matmul(liner_q, feature_queue.transpose(0, 1))  # [bs, K]

    queue_labels = label_queue.unsqueeze(0).expand(bs, queue_size)
    query_labels = label_q.unsqueeze(1).expand(bs, queue_size)
    pos_mask = queue_labels.eq(query_labels)
    neg_mask = ~pos_mask

    pos_count = pos_mask.sum(dim=1)
    neg_count = neg_mask.sum(dim=1)

    # A rectangular logits tensor requires a common number of valid entries
    # across the batch. Use the minimum available count, as in the original
    # implementation, while handling sparse early queues safely.
    min_pos = int(pos_count.min().item())
    min_neg = int(neg_count.min().item())
    if min_pos <= 0 or min_neg <= 0:
        return None

    neg_inf = torch.tensor(float("-inf"), device=cos_sim.device, dtype=cos_sim.dtype)
    pos_scores = cos_sim.masked_fill(~pos_mask, neg_inf)
    neg_scores = cos_sim.masked_fill(~neg_mask, neg_inf)

    # Descending order: the first positives are most similar and the last are
    # least similar (hard positives). Allocate non-overlapping selections.
    pos_sorted = pos_scores.topk(min_pos, dim=1, largest=True, sorted=True).values
    n_end = min(int(cfg.end_k), min_pos)
    n_top = min(int(cfg.top_k), min_pos - n_end)

    selected_parts = []
    if n_top > 0:
        selected_parts.append(pos_sorted[:, :n_top])
    if n_end > 0:
        selected_parts.append(pos_sorted[:, -n_end:])
    if not selected_parts:
        return None

    selected_pos = torch.cat(selected_parts, dim=1)  # [bs, n_selected]
    n_selected = int(selected_pos.shape[1])
    selected_pos = selected_pos.reshape(-1, 1)

    # Retain the most similar negatives first. Using all common valid negatives
    # preserves the original loss construction while avoiding invalid entries.
    neg_sorted = neg_scores.topk(min_neg, dim=1, largest=True, sorted=True).values
    neg_repeated = (
        neg_sorted.unsqueeze(1)
        .expand(bs, n_selected, min_neg)
        .reshape(bs * n_selected, min_neg)
    )

    logits = torch.cat([selected_pos, neg_repeated], dim=1)
    return logits / float(cfg.temperature)

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ContrastiveSamplingConfig:
    """
    Sampling hyper-parameters matching the original SCCL implementation.
    """
    top_k: int
    end_k: int
    temperature: float


def _masked_scatter_by_bool_mask(
    base: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Helper to scatter a 1D 'values' tensor into 'base' at mask==True positions.

    This mirrors the original 'masked_select + masked_scatter' pattern,
    but avoids hard-coded CUDA calls.
    """
    out = base.clone()
    out = out.masked_scatter(mask, values)
    return out


def select_pos_neg_logits_from_queue(
    liner_q: torch.Tensor,         # [bs, D] normalized query projection
    label_q: torch.Tensor,         # [bs]
    feature_queue: torch.Tensor,   # [K, D] normalized
    label_queue: torch.Tensor,     # [K]
    cfg: ContrastiveSamplingConfig,
) -> torch.Tensor:
    """
    Ported from your original:
      - Expand queue to [bs, K, D]
      - Compute cosine similarity via einsum
      - Build pos/neg masks by label equality
      - Take top-k and end-k positives; replicate negatives; concatenate and temperature-scale

    Returns:
      logits_con: [bs*(top_k+end_k), 1+neg_min] in the original shape semantics
                 (first column is positive logit, remaining columns negatives)
    """
    device = liner_q.device
    bs = int(label_q.shape[0])
    K = int(label_queue.shape[0])

    # 1) Expand queue to match batch
    tmp_label_queue = label_queue.view(1, K).repeat(bs, 1)           # [bs, K]
    tmp_feature_queue = feature_queue.view(1, K, -1).repeat(bs, 1, 1) # [bs, K, D]

    # 2) Cosine similarity (assuming features already normalized)
    cos_sim = torch.einsum("nc,nkc->nk", liner_q, tmp_feature_queue) # [bs, K]

    # 3) Positive/negative masks
    tmp_label = label_q.view(bs, 1).repeat(1, K)                     # [bs, K]
    pos_mask = (tmp_label_queue == tmp_label)                        # [bs, K]
    neg_mask = ~pos_mask                                             # [bs, K]

    # 4) Fill pos/neg matrices with -inf except masked positions
    neg_inf = torch.tensor(-np.inf, device=device, dtype=cos_sim.dtype)
    pos_mat = torch.full_like(cos_sim, neg_inf)
    neg_mat = torch.full_like(cos_sim, neg_inf)

    pos_vals = cos_sim.masked_select(pos_mask)
    neg_vals = cos_sim.masked_select(neg_mask)

    pos_mat = _masked_scatter_by_bool_mask(pos_mat, pos_mask, pos_vals)
    neg_mat = _masked_scatter_by_bool_mask(neg_mat, neg_mask, neg_vals)

    # 5) Handle variable number of positives/negatives per sample:
    #    mimic your 'pos_min/neg_min' strategy
    pos_num = pos_mask.int().sum(dim=-1)  # [bs]
    neg_num = neg_mask.int().sum(dim=-1)  # [bs]

    pos_min = int(pos_num.min().item())
    neg_min = int(neg_num.min().item())

    if pos_min <= 0 or neg_min <= 0:
        # No valid contrastive pairs; caller should handle this.
        return None  # type: ignore

    # Keep only valid entries: topk(pos_min) gives the valid positive similarities for each sample
    pos_sorted, _ = pos_mat.topk(pos_min, dim=-1)  # [bs, pos_min]
    pos_top = pos_sorted[:, : cfg.top_k]           # [bs, top_k]
    pos_end = pos_sorted[:, -cfg.end_k :]          # [bs, end_k]
    pos_sel = torch.cat([pos_top, pos_end], dim=-1).contiguous()  # [bs, top_k+end_k]
    pos_sel = pos_sel.view(-1, 1)                                   # [bs*(top_k+end_k), 1]

    neg_sorted, _ = neg_mat.topk(neg_min, dim=-1)  # [bs, neg_min]
    # Replicate negatives to match the flattened positives, exactly as your original code
    neg_rep = neg_sorted.repeat(1, cfg.top_k + cfg.end_k)           # [bs, neg_min*(top_k+end_k)]
    neg_rep = neg_rep.view(-1, neg_min).contiguous()                # [bs*(top_k+end_k), neg_min]

    logits = torch.cat([pos_sel, neg_rep], dim=-1) / float(cfg.temperature)
    return logits

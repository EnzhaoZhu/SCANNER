from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..utils.masking import apply_sample_mask_to_loss


class FocalLoss(nn.Module):
    """
    Standard multi-class focal loss.

    Args:
      gamma: focusing parameter
      alpha: optional class balancing factor (float or per-class tensor). If None, no alpha weighting.
      reduction: "mean" | "sum" | "none"
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor | float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)  # pt = softmax prob of the true class
        loss = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                loss = self.alpha * loss
            else:
                # alpha as per-class tensor
                a = self.alpha.to(device=logits.device, dtype=logits.dtype)
                loss = a.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Loss utilities for:
      - Diagnosis classification (CE or focal)
      - Contrastive InfoNCE (CE or focal with fixed target=0)
      - NPS token tagging CE with:
          * ignore_index
          * attention_mask for padding
          * has_nps_labels for partially labeled batches
    """

    def __init__(self, *, use_focal: bool, focal_gamma: float, focal_alpha: Optional[float], nps_ignore_index: int):
        super().__init__()
        self.use_focal = bool(use_focal)
        self.nps_ignore_index = int(nps_ignore_index)

        if self.use_focal:
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
            self.con_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        else:
            self.cls_loss = nn.CrossEntropyLoss()
            self.con_loss = nn.CrossEntropyLoss()

    def loss_diag(self, diag_logits: torch.Tensor, diag_labels: torch.Tensor) -> torch.Tensor:
        return self.cls_loss(diag_logits, diag_labels.long())

    def loss_infonce(self, logits: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE target is always 0 (positive logit at column 0).
        """
        targets = torch.zeros((logits.shape[0],), device=logits.device, dtype=torch.long)
        return self.con_loss(logits, targets)

    def loss_nps_token_ce(
        self,
        token_logits: torch.Tensor,              # [bs, L, C]
        nps_labels: torch.Tensor,                # [bs, L]
        attention_mask: Optional[torch.Tensor],  # [bs, L]
        has_nps_labels: torch.BoolTensor,        # [bs]
    ) -> torch.Tensor:
        bs, L, C = token_logits.shape
        if attention_mask is None:
            attention_mask = torch.ones((bs, L), device=token_logits.device, dtype=torch.long)

        logits_2d = token_logits.reshape(bs * L, C)
        labels_1d = nps_labels.reshape(bs * L).long()

        per_token = F.cross_entropy(
            logits_2d,
            labels_1d,
            ignore_index=self.nps_ignore_index,
            reduction="none",
        ).reshape(bs, L)

        per_token = per_token * attention_mask.float()
        denom = attention_mask.float().sum(dim=1).clamp_min(1.0)
        per_sample = per_token.sum(dim=1) / denom

        return apply_sample_mask_to_loss(per_sample, has_nps_labels)

    def loss_nps_crf_nll(
        self,
        nps_head,  # NPSHead
        token_logits: torch.Tensor,            # [bs, L, C]
        nps_labels: torch.Tensor,              # [bs, L]
        attention_mask: Optional[torch.Tensor],# [bs, L]
        has_nps_labels: torch.BoolTensor,      # [bs]
    ) -> torch.Tensor:
        """
        CRF negative log-likelihood for NPS, with:
          - attention_mask for padding
          - ignore_index masking for missing token labels
          - sample-level masking for partially labeled batches
        """
        bs, L, _ = token_logits.shape
        if attention_mask is None:
            attention_mask = torch.ones((bs, L), device=token_logits.device, dtype=torch.long)

        # Build a CRF-compatible mask:
        #  - padding tokens are masked out by attention_mask
        #  - ignore_index tokens are also masked out (CRF cannot consume -100 tags)
        attn = attention_mask.bool()
        valid_token = (nps_labels != self.nps_ignore_index)

        tags = nps_labels.clone()
        tags[~valid_token] = 0
        tags = tags.long()

        crf_mask = attn & valid_token

        # Enforce torchcrf constraint: mask[:,0] must be True whenever a sequence exists.
        # If attention_mask[:,0] is True but valid_token[:,0] is False, force mask True and use a dummy tag id.
        if crf_mask.size(1) > 0:
            need_fix = attn[:, 0] & (~crf_mask[:, 0])
            if need_fix.any():
                crf_mask[need_fix, 0] = True
                tags[need_fix, 0] = 0

        # Compute per-sample NLL with reduction="none", then apply sample-level mask.
        # torchcrf returns shape [bs] for reduction="none".
        per_sample_nll = nps_head.neg_log_likelihood(
            token_logits, tags, crf_mask, reduction="none"
        )  # [bs]

        return apply_sample_mask_to_loss(per_sample_nll, has_nps_labels)

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..utils.bio import bio_tags_to_spans


class DiagnosisHead(nn.Module):
    """
    Diagnosis classification head: pooled_output -> logits.

    This is intentionally simple and easily swappable.
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(pooled))


class ProjectionHead(nn.Module):
    """
    Contrastive projection head: pooled_output -> projected feature.

    You can replace this with your exact SCCL implementation if needed,
    but keep the signature stable.
    """

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NPSHead(nn.Module):
    """
    Token-level NPS head: sequence_output -> token logits (+ optional CRF decode).

    Design choices for a public repo:
      - By default, use plain token classification (CE + argmax decoding).
      - If use_crf=True and torchcrf is installed, CRF decoding is enabled.
      - If CRF is unavailable, it falls back to non-CRF behavior automatically.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float, use_crf: bool = False):
        super().__init__()
        self.num_labels = int(num_labels)
        self.use_crf = bool(use_crf)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self._crf = None
        if self.use_crf:
            try:
                from torchcrf import CRF  # type: ignore
                self._crf = CRF(num_tags=self.num_labels, batch_first=True)
            except Exception:
                self._crf = None
                self.use_crf = False

    def forward_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(sequence_output))

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute CRF negative log-likelihood for a batch.

        Args:
          emissions: [bs, L, C] raw logits (no softmax)
          tags:      [bs, L] tag ids
          mask:      [bs, L] boolean mask for valid positions
          reduction: "none" | "sum" | "mean" (torchcrf supports these)

        Returns:
          Scalar loss if reduction != "none", else [bs] loss vector.
        """
        if not (self.use_crf and self._crf is not None):
            raise RuntimeError("CRF is not enabled/available. Set use_crf=True and install torchcrf.")

        # torchcrf returns log-likelihood; we minimize negative log-likelihood.
        ll = self._crf(emissions, tags, mask=mask, reduction=reduction)
        return -ll

    @torch.no_grad()
    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        id2tag: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Decode token logits into tag sequences and spans.

        Args:
          logits: [bs, L, C]
          attention_mask: [bs, L] (1 for valid tokens)
          id2tag: optional mapping from label id to BIO tag string.

        Returns:
          {
            "nps_tag_ids": List[List[int]],
            "nps_tags": List[List[str]],
            "nps_spans": List[List[{"type":..., "start":..., "end":...}]]
          }
        """
        if attention_mask is None:
            attention_mask = torch.ones(logits.shape[:2], device=logits.device, dtype=torch.long)

        if self.use_crf and self._crf is not None:
            paths = self._crf.decode(logits, mask=attention_mask.bool())  # List[List[int]]
        else:
            paths = logits.argmax(dim=-1).tolist()

        out_tag_ids = []
        out_tags = []
        out_spans = []

        for i, path in enumerate(paths):
            L = int(attention_mask[i].sum().item())
            path = path[:L]
            out_tag_ids.append([int(x) for x in path])

            if id2tag is None:
                # If tag names are not provided, return numeric tags and empty spans.
                tags = [str(int(x)) for x in path]
                spans = []
            else:
                tags = [id2tag[int(x)] for x in path]
                spans = bio_tags_to_spans(tags)

            out_tags.append(tags)
            out_spans.append(spans)

        return {"nps_tag_ids": out_tag_ids, "nps_tags": out_tags, "nps_spans": out_spans}

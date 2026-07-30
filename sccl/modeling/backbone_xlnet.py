from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import XLNetModel
from transformers.modeling_utils import SequenceSummary


@dataclass
class BackboneOutput:
    """Standardized backbone outputs to keep the rest of the model backbone-agnostic."""
    sequence_output: torch.Tensor  # [bs, seq_len, hidden]
    pooled_output: torch.Tensor    # [bs, hidden]


class XLNetBackbone(nn.Module):
    """
    XLNet encoder wrapper.

    Outputs:
      - sequence_output: token-level representations for token tagging (NPS)
      - pooled_output: sentence-level representation for diagnosis + contrastive projection
    """

    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_name_or_path)
        self.summary = SequenceSummary(pretrained_model_name_or_path)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> BackboneOutput:
        out = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq = out.last_hidden_state
        pooled = self.summary(seq)
        return BackboneOutput(sequence_output=seq, pooled_output=pooled)

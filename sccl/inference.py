"""
Minimal inference demo for the public SCCL architecture repo.

This script intentionally uses random token ids (no private data),
to verify that:
  - the model can be instantiated
  - inference returns diagnosis probabilities
  - inference returns NPS decoded outputs when NPS head is enabled
"""

from __future__ import annotations

import torch

from sccl.config import SCCLConfig
from sccl.modeling.sccl import SCCLModel


def main() -> None:
    torch.set_grad_enabled(False)

    cfg = SCCLConfig(
        pretrained_model_name_or_path="../xlnet_base",
        num_diag_classes=5,
        num_nps_labels=9,  # e.g., small BIO tag set for demo
        use_crf=False,
        queue_size_his=128,
        queue_size_sc=128,
        top_k=2,
        end_k=1,
    )

    model = SCCLModel(cfg).eval()

    bs, L = 2, 32
    vocab_size = 32000  # dummy range

    batch = {
        "input_ids": torch.randint(0, vocab_size, (bs, L), dtype=torch.long),
        "token_type_ids": torch.zeros((bs, L), dtype=torch.long),
        "attention_mask": torch.ones((bs, L), dtype=torch.long),
        "labels": torch.randint(0, cfg.num_diag_classes, (bs,), dtype=torch.long),  # kept for SCCL interface
    }

    # Optional id2tag mapping for readable NPS tags.
    id2tag = {0: "O", 1: "B-X", 2: "I-X", 3: "B-Y", 4: "I-Y", 5: "B-Z", 6: "I-Z", 7: "B-W", 8: "I-W"}

    out = model(
        query=batch,
        positive_sample=None,
        positive_sc=None,
        id2tag=id2tag,
        return_dict=True,
    )

    print("diag_probs shape:", out["diag_probs"].shape)  # [bs, C]
    if "nps_tags" in out:
        print("nps_tags[0][:10]:", out["nps_tags"][0][:10])
        print("nps_spans[0]:", out["nps_spans"][0])


if __name__ == "__main__":
    main()

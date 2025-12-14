from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SCCLConfig:
    """
    Architecture-only configuration for SCCL (no dataset paths, no training loops).

    Key requirements covered:
      - Two-stage contrastive learning (SC queue + HIS queue)
      - Two heads (diagnosis + NPS tagging)
      - Partial-label masking for NPS
      - Inference outputs both diagnosis + NPS predictions

    Default backbone is a Chinese XLNet base checkpoint: hfl/chinese-xlnet-base.
    """

    # ===== Backbone =====
    pretrained_model_name_or_path: str = "xlnet_base"
    hidden_size: int = 768
    dropout: float = 0.1

    # ===== Heads =====
    num_diag_classes: int = 5
    num_nps_labels: int = 27
    use_crf: bool = True
    nps_ignore_index: int = -100  # compatible with HF token-classification ignore index

    # ===== Contrastive (MoCo-style) =====
    temperature: float = 0.07
    momentum: float = 0.999
    queue_size_sc: int = 768
    queue_size_his: int = 768

    # ===== Loss weights =====
    w_diag: float = 1.0
    w_nps: float = 1.0
    w_contrast_sc: float = 0.2
    w_contrast_his: float = 0.2

    # ===== Misc =====
    return_dict: bool = True

    # ===== Contrastive sampling (ported from your original SCCL) =====
    top_k: int = 8
    end_k: int = 2

    # ===== Queue update multiplicity (how many keys per query) =====
    positive_num: int = 1  # corresponds to config.positive_num in your original code
    sc_positive_num: int = 1  # corresponds to config.sc_positive_num in your original code

    # ===== Training mixture weights (match your original naming) =====
    contrastive_rate_in_training: float = 0.2
    sc_rate_in_training: float = 0.2

    # ===== Loss =====
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float | None = None


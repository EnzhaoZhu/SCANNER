from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SCCLConfig:
    """
    Configuration for the public SCCL reference implementation.

    The defaults describe the architecture used in the manuscript:
      - six diagnostic classes;
      - 27 NPS categories represented with BIO tags (55 token labels);
      - historical-sample and standard-case contrastive queues;
      - diagnosis classification and NPS sequence-labelling heads.

    Dataset paths and training-loop settings should be supplied by the calling
    script rather than embedded in this architecture configuration.
    """

    # ===== Backbone =====
    # The local project can point this to ``xlnet_base``. Public users may
    # override it with an equivalent Hugging Face checkpoint identifier.
    pretrained_model_name_or_path: str = "xlnet_base"
    hidden_size: int = 768
    dropout: float = 0.1
    max_length: int = 512

    # ===== Heads =====
    num_diag_classes: int = 6
    num_nps_labels: int = 27 * 2 + 1  # O + B/I tags for 27 NPS categories
    use_crf: bool = True
    nps_ignore_index: int = -100

    # ===== Contrastive learning (MoCo-style) =====
    temperature: float = 0.07
    momentum: float = 0.999
    queue_size_sc: int = 768
    queue_size_his: int = 768

    # Positive-sample selection follows the original SCCL implementation:
    # select the most similar positives and the least similar (hard) positives.
    top_k: int = 8
    end_k: int = 2

    # Expected number of queue-update keys per query when key batches do not
    # provide their own labels. Explicit key labels take precedence.
    positive_num: int = 1
    sc_positive_num: int = 1

    # ===== Training mixture weights =====
    # The diagnosis weight is computed as 1 - historical rate - SC rate.
    contrastive_rate_in_training: float = 0.2
    sc_rate_in_training: float = 0.2
    w_nps: float = 1.0

    # ===== Diagnosis and contrastive losses =====
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # ===== Miscellaneous =====
    seed: int = 42
    return_dict: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values early and fail with clear messages."""
        if not self.pretrained_model_name_or_path:
            raise ValueError("pretrained_model_name_or_path must not be empty.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")

        if self.num_diag_classes <= 1:
            raise ValueError("num_diag_classes must be greater than 1.")
        if self.num_nps_labels <= 0:
            raise ValueError("num_nps_labels must be positive.")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.queue_size_sc <= 0 or self.queue_size_his <= 0:
            raise ValueError("queue sizes must be positive.")

        if self.top_k < 0 or self.end_k < 0 or self.top_k + self.end_k == 0:
            raise ValueError("top_k and end_k must be non-negative and not both zero.")
        if self.positive_num <= 0 or self.sc_positive_num <= 0:
            raise ValueError("positive_num and sc_positive_num must be positive.")

        if self.contrastive_rate_in_training < 0 or self.sc_rate_in_training < 0:
            raise ValueError("contrastive mixture weights must be non-negative.")
        if self.contrastive_rate_in_training + self.sc_rate_in_training > 1.0:
            raise ValueError(
                "contrastive_rate_in_training + sc_rate_in_training must not exceed 1."
            )
        if self.w_nps < 0:
            raise ValueError("w_nps must be non-negative.")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")

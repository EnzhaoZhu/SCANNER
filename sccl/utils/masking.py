from __future__ import annotations
import torch


def build_has_nps_labels(nps_labels: torch.Tensor | None, *, batch_size: int) -> torch.BoolTensor:
    """
    Build a sample-level mask indicating whether each sample has NPS token labels.

    - If nps_labels is None: all False
    - If nps_labels is provided: default assumes all True
      (for stricter behavior, pass has_nps_labels explicitly from your dataloader).
    """
    if nps_labels is None:
        return torch.zeros(batch_size, dtype=torch.bool)
    return torch.ones(batch_size, dtype=torch.bool)


def apply_sample_mask_to_loss(loss_per_sample: torch.Tensor, has_label: torch.BoolTensor) -> torch.Tensor:
    """
    Mask a per-sample loss vector by has_label and return a mean over valid samples.
    """
    if has_label.sum() == 0:
        return loss_per_sample.new_zeros(())
    return loss_per_sample[has_label].mean()

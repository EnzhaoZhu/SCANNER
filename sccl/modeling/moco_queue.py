from __future__ import annotations
from typing import Tuple

import torch
from torch import nn


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    L2-normalize the last dimension of a tensor in a numerically stable way.
    """
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class MoCoQueue(nn.Module):
    """
    A ring-buffer queue that stores (feature, label) pairs for MoCo-style contrastive learning.

    This module is intentionally minimal and framework-agnostic:
      - No distributed all_gather inside (keep public repo lightweight).
      - Enqueue supports variable last-batch sizes safely.

    Buffers:
      - feature_queue: [K, D]
      - label_queue:   [K]
      - ptr:           scalar long tensor
    """

    def __init__(self, queue_size: int, feature_dim: int, *, init_labels_range: int = 2):
        super().__init__()
        K = int(queue_size)

        self.K = K
        self.register_buffer("feature_queue", torch.randn(K, feature_dim))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        # Initialize labels to a plausible range; real labels will overwrite during training.
        self.register_buffer("label_queue", torch.randint(0, int(init_labels_range), (K,), dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Enqueue a batch into the ring buffer.

        Args:
          keys:   [bs, D] normalized features (recommended)
          labels: [bs]    integer labels aligned with keys
        """
        keys = keys.detach()
        labels = labels.detach().to(dtype=torch.long, device=self.label_queue.device)

        bs = int(keys.shape[0])
        if bs <= 0:
            return

        ptr = int(self.ptr.item())
        end = ptr + bs

        if end <= self.K:
            self.feature_queue[ptr:end] = keys
            self.label_queue[ptr:end] = labels
        else:
            head = self.K - ptr
            self.feature_queue[ptr:] = keys[:head]
            self.label_queue[ptr:] = labels[:head]

            tail = bs - head
            self.feature_queue[:tail] = keys[head:]
            self.label_queue[:tail] = labels[head:]

        self.ptr[...] = (ptr + bs) % self.K

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          feature_queue: [K, D]
          label_queue:   [K]
        """
        return self.feature_queue, self.label_queue

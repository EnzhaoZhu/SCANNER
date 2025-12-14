"""
Fuller training example for the SCCL public architecture repo.

This script is designed as a *runnable training loop demo* (not a reproducibility benchmark):
  - It uses random token ids and random labels (no private data).
  - It verifies end-to-end wiring: forward -> losses -> backward -> optimizer step.
  - It exercises two-stage contrastive inputs (HIS + SC queues) and partial-label NPS masking.
  - It can optionally exercise CRF-NLL training if `use_crf=True` and `torchcrf` is installed.

Expected takeaway:
  - "The architecture is trainable end-to-end and interfaces are stable."
Not guaranteed:
  - meaningful learning dynamics (random data), or paper-level performance replication.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from sccl.config import SCCLConfig
from sccl.modeling.sccl import SCCLModel


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 7) -> None:
    """Best-effort deterministic setup for demo purposes."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism flags (may reduce speed; OK for a demo).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DummyBatchConfig:
    """Controls shapes of random batches for this demo."""
    batch_size: int = 4
    seq_len: int = 48
    vocab_size: int = 32000
    device: str = "cpu"


def make_query_batch(cfg: SCCLConfig, bcfg: DummyBatchConfig) -> Dict[str, torch.Tensor]:
    """
    Build a query batch that matches SCCL forward signature.

    Required keys:
      - input_ids, token_type_ids, attention_mask, labels

    Optional keys (to exercise NPS + masking):
      - nps_labels, has_nps_labels
    """
    bs, L = bcfg.batch_size, bcfg.seq_len
    device = bcfg.device

    query = {
        "input_ids": torch.randint(0, bcfg.vocab_size, (bs, L), dtype=torch.long, device=device),
        "token_type_ids": torch.zeros((bs, L), dtype=torch.long, device=device),
        "attention_mask": torch.ones((bs, L), dtype=torch.long, device=device),
        "labels": torch.randint(0, cfg.num_diag_classes, (bs,), dtype=torch.long, device=device),
    }

    # NPS labels (token-level). We intentionally mix labeled and unlabeled samples.
    # For unlabeled samples, has_nps_labels=False will mask them out at the sample level.
    if cfg.num_nps_labels > 0:
        nps_labels = torch.randint(0, cfg.num_nps_labels, (bs, L), dtype=torch.long, device=device)

        # Optionally inject ignore_index tokens to simulate partially annotated sequences.
        # This is especially important for CRF-NLL path (we mask ignore_index positions).
        if cfg.nps_ignore_index is not None:
            # randomly set ~10% tokens to ignore_index
            ignore_mask = torch.rand((bs, L), device=device) < 0.10
            nps_labels[ignore_mask] = cfg.nps_ignore_index

        query["nps_labels"] = nps_labels

        # Sample-level mask: first half labeled, second half unlabeled (demo).
        has = torch.zeros((bs,), dtype=torch.bool, device=device)
        has[: bs // 2] = True
        query["has_nps_labels"] = has

    return query


def make_positive_batch(bcfg: DummyBatchConfig, *, num_pos: int) -> Dict[str, torch.Tensor]:
    """
    Build positive samples in shape [bs, num_pos, L], matching SCCL flattening logic.
    """
    bs, L = bcfg.batch_size, bcfg.seq_len
    device = bcfg.device

    return {
        "input_ids": torch.randint(0, bcfg.vocab_size, (bs, num_pos, L), dtype=torch.long, device=device),
        "token_type_ids": torch.zeros((bs, num_pos, L), dtype=torch.long, device=device),
        "attention_mask": torch.ones((bs, num_pos, L), dtype=torch.long, device=device),
    }


def maybe_build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """
    Optional linear warmup + cosine decay scheduler.
    This is purely for a realistic-looking training loop.
    """
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # cosine decay to 10% of the initial LR
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# -----------------------------
# Main training demo
# -----------------------------

def main() -> None:
    set_seed(7)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    # IMPORTANT: use the Chinese base checkpoint by default to match your repo intent.
    # You can leave pretrained_model_name_or_path unset to use SCCLConfig defaults.
    cfg = SCCLConfig(
        pretrained_model_name_or_path="../xlnet_base",
        num_diag_classes=5,
        num_nps_labels=27,
        use_crf=True,  # enable CRF-NLL if torchcrf is installed; else it will fall back.
        queue_size_his=256,
        queue_size_sc=256,
        top_k=8,
        end_k=2,
        positive_num=2,
        sc_positive_num=2,
        contrastive_rate_in_training=0.2,
        sc_rate_in_training=0.2,
        use_focal_loss=True,
        focal_gamma=2.0,
    )

    model = SCCLModel(cfg).to(device).train()

    # AdamW is a sensible default for transformer-style models.
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    total_steps = 10
    scheduler = maybe_build_scheduler(optimizer, total_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    bcfg = DummyBatchConfig(batch_size=4, seq_len=48, vocab_size=32000, device=device)

    # Show initial queue pointers (to verify they update).
    his_ptr0 = int(model.queue_his.ptr.item())
    sc_ptr0 = int(model.queue_sc.ptr.item())
    print(f"[init] queue_his.ptr={his_ptr0}, queue_sc.ptr={sc_ptr0}")

    t0 = time.time()

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        query = make_query_batch(cfg, bcfg)
        positive_sample = make_positive_batch(bcfg, num_pos=cfg.positive_num)
        positive_sc = make_positive_batch(bcfg, num_pos=cfg.sc_positive_num)

        # Mixed precision forward (CUDA only).
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            out = model(
                query=query,
                positive_sample=positive_sample,
                positive_sc=positive_sc,
                return_dict=True,
            )
            loss = out["loss"]

        # Backward
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first when using AMP)
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step optimizer
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Logging (loss components may be tensors or None)
        loss_cls = float(out.get("loss_cls", torch.tensor(0.0)).item()) if out.get("loss_cls") is not None else 0.0
        loss_his = float(out.get("loss_con_his", torch.tensor(0.0)).item()) if out.get("loss_con_his") is not None else 0.0
        loss_sc = float(out.get("loss_con_sc", torch.tensor(0.0)).item()) if out.get("loss_con_sc") is not None else 0.0
        loss_nps = float(out.get("loss_nps", torch.tensor(0.0)).item()) if out.get("loss_nps") is not None else 0.0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[step {step:02d}] "
            f"loss={float(loss.item()):.4f} | "
            f"cls={loss_cls:.4f} his={loss_his:.4f} sc={loss_sc:.4f} nps={loss_nps:.4f} | "
            f"lr={lr:.2e}"
        )

    t1 = time.time()

    # Show queue pointers after training
    his_ptr1 = int(model.queue_his.ptr.item())
    sc_ptr1 = int(model.queue_sc.ptr.item())
    print(f"[done] queue_his.ptr={his_ptr1}, queue_sc.ptr={sc_ptr1} (elapsed {t1 - t0:.1f}s)")

    # Minimal sanity: pointers should have advanced if positives were provided.
    # If you see no change, positive_sample/positive_sc may not be wired correctly.
    if his_ptr1 == his_ptr0 or sc_ptr1 == sc_ptr0:
        print("[warn] queue pointers did not change as expected. Check positive inputs and enqueue logic.")
    else:
        print("[ok] queues updated; training loop executed end-to-end.")


if __name__ == "__main__":
    main()

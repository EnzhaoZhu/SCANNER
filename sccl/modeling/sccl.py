from __future__ import annotations
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..config import SCCLConfig
from ..utils.masking import build_has_nps_labels
from .backbone_xlnet import XLNetBackbone
from .heads import DiagnosisHead, ProjectionHead, NPSHead
from .moco_queue import MoCoQueue, l2norm
from .contrastive import ContrastiveSamplingConfig, select_pos_neg_logits_from_queue
from .losses import MultiTaskLoss


@torch.no_grad()
def momentum_update(q_encoder: nn.Module, k_encoder: nn.Module, m: float) -> None:
    """
    MoCo momentum update:
      theta_k <- m * theta_k + (1-m) * theta_q
    """
    for p_q, p_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=(1.0 - m))


class SCCLModel(nn.Module):
    """
    SCCL architecture (public): two-stage contrastive learning + multi-task heads.

    Inputs (training):
      query: dict with keys:
        - input_ids, token_type_ids, attention_mask
        - labels (diagnosis labels)
        - optionally nps_labels (token-level)
        - optionally has_nps_labels (sample-level bool mask)
      positive_sample: dict (same tensor keys), used only to update HIS queue
      positive_sc: dict (same tensor keys), used only to update SC queue

    Outputs:
      - training: loss dict including loss_cls, loss_con_his, loss_con_sc, (optional) loss_nps
      - inference: diag_probs + (optional) NPS decoded tags/spans
    """

    def __init__(self, cfg: SCCLConfig):
        super().__init__()
        self.cfg = cfg

        # ===== Encoders =====
        self.encoder_q = XLNetBackbone(cfg.pretrained_model_name_or_path)
        self.encoder_k_his = XLNetBackbone(cfg.pretrained_model_name_or_path)
        self.encoder_k_sc = XLNetBackbone(cfg.pretrained_model_name_or_path)
        self._init_key_encoder(self.encoder_k_his)
        self._init_key_encoder(self.encoder_k_sc)

        # ===== Heads =====
        self.diag_head = DiagnosisHead(cfg.hidden_size, cfg.num_diag_classes, cfg.dropout)
        self.proj_q = ProjectionHead(cfg.hidden_size, cfg.dropout)
        self.proj_k = ProjectionHead(cfg.hidden_size, cfg.dropout)  # momentum-updated projection for keys
        self._init_key_proj()

        self.nps_head: Optional[NPSHead] = None
        if cfg.num_nps_labels and cfg.num_nps_labels > 0:
            self.nps_head = NPSHead(cfg.hidden_size, cfg.num_nps_labels, cfg.dropout, use_crf=cfg.use_crf)

        # ===== Queues =====
        self.queue_his = MoCoQueue(queue_size=cfg.queue_size_his, feature_dim=cfg.hidden_size, init_labels_range=cfg.num_diag_classes)
        self.queue_sc = MoCoQueue(queue_size=cfg.queue_size_sc, feature_dim=cfg.hidden_size, init_labels_range=cfg.num_diag_classes)

        # ===== Sampling cfg (ported from original SCCL) =====
        self.samp_cfg = ContrastiveSamplingConfig(top_k=cfg.top_k, end_k=cfg.end_k, temperature=cfg.temperature)

        # ===== Loss =====
        self.losses = MultiTaskLoss(
            use_focal=cfg.use_focal_loss,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
            nps_ignore_index=cfg.nps_ignore_index,
        )

    def _init_key_encoder(self, enc_k: nn.Module) -> None:
        """
        Initialize key encoder as a copy of the query encoder, then freeze gradients.
        """
        enc_k.load_state_dict(self.encoder_q.state_dict(), strict=True)
        for p in enc_k.parameters():
            p.requires_grad_(False)

    def _init_key_proj(self) -> None:
        """
        Initialize key projection head as a copy of query projection head; freeze grads.
        """
        self.proj_k.load_state_dict(self.proj_q.state_dict(), strict=True)
        for p in self.proj_k.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _flatten_batch_dict(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Flatten a dict of tensors shaped [bs, n, L] into [bs*n, L] when needed.

        This mirrors your original reshape_dict but is more defensive:
          - only flattens tensors with >=2 dims
          - preserves dtype/device
        """
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.ndim >= 2:
                out[k] = v.view(-1, v.shape[-1])
            else:
                out[k] = v
        return out

    def forward(
        self,
        query: Dict[str, torch.Tensor],
        positive_sample: Optional[Dict[str, torch.Tensor]] = None,
        positive_sc: Optional[Dict[str, torch.Tensor]] = None,
        *,
        id2tag: Optional[Dict[int, str]] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        A single forward that supports both training and inference.

        Inference mode is triggered by self.training == False, consistent with your original code.
        """
        # ---- Basic tensors ----
        labels = query.get("labels", None)
        if labels is None:
            raise ValueError("query must contain 'labels' (diagnosis labels) to match SCCL interface.")

        labels = labels.view(-1).long()

        # ---- Query encoder forward (always) ----
        q_inputs = {k: query[k] for k in ["input_ids", "token_type_ids", "attention_mask"] if k in query}
        out_q = self.encoder_q(**q_inputs)
        pooled_q = out_q.pooled_output
        seq_q = out_q.sequence_output

        diag_logits = self.diag_head(pooled_q)
        diag_probs = torch.softmax(diag_logits, dim=-1)

        # ---- Optional NPS logits (for both train/infer) ----
        nps_token_logits = None
        if self.nps_head is not None:
            nps_token_logits = self.nps_head.forward_logits(seq_q)

        # =====================
        # Inference branch
        # =====================
        if not self.training:
            pred: Dict[str, Any] = {
                "diag_logits": diag_logits,
                "diag_probs": diag_probs,
            }
            if self.nps_head is not None and nps_token_logits is not None:
                decoded = self.nps_head.decode(nps_token_logits, query.get("attention_mask", None), id2tag=id2tag)
                pred.update({"nps_token_logits": nps_token_logits, **decoded})
            return pred if return_dict else (pred,)

        # =====================
        # Training branch
        # =====================

        # ---- 1) Momentum update for key encoders & key projection ----
        with torch.no_grad():
            momentum_update(self.encoder_q, self.encoder_k_his, self.cfg.momentum)
            momentum_update(self.encoder_q, self.encoder_k_sc, self.cfg.momentum)

            # projection head momentum update (mirrors your original contrastive_liner_q/k update)
            for p_q, p_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
                p_k.data.mul_(self.cfg.momentum).add_(p_q.data, alpha=(1.0 - self.cfg.momentum))

        # ---- 2) Update queues using positive_sample / positive_sc (as in your original) ----
        with torch.no_grad():
            if positive_sample is not None:
                ps = self._flatten_batch_dict(positive_sample)
                ps_inputs = {k: ps[k] for k in ["input_ids", "token_type_ids", "attention_mask"] if k in ps}
                pooled_k = self.encoder_k_his(**ps_inputs).pooled_output
                k_feat = l2norm(self.proj_k(pooled_k))

                # Repeat labels to match the number of positive samples per query (positive_num)
                tmp = labels.view(-1, 1).repeat(1, self.cfg.positive_num).view(-1)
                self.queue_his.enqueue(k_feat, tmp)

            if positive_sc is not None:
                scb = self._flatten_batch_dict(positive_sc)
                sc_inputs = {k: scb[k] for k in ["input_ids", "token_type_ids", "attention_mask"] if k in scb}
                pooled_sc = self.encoder_k_sc(**sc_inputs).pooled_output
                sc_feat = l2norm(self.proj_k(pooled_sc))

                tmp = labels.view(-1, 1).repeat(1, self.cfg.sc_positive_num).view(-1)
                self.queue_sc.enqueue(sc_feat, tmp)

        # ---- 3) Build query contrastive features ----
        q_feat = l2norm(self.proj_q(pooled_q))

        # ---- 4) Contrastive logits from HIS queue (ported select_pos_neg_sample) ----
        his_feats, his_labels = self.queue_his.get()
        logits_his = select_pos_neg_logits_from_queue(
            q_feat, labels, his_feats, his_labels, self.samp_cfg
        )

        # ---- 5) Contrastive logits from SC queue (ported select_pos_neg_sample_sc) ----
        sc_feats, sc_labels = self.queue_sc.get()
        logits_sc = select_pos_neg_logits_from_queue(
            q_feat, labels, sc_feats, sc_labels, self.samp_cfg
        )

        # ---- 6) Losses ----
        loss_cls = self.losses.loss_diag(diag_logits, labels)

        loss_con_his = diag_logits.new_zeros(())
        loss_con_sc = diag_logits.new_zeros(())

        if logits_his is not None:
            loss_con_his = self.losses.loss_infonce(logits_his)

        if logits_sc is not None:
            loss_con_sc = self.losses.loss_infonce(logits_sc)

        # Optional NPS loss with partial-label masking
        loss_nps = diag_logits.new_zeros(())
        if self.nps_head is not None and nps_token_logits is not None and ("nps_labels" in query):
            nps_labels = query["nps_labels"]
            has_mask = query.get("has_nps_labels", None)
            if has_mask is None:
                has_mask = build_has_nps_labels(nps_labels, batch_size=int(labels.shape[0])).to(nps_token_logits.device)

            if self.cfg.use_crf and getattr(self.nps_head, "use_crf", False):
                # CRF sequence-level NLL (matches Methods)
                loss_nps = self.losses.loss_nps_crf_nll(
                    self.nps_head,
                    nps_token_logits,
                    nps_labels,
                    query.get("attention_mask", None),
                    has_mask,
                )
            else:
                # Token-level CE fallback (public repo default if CRF not installed/enabled)
                loss_nps = self.losses.loss_nps_token_ce(
                    nps_token_logits,
                    nps_labels,
                    query.get("attention_mask", None),
                    has_mask,
                )

        # ---- 7) Mixture (match your original weighting semantics) ----
        w_his = float(self.cfg.contrastive_rate_in_training)
        w_sc = float(self.cfg.sc_rate_in_training)
        w_cls = max(0.0, 1.0 - w_his - w_sc)

        loss = w_cls * loss_cls + w_his * loss_con_his + w_sc * loss_con_sc + self.cfg.w_nps * loss_nps

        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_con_his": loss_con_his.detach(),
            "loss_con_sc": loss_con_sc.detach(),
            "loss_nps": loss_nps.detach(),
            "diag_logits": diag_logits.detach(),
            "diag_probs": diag_probs.detach(),
            "nps_token_logits": nps_token_logits.detach() if nps_token_logits is not None else None,
        }

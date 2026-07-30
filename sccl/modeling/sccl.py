from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from config import SCCLConfig
from utils.masking import build_has_nps_labels
from modeling.backbone_xlnet import XLNetBackbone
from modeling.heads import DiagnosisHead, NPSHead, ProjectionHead
from modeling.moco_queue import MoCoQueue, l2norm
from modeling.contrastive import ContrastiveSamplingConfig, select_pos_neg_logits_from_queue
from modeling.losses import MultiTaskLoss


@torch.no_grad()
def momentum_update(q_encoder: nn.Module, k_encoder: nn.Module, m: float) -> None:
    """Apply the MoCo exponential moving-average update to a key encoder."""
    for p_q, p_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=(1.0 - m))


class SCCLModel(nn.Module):
    """
    Standard-case Contrastive Clinical Learning model.

    Training input ``query`` contains tokenized SCIL/DSPA text and diagnosis
    labels, with optional token-level NPS labels. ``positive_sample`` and
    ``positive_sc`` supply historical and standard-case keys for queue updates.
    Their own diagnosis labels are used when provided.

    In evaluation mode, only tokenized text is required. The model returns
    diagnostic probabilities and, when the NPS head is enabled, decoded NPS
    tags and spans.
    """

    _ENCODER_KEYS = ("input_ids", "token_type_ids", "attention_mask")

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
        self.diag_head = DiagnosisHead(
            cfg.hidden_size, cfg.num_diag_classes, cfg.dropout
        )
        self.proj_q = ProjectionHead(cfg.hidden_size, cfg.dropout)
        self.proj_k = ProjectionHead(cfg.hidden_size, cfg.dropout)
        self._init_key_proj()

        self.nps_head: Optional[NPSHead] = None
        if cfg.num_nps_labels > 0:
            self.nps_head = NPSHead(
                cfg.hidden_size,
                cfg.num_nps_labels,
                cfg.dropout,
                use_crf=cfg.use_crf,
            )

        # ===== Queues =====
        self.queue_his = MoCoQueue(
            queue_size=cfg.queue_size_his,
            feature_dim=cfg.hidden_size,
            init_labels_range=cfg.num_diag_classes,
        )
        self.queue_sc = MoCoQueue(
            queue_size=cfg.queue_size_sc,
            feature_dim=cfg.hidden_size,
            init_labels_range=cfg.num_diag_classes,
        )

        # ===== Contrastive sampling =====
        self.samp_cfg = ContrastiveSamplingConfig(
            top_k=cfg.top_k,
            end_k=cfg.end_k,
            temperature=cfg.temperature,
        )

        # ===== Losses =====
        self.losses = MultiTaskLoss(
            use_focal=cfg.use_focal_loss,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
            nps_ignore_index=cfg.nps_ignore_index,
        )

    def _init_key_encoder(self, enc_k: nn.Module) -> None:
        """Copy the query encoder into a frozen momentum key encoder."""
        enc_k.load_state_dict(self.encoder_q.state_dict(), strict=True)
        for parameter in enc_k.parameters():
            parameter.requires_grad_(False)

    def _init_key_proj(self) -> None:
        """Copy the query projection head into a frozen key projection head."""
        self.proj_k.load_state_dict(self.proj_q.state_dict(), strict=True)
        for parameter in self.proj_k.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def _encoder_inputs(cls, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract the tensors accepted by the XLNet backbone."""
        if "input_ids" not in batch:
            raise ValueError("Each model batch must contain 'input_ids'.")
        return {key: batch[key] for key in cls._ENCODER_KEYS if key in batch}

    @staticmethod
    def _flatten_key_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Flatten optional multiple keys per query.

        Sequence tensors may be shaped ``[bs, n_key, length]`` and are reshaped
        to ``[bs * n_key, length]``. Key-level labels may be shaped
        ``[bs, n_key]`` and are flattened to ``[bs * n_key]``. Already-flat
        tensors are left unchanged.
        """
        flattened: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue

            if key in {"labels", "has_nps_labels"}:
                flattened[key] = value.reshape(-1)
            elif value.ndim >= 3:
                flattened[key] = value.reshape(-1, *value.shape[2:])
            else:
                flattened[key] = value
        return flattened

    @staticmethod
    def _resolve_key_labels(
        key_batch: Dict[str, torch.Tensor],
        query_labels: torch.Tensor,
        *,
        expected_keys: int,
        fallback_keys_per_query: int,
        queue_name: str,
    ) -> torch.Tensor:
        """
        Resolve diagnosis labels for queue-update keys.

        Explicit labels in the key batch take precedence. When absent, query
        labels are repeated according to the configured number of keys per
        query, matching the behavior of the original SCCL implementation.
        """
        explicit = key_batch.get("labels")
        if explicit is not None:
            key_labels = explicit.reshape(-1).long()
        else:
            expected_from_config = int(query_labels.numel()) * int(
                fallback_keys_per_query
            )
            if expected_from_config != expected_keys:
                raise ValueError(
                    f"{queue_name} provided {expected_keys} key features but no key "
                    f"labels; expected {expected_from_config} from the configured "
                    f"keys-per-query value. Provide explicit '{queue_name}' labels "
                    "or correct the configuration."
                )
            key_labels = query_labels.repeat_interleave(fallback_keys_per_query)

        if int(key_labels.numel()) != expected_keys:
            raise ValueError(
                f"{queue_name} has {expected_keys} key features but "
                f"{int(key_labels.numel())} diagnosis labels."
            )
        return key_labels

    def forward(
        self,
        query: Dict[str, torch.Tensor],
        positive_sample: Optional[Dict[str, torch.Tensor]] = None,
        positive_sc: Optional[Dict[str, torch.Tensor]] = None,
        *,
        id2tag: Optional[Dict[int, str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run SCCL training or inference according to ``self.training``."""
        if return_dict is None:
            return_dict = self.cfg.return_dict

        # ===== Shared query forward pass =====
        out_q = self.encoder_q(**self._encoder_inputs(query))
        pooled_q = out_q.pooled_output
        sequence_q = out_q.sequence_output

        diag_logits = self.diag_head(pooled_q)
        diag_probs = torch.softmax(diag_logits, dim=-1)
        diag_pred_ids = diag_probs.argmax(dim=-1)

        nps_token_logits = None
        if self.nps_head is not None:
            nps_token_logits = self.nps_head.forward_logits(sequence_q)

        # ===== Inference =====
        # Diagnosis labels are intentionally not required in evaluation mode.
        if not self.training:
            prediction: Dict[str, Any] = {
                "diag_logits": diag_logits,
                "diag_probs": diag_probs,
                "diag_pred_ids": diag_pred_ids,
            }
            if self.nps_head is not None and nps_token_logits is not None:
                decoded = self.nps_head.decode(
                    nps_token_logits,
                    query.get("attention_mask"),
                    id2tag=id2tag,
                )
                prediction.update(
                    {"nps_token_logits": nps_token_logits, **decoded}
                )
            return prediction if return_dict else (prediction,)  # type: ignore[return-value]

        # ===== Training =====
        labels = query.get("labels")
        if labels is None:
            raise ValueError("Training queries must contain diagnosis 'labels'.")
        labels = labels.reshape(-1).long()
        if int(labels.numel()) != int(diag_logits.shape[0]):
            raise ValueError(
                "The number of query diagnosis labels does not match the query batch size."
            )

        # 1) Momentum-update the key encoders and key projection head.
        with torch.no_grad():
            momentum_update(self.encoder_q, self.encoder_k_his, self.cfg.momentum)
            momentum_update(self.encoder_q, self.encoder_k_sc, self.cfg.momentum)
            momentum_update(self.proj_q, self.proj_k, self.cfg.momentum)

        # 2) Encode key batches and update historical and standard-case queues.
        with torch.no_grad():
            if positive_sample is not None:
                historical_batch = self._flatten_key_batch(positive_sample)
                pooled_historical = self.encoder_k_his(
                    **self._encoder_inputs(historical_batch)
                ).pooled_output
                historical_features = l2norm(self.proj_k(pooled_historical))
                historical_labels = self._resolve_key_labels(
                    historical_batch,
                    labels,
                    expected_keys=int(historical_features.shape[0]),
                    fallback_keys_per_query=self.cfg.positive_num,
                    queue_name="positive_sample",
                ).to(historical_features.device)
                self.queue_his.enqueue(historical_features, historical_labels)

            if positive_sc is not None:
                standard_case_batch = self._flatten_key_batch(positive_sc)
                pooled_standard_case = self.encoder_k_sc(
                    **self._encoder_inputs(standard_case_batch)
                ).pooled_output
                standard_case_features = l2norm(
                    self.proj_k(pooled_standard_case)
                )
                standard_case_labels = self._resolve_key_labels(
                    standard_case_batch,
                    labels,
                    expected_keys=int(standard_case_features.shape[0]),
                    fallback_keys_per_query=self.cfg.sc_positive_num,
                    queue_name="positive_sc",
                ).to(standard_case_features.device)
                self.queue_sc.enqueue(
                    standard_case_features, standard_case_labels
                )

        # 3) Construct query projections and contrastive logits.
        query_features = l2norm(self.proj_q(pooled_q))

        historical_queue_features, historical_queue_labels = self.queue_his.get()
        logits_historical = select_pos_neg_logits_from_queue(
            query_features,
            labels,
            historical_queue_features,
            historical_queue_labels,
            self.samp_cfg,
        )

        standard_queue_features, standard_queue_labels = self.queue_sc.get()
        logits_standard_case = select_pos_neg_logits_from_queue(
            query_features,
            labels,
            standard_queue_features,
            standard_queue_labels,
            self.samp_cfg,
        )

        # 4) Diagnosis and contrastive losses.
        loss_cls = self.losses.loss_diag(diag_logits, labels)
        loss_con_his = diag_logits.new_zeros(())
        loss_con_sc = diag_logits.new_zeros(())

        if logits_historical is not None:
            loss_con_his = self.losses.loss_infonce(logits_historical)
        if logits_standard_case is not None:
            loss_con_sc = self.losses.loss_infonce(logits_standard_case)

        # 5) Optional NPS loss with sample-level partial-label masking.
        loss_nps = diag_logits.new_zeros(())
        if (
            self.nps_head is not None
            and nps_token_logits is not None
            and "nps_labels" in query
        ):
            nps_labels = query["nps_labels"]
            has_mask = query.get("has_nps_labels")
            if has_mask is None:
                has_mask = build_has_nps_labels(
                    nps_labels,
                    batch_size=int(labels.shape[0]),
                )
            has_mask = has_mask.reshape(-1).bool().to(nps_token_logits.device)

            if int(has_mask.numel()) != int(labels.numel()):
                raise ValueError(
                    "has_nps_labels must contain one value per query sample."
                )

            if self.cfg.use_crf and getattr(self.nps_head, "use_crf", False):
                loss_nps = self.losses.loss_nps_crf_nll(
                    self.nps_head,
                    nps_token_logits,
                    nps_labels,
                    query.get("attention_mask"),
                    has_mask,
                )
            else:
                loss_nps = self.losses.loss_nps_token_ce(
                    nps_token_logits,
                    nps_labels,
                    query.get("attention_mask"),
                    has_mask,
                )

        # 6) Weighted objective. The two contrastive rates retain the original
        # SCCL semantics; the remaining weight is assigned to diagnosis loss.
        weight_historical = float(self.cfg.contrastive_rate_in_training)
        weight_standard = float(self.cfg.sc_rate_in_training)
        weight_diagnosis = 1.0 - weight_historical - weight_standard

        total_loss = (
            weight_diagnosis * loss_cls
            + weight_historical * loss_con_his
            + weight_standard * loss_con_sc
            + float(self.cfg.w_nps) * loss_nps
        )

        result: Dict[str, Any] = {
            "loss": total_loss,
            "loss_cls": loss_cls.detach(),
            "loss_con_his": loss_con_his.detach(),
            "loss_con_sc": loss_con_sc.detach(),
            "loss_nps": loss_nps.detach(),
            "diag_logits": diag_logits.detach(),
            "diag_probs": diag_probs.detach(),
            "diag_pred_ids": diag_pred_ids.detach(),
            "nps_token_logits": (
                nps_token_logits.detach() if nps_token_logits is not None else None
            ),
        }
        return result if return_dict else (result,)  # type: ignore[return-value]

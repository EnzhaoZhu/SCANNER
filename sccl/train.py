"""
Toy-data training workflow for the public SCCL reference implementation.

This script replaces the former random-tensor demonstration with a small,
privacy-preserving workflow based on manually authored examples:

Stage 1
    Diagnosis-only pre-adaptation on synthetic DSPA records.

Stage 2
    Multitask fine-tuning on synthetic SCIL question-answer records, with
    historical-sample and standard-case contrastive queues.

The examples are intended only to demonstrate data schemas, preprocessing,
partial-label masking, queue updates, and end-to-end optimization. They are
not a benchmark and cannot reproduce the clinical performance reported in
the manuscript.
"""

from __future__ import annotations

import argparse

import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

# Make the project root importable when this file is executed directly from
# the ``sccl`` directory, e.g. ``python train.py`` or ``python inference.py``.
# Importing through the package name keeps relative imports inside
# ``sccl/modeling`` valid.
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SCCLConfig
from modeling.sccl import SCCLModel
from labels import (
    BIO_LABELS,
    BIO_TO_ID,
    DIAGNOSIS_LABELS,
    DIAGNOSIS_TO_ID,
    NPS_TYPES,
)

from data import (
    class_representatives,
    encode_dspa_record,
    encode_scil_record,
    encode_standard_case_record,
    iter_batches,
    load_fast_tokenizer,
    move_batch,
    read_jsonl,
    stack_batch,
)




def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run the privacy-preserving two-stage SCCL toy workflow."
    )
    parser.add_argument(
        "--backbone",
        default=str(project_dir / "xlnet_base"),
        help=(
            "Local XLNet checkpoint directory or a compatible Hugging Face "
            "checkpoint identifier. The default expects xlnet_base in the "
            "project root."
        ),
    )
    parser.add_argument(
        "--scil-data",
        type=Path,
        default=script_dir / "examples" / "toy_scil.jsonl",
    )
    parser.add_argument(
        "--dspa-data",
        type=Path,
        default=script_dir / "examples" / "toy_dspa.jsonl",
    )
    parser.add_argument(
        "--standard-case-data",
        type=Path,
        default=script_dir / "examples" / "toy_standard_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "outputs" / "toy_sccl.pt",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-crf",
        action="store_true",
        help="Use CRF NPS loss when pytorch-crf is installed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def validate_diagnosis(record: Mapping[str, Any]) -> int:
    diagnosis = record.get("diagnosis")
    if diagnosis not in DIAGNOSIS_TO_ID:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} has unsupported diagnosis "
            f"{diagnosis!r}. Expected one of {DIAGNOSIS_LABELS}."
        )
    return DIAGNOSIS_TO_ID[str(diagnosis)]


def build_scil_text(
    record: Mapping[str, Any],
) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """
    Concatenate ordered QA pairs and convert answer-relative spans to global
    character offsets.

    The source toy file uses end-exclusive offsets within the answer field.
    """
    qa_pairs = record.get("qa_pairs")
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError(
            f"SCIL record {record.get('id', '<unknown>')} has no qa_pairs."
        )

    pieces: List[str] = []
    answer_starts: List[int] = []
    cursor = 0

    for pair in qa_pairs:
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()

        question_part = f"[Q] {question}\n[A] "
        pieces.append(question_part)
        cursor += len(question_part)

        answer_starts.append(cursor)
        pieces.append(answer)
        cursor += len(answer)

        pieces.append("\n")
        cursor += 1

    text = "".join(pieces)

    global_spans: List[Dict[str, Any]] = []
    has_labels = bool(record.get("has_nps_labels", False))
    source_spans = record.get("nps_spans")

    if has_labels:
        if source_spans is None:
            raise ValueError(
                f"SCIL record {record.get('id')} is marked as labelled but "
                "nps_spans is null."
            )

        for span in source_spans:
            qa_index = int(span["qa_index"])
            if qa_index < 0 or qa_index >= len(qa_pairs):
                raise ValueError(
                    f"Invalid qa_index in record {record.get('id')}: {qa_index}"
                )

            start = int(span["start"])
            end = int(span["end"])
            label = str(span["label"])
            answer = str(qa_pairs[qa_index].get("answer", "")).strip()
            expected_text = str(span.get("text", answer[start:end]))

            if not (0 <= start < end <= len(answer)):
                raise ValueError(
                    f"Invalid span [{start}, {end}) in record {record.get('id')}."
                )
            if answer[start:end] != expected_text:
                raise ValueError(
                    f"Span text mismatch in record {record.get('id')}: "
                    f"{answer[start:end]!r} != {expected_text!r}."
                )
            if label not in NPS_TYPES:
                raise ValueError(
                    f"Unsupported NPS label {label!r} in record {record.get('id')}."
                )

            global_spans.append(
                {
                    "start": answer_starts[qa_index] + start,
                    "end": answer_starts[qa_index] + end,
                    "label": label,
                }
            )

    return text, answer_starts, global_spans




def tokenize_text(
    tokenizer,
    text: str,
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    output = {
        "input_ids": encoded["input_ids"].squeeze(0).long(),
        "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        "offset_mapping": encoded["offset_mapping"].squeeze(0).long(),
    }
    if "token_type_ids" in encoded:
        output["token_type_ids"] = encoded["token_type_ids"].squeeze(0).long()
    else:
        output["token_type_ids"] = torch.zeros_like(output["input_ids"])
    return output


def spans_to_bio_ids(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: Sequence[Mapping[str, Any]],
    *,
    has_nps_labels: bool,
    ignore_index: int,
) -> torch.Tensor:
    """
    Align end-exclusive character spans to token-level BIO labels.

    Padding and special tokens receive ignore_index. For a reviewed record with
    no positive span, valid text tokens receive the O label. For an unannotated
    record, every token receives ignore_index and the sample-level mask is false.
    """
    labels = torch.full(
        (offsets.shape[0],),
        fill_value=ignore_index,
        dtype=torch.long,
    )

    if not has_nps_labels:
        return labels

    valid_token = (
        attention_mask.bool()
        & (offsets[:, 1] > offsets[:, 0])
    )
    labels[valid_token] = BIO_TO_ID["O"]

    for span in spans:
        span_start = int(span["start"])
        span_end = int(span["end"])
        nps_type = str(span["label"])

        overlaps = (
            valid_token
            & (offsets[:, 0] < span_end)
            & (offsets[:, 1] > span_start)
        )
        token_indices = overlaps.nonzero(as_tuple=False).view(-1)
        if token_indices.numel() == 0:
            # The span may have been truncated by max_length.
            continue

        labels[token_indices[0]] = BIO_TO_ID[f"B-{nps_type}"]
        if token_indices.numel() > 1:
            labels[token_indices[1:]] = BIO_TO_ID[f"I-{nps_type}"]

    return labels


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_stage(
    model: SCCLModel,
    query_records: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    historical_keys: Dict[str, torch.Tensor] | None,
    standard_case_keys: Dict[str, torch.Tensor] | None,
    stage_name: str,
) -> None:
    if epochs <= 0:
        print(f"[{stage_name}] skipped because epochs={epochs}.")
        return

    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    steps_per_epoch = math.ceil(len(query_records) / batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = build_scheduler(optimizer, total_steps=total_steps)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0

    print(
        f"[{stage_name}] records={len(query_records)}, epochs={epochs}, "
        f"batch_size={batch_size}"
    )
    start_time = time.time()

    for epoch in range(epochs):
        for batch_records in iter_batches(
            query_records,
            batch_size=batch_size,
            shuffle=True,
        ):
            query = move_batch(stack_batch(batch_records), device)
            positive_sample = (
                move_batch(historical_keys, device)
                if historical_keys is not None
                else None
            )

            # Standard cases contain only the five target disorders. For a
            # control-only batch, standard-case contrastive learning is omitted.
            query_has_non_control = bool((query["labels"] != DIAGNOSIS_TO_ID["CON"]).any())
            positive_sc = (
                move_batch(standard_case_keys, device)
                if standard_case_keys is not None and query_has_non_control
                else None
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                output = model(
                    query=query,
                    positive_sample=positive_sample,
                    positive_sc=positive_sc,
                    return_dict=True,
                )
                loss = output["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered during {stage_name}."
                )

            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            print(
                f"[{stage_name} epoch={epoch + 1} step={global_step}] "
                f"loss={float(loss.item()):.4f} "
                f"diag={float(output['loss_cls'].item()):.4f} "
                f"his={float(output['loss_con_his'].item()):.4f} "
                f"sc={float(output['loss_con_sc'].item()):.4f} "
                f"nps={float(output['loss_nps'].item()):.4f}"
            )

    print(
        f"[{stage_name}] completed in {time.time() - start_time:.1f}s."
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print(f"[setup] backbone={args.backbone}")

    tokenizer = load_fast_tokenizer(args.backbone)

    raw_dspa = read_jsonl(args.dspa_data)
    raw_scil = read_jsonl(args.scil_data)
    raw_standard = read_jsonl(args.standard_case_data)

    encoded_dspa = [
        encode_dspa_record(record, tokenizer, max_length=args.max_length)
        for record in raw_dspa
    ]
    encoded_scil = [
        encode_scil_record(
            record,
            tokenizer,
            max_length=args.max_length,
            ignore_index=-100,
        )
        for record in raw_scil
    ]

    historical_representatives = class_representatives(raw_dspa)
    standard_representatives = class_representatives(raw_standard)

    encoded_historical_keys = [
        encode_dspa_record(record, tokenizer, max_length=args.max_length)
        for record in historical_representatives
    ]
    encoded_standard_keys = [
        encode_standard_case_record(record, tokenizer, max_length=args.max_length)
        for record in standard_representatives
    ]

    historical_key_batch = stack_batch(encoded_historical_keys)
    standard_key_batch = stack_batch(encoded_standard_keys)

    common_config = dict(
        pretrained_model_name_or_path=args.backbone,
        hidden_size=768,
        max_length=args.max_length,
        num_diag_classes=len(DIAGNOSIS_LABELS),
        num_nps_labels=len(BIO_LABELS),
        use_crf=args.use_crf,
        queue_size_his=64,
        queue_size_sc=64,
        top_k=2,
        end_k=1,
        positive_num=1,
        sc_positive_num=1,
        use_focal_loss=True,
        focal_gamma=2.0,
        seed=args.seed,
    )

    # Stage 1 uses diagnosis loss only.
    stage1_config = SCCLConfig(
        **common_config,
        contrastive_rate_in_training=0.0,
        sc_rate_in_training=0.0,
        w_nps=0.0,
    )
    stage1_model = SCCLModel(stage1_config)
    train_stage(
        stage1_model,
        encoded_dspa,
        device=device,
        epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        historical_keys=None,
        standard_case_keys=None,
        stage_name="stage1-dspa",
    )

    # Stage 2 starts from all Stage 1 parameters and activates multitask and
    # contrastive objectives.
    stage2_config = SCCLConfig(
        **common_config,
        contrastive_rate_in_training=0.2,
        sc_rate_in_training=0.2,
        w_nps=1.0,
    )
    stage2_model = SCCLModel(stage2_config)
    stage2_model.load_state_dict(stage1_model.state_dict(), strict=True)
    del stage1_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_stage(
        stage2_model,
        encoded_scil,
        device=device,
        epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        historical_keys=historical_key_batch,
        standard_case_keys=standard_key_batch,
        stage_name="stage2-scil",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": stage2_model.state_dict(),
        "config": asdict(stage2_config),
        "diagnosis_labels": list(DIAGNOSIS_LABELS),
        "bio_labels": list(BIO_LABELS),
        "metadata": {
            "toy_data_only": True,
            "clinical_use": False,
            "description": (
                "Privacy-preserving toy checkpoint for interface demonstration; "
                "not a clinical model and not a reproduction benchmark."
            ),
        },
    }
    torch.save(checkpoint, args.output)

    print(f"[saved] {args.output}")
    print(
        "[notice] The checkpoint was trained only on manually authored toy "
        "examples and must not be used for clinical interpretation."
    )


if __name__ == "__main__":
    main()
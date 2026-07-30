"""
Data utilities for the public SCCL toy workflow.

This module supports only the privacy-preserving examples distributed with
the reference implementation. It does not reproduce the clinical database
extraction, institutional data cleaning, or study cohort construction.

Supported inputs
----------------
1. SCIL-style ordered question-answer records with optional character-level
   NPS spans.
2. DSPA-style free-text summaries with a primary diagnostic label.
3. Standard-case free-text records with a primary diagnostic label.
4. Unlabelled SCIL-style records for inference.

Span convention
---------------
Character offsets are zero-based and end-exclusive. For an annotated answer:

    answer[start:end] == span_text

NPS annotations are aligned to tokenizer offsets and converted to BIO labels.
Padding and special tokens receive ``ignore_index``. For a reviewed record
without a positive NPS span, valid text tokens receive the O label. For an
unannotated record, all token labels receive ``ignore_index`` and the
sample-level ``has_nps_labels`` flag is false.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from labels import (
        BIO_TO_ID,
        DIAGNOSIS_LABELS,
        DIAGNOSIS_TO_ID,
        NPS_TYPES,
    )
except ImportError:  # Allows limited direct use from the sccl directory.
    from labels import (
        BIO_TO_ID,
        DIAGNOSIS_LABELS,
        DIAGNOSIS_TO_ID,
        NPS_TYPES,
    )


TensorBatch = Dict[str, torch.Tensor]
CharacterOffset = Tuple[int, int]


def read_json(path: Path | str) -> Dict[str, Any]:
    """Read one UTF-8 JSON object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Read a UTF-8 JSON Lines file and return non-empty records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records were found in {path}.")
    return records


def validate_diagnosis(record: Mapping[str, Any]) -> int:
    """Validate and encode the primary diagnostic label."""
    diagnosis = record.get("diagnosis")
    if diagnosis not in DIAGNOSIS_TO_ID:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} has unsupported diagnosis "
            f"{diagnosis!r}. Expected one of {DIAGNOSIS_LABELS}."
        )
    return DIAGNOSIS_TO_ID[str(diagnosis)]


def validate_qa_pairs(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return a validated, non-empty ordered QA list."""
    qa_pairs = record.get("qa_pairs")
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError(
            f"SCIL record {record.get('id', '<unknown>')} must contain a "
            "non-empty qa_pairs list."
        )

    for index, pair in enumerate(qa_pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(
                f"qa_pairs[{index}] in record {record.get('id')} is not an object."
            )
        if "question" not in pair or "answer" not in pair:
            raise ValueError(
                f"qa_pairs[{index}] in record {record.get('id')} must contain "
                "question and answer."
            )
    return qa_pairs


def format_scil_text(record: Mapping[str, Any]) -> str:
    """
    Format ordered QA pairs as the model input text.

    This function is suitable for unlabelled inference records.
    """
    qa_pairs = validate_qa_pairs(record)
    pieces: List[str] = []

    for pair in qa_pairs:
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        pieces.append(f"[Q] {question}\n[A] {answer}\n")

    return "".join(pieces)


def build_scil_text(
    record: Mapping[str, Any],
) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """
    Format SCIL text and map answer-relative NPS spans to global offsets.

    Returns
    -------
    text
        Concatenated QA text.
    answer_starts
        Global character start of each answer.
    global_spans
        End-exclusive NPS spans in the concatenated text.
    """
    qa_pairs = validate_qa_pairs(record)

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
    has_labels = bool(record.get("has_nps_labels", False))
    source_spans = record.get("nps_spans")
    global_spans: List[Dict[str, Any]] = []

    if not has_labels:
        if source_spans not in (None, []):
            raise ValueError(
                f"Record {record.get('id')} has has_nps_labels=false but "
                "contains NPS spans."
            )
        return text, answer_starts, global_spans

    if source_spans is None:
        raise ValueError(
            f"Record {record.get('id')} is marked as annotated but nps_spans is null."
        )
    if not isinstance(source_spans, list):
        raise ValueError(
            f"nps_spans in record {record.get('id')} must be a list or null."
        )

    for span_index, span in enumerate(source_spans):
        if not isinstance(span, Mapping):
            raise ValueError(
                f"nps_spans[{span_index}] in record {record.get('id')} is invalid."
            )

        qa_index = int(span["qa_index"])
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])

        if not (0 <= qa_index < len(qa_pairs)):
            raise ValueError(
                f"Invalid qa_index={qa_index} in record {record.get('id')}."
            )
        if label not in NPS_TYPES:
            raise ValueError(
                f"Unsupported NPS label {label!r} in record {record.get('id')}."
            )

        answer = str(qa_pairs[qa_index].get("answer", "")).strip()
        if not (0 <= start < end <= len(answer)):
            raise ValueError(
                f"Invalid span [{start}, {end}) in record {record.get('id')}."
            )

        expected_text = span.get("text")
        if expected_text is not None and answer[start:end] != str(expected_text):
            raise ValueError(
                f"Span text mismatch in record {record.get('id')}: "
                f"{answer[start:end]!r} != {expected_text!r}."
            )

        global_spans.append(
            {
                "start": answer_starts[qa_index] + start,
                "end": answer_starts[qa_index] + end,
                "label": label,
            }
        )

    return text, answer_starts, global_spans


def load_fast_tokenizer(
    backbone: str | Path,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer that exposes character offset mappings."""
    tokenizer = AutoTokenizer.from_pretrained(str(backbone), use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required because NPS character spans must be "
            "aligned to token offsets."
        )
    return tokenizer


def tokenize_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    padding: str | bool = "max_length",
) -> TensorBatch:
    """Tokenize one text and retain token-to-character offsets."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=padding,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    if "offset_mapping" not in encoded:
        raise RuntimeError("The tokenizer did not return offset_mapping.")

    result: TensorBatch = {
        "input_ids": encoded["input_ids"].squeeze(0).long(),
        "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        "offset_mapping": encoded["offset_mapping"].squeeze(0).long(),
    }

    if "token_type_ids" in encoded:
        result["token_type_ids"] = encoded["token_type_ids"].squeeze(0).long()
    else:
        result["token_type_ids"] = torch.zeros_like(result["input_ids"])

    return result


def spans_to_bio_ids(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: Sequence[Mapping[str, Any]],
    *,
    has_nps_labels: bool,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Convert end-exclusive character spans to token-level BIO label IDs.

    Overlapping annotations are rejected because one BIO sequence cannot encode
    two different labels on the same token.
    """
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape [sequence_length, 2].")
    if attention_mask.ndim != 1 or attention_mask.shape[0] != offsets.shape[0]:
        raise ValueError(
            "attention_mask must have shape [sequence_length] matching offsets."
        )

    labels = torch.full(
        (offsets.shape[0],),
        fill_value=ignore_index,
        dtype=torch.long,
    )

    if not has_nps_labels:
        return labels

    valid_token = attention_mask.bool() & (offsets[:, 1] > offsets[:, 0])
    labels[valid_token] = BIO_TO_ID["O"]
    occupied = torch.zeros_like(valid_token, dtype=torch.bool)

    for span in sorted(spans, key=lambda item: (int(item["start"]), int(item["end"]))):
        span_start = int(span["start"])
        span_end = int(span["end"])
        nps_type = str(span["label"])

        if nps_type not in NPS_TYPES:
            raise ValueError(f"Unsupported NPS label: {nps_type!r}")
        if span_start < 0 or span_end <= span_start:
            raise ValueError(f"Invalid character span [{span_start}, {span_end}).")

        overlaps = (
            valid_token
            & (offsets[:, 0] < span_end)
            & (offsets[:, 1] > span_start)
        )
        token_indices = overlaps.nonzero(as_tuple=False).view(-1)

        # A valid source span can disappear after truncation.
        if token_indices.numel() == 0:
            continue

        if bool(occupied[token_indices].any()):
            raise ValueError(
                "Overlapping NPS annotations map to the same token. "
                "The public BIO representation supports one label per token."
            )

        labels[token_indices[0]] = BIO_TO_ID[f"B-{nps_type}"]
        if token_indices.numel() > 1:
            labels[token_indices[1:]] = BIO_TO_ID[f"I-{nps_type}"]
        occupied[token_indices] = True

    return labels


def encode_dspa_record(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> TensorBatch:
    """Encode a labelled diagnosis-only DSPA record."""
    text = str(record.get("text", "")).strip()
    if not text:
        raise ValueError(f"DSPA record {record.get('id')} has empty text.")

    encoded = tokenize_text(tokenizer, text, max_length=max_length)
    encoded.pop("offset_mapping")
    encoded["labels"] = torch.tensor(validate_diagnosis(record), dtype=torch.long)
    return encoded


def encode_standard_case_record(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> TensorBatch:
    """Encode a labelled standard-case record."""
    return encode_dspa_record(record, tokenizer, max_length=max_length)


def encode_scil_record(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    ignore_index: int = -100,
) -> TensorBatch:
    """Encode a labelled SCIL record with optional NPS supervision."""
    text, _, global_spans = build_scil_text(record)
    encoded = tokenize_text(tokenizer, text, max_length=max_length)

    has_labels = bool(record.get("has_nps_labels", False))
    encoded["nps_labels"] = spans_to_bio_ids(
        encoded["offset_mapping"],
        encoded["attention_mask"],
        global_spans,
        has_nps_labels=has_labels,
        ignore_index=ignore_index,
    )
    encoded.pop("offset_mapping")
    encoded["labels"] = torch.tensor(validate_diagnosis(record), dtype=torch.long)
    encoded["has_nps_labels"] = torch.tensor(has_labels, dtype=torch.bool)
    return encoded


def encode_inference_record(
    record: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    device: torch.device | None = None,
) -> Tuple[TensorBatch, str, List[CharacterOffset]]:
    """
    Encode one unlabelled SCIL record and preserve offsets for readable output.
    """
    text = format_scil_text(record)
    encoded = tokenize_text(
        tokenizer,
        text,
        max_length=max_length,
        padding=False,
    )

    offsets: List[CharacterOffset] = [
        (int(start), int(end))
        for start, end in encoded.pop("offset_mapping").tolist()
    ]
    query = {
        key: value.unsqueeze(0)
        for key, value in encoded.items()
    }

    if device is not None:
        query = move_batch(query, device)

    return query, text, offsets


def token_span_to_character_span(
    token_span: Mapping[str, Any],
    offsets: Sequence[CharacterOffset],
    text: str,
) -> Dict[str, Any] | None:
    """Convert one end-exclusive token span to a readable character span."""
    token_start = int(token_span["start"])
    token_end = int(token_span["end"])

    if not (0 <= token_start < token_end <= len(offsets)):
        return None

    valid_offsets = [
        offsets[index]
        for index in range(token_start, token_end)
        if offsets[index][1] > offsets[index][0]
    ]
    if not valid_offsets:
        return None

    char_start = min(start for start, _ in valid_offsets)
    char_end = max(end for _, end in valid_offsets)
    if not (0 <= char_start < char_end <= len(text)):
        return None

    label = token_span.get("type", token_span.get("label", ""))
    return {
        "label": str(label),
        "text": text[char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
    }


def stack_batch(
    records: Sequence[Mapping[str, torch.Tensor]],
) -> TensorBatch:
    """Stack tensor fields shared by all records."""
    if not records:
        raise ValueError("Cannot construct an empty batch.")

    common_keys = set(records[0].keys())
    for record in records[1:]:
        common_keys &= set(record.keys())

    if not common_keys:
        raise ValueError("The records have no common tensor fields.")

    batch: TensorBatch = {}
    for key in sorted(common_keys):
        values = [record[key] for record in records]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        batch[key] = torch.stack(values, dim=0)
    return batch


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> TensorBatch:
    """Move every tensor in a batch to the requested device."""
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def iter_batches(
    records: Sequence[Mapping[str, torch.Tensor]],
    *,
    batch_size: int,
    shuffle: bool = False,
) -> Iterable[List[Mapping[str, torch.Tensor]]]:
    """Yield simple in-memory mini-batches for the small toy dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    indices = list(range(len(records)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield [records[index] for index in selected]


def class_representatives(
    records: Sequence[Mapping[str, Any]],
    *,
    include_control: bool = True,
) -> List[Mapping[str, Any]]:
    """Select the first available record for each diagnosis in fixed order."""
    representatives: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        diagnosis = record.get("diagnosis")
        if diagnosis in DIAGNOSIS_TO_ID and diagnosis not in representatives:
            representatives[str(diagnosis)] = record

    ordered_labels = DIAGNOSIS_LABELS
    if not include_control:
        ordered_labels = tuple(
            label for label in DIAGNOSIS_LABELS if label != "CON"
        )

    return [
        representatives[label]
        for label in ordered_labels
        if label in representatives
    ]


def validate_scil_examples(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate SCIL schemas and all answer-relative span offsets."""
    for record in records:
        validate_diagnosis(record)
        build_scil_text(record)


def validate_text_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    allow_control: bool,
) -> None:
    """Validate DSPA or standard-case schemas."""
    for record in records:
        diagnosis_id = validate_diagnosis(record)
        diagnosis = DIAGNOSIS_LABELS[diagnosis_id]
        if not allow_control and diagnosis == "CON":
            raise ValueError(
                f"Standard case {record.get('id')} must not use the CON label."
            )
        text = str(record.get("text", "")).strip()
        if not text:
            raise ValueError(f"Record {record.get('id')} has empty text.")


__all__ = [
    "TensorBatch",
    "CharacterOffset",
    "read_json",
    "read_jsonl",
    "validate_diagnosis",
    "validate_qa_pairs",
    "format_scil_text",
    "build_scil_text",
    "load_fast_tokenizer",
    "tokenize_text",
    "spans_to_bio_ids",
    "encode_dspa_record",
    "encode_standard_case_record",
    "encode_scil_record",
    "encode_inference_record",
    "token_span_to_character_span",
    "stack_batch",
    "move_batch",
    "iter_batches",
    "class_representatives",
    "validate_scil_examples",
    "validate_text_examples",
]

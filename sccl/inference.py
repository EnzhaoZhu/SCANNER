"""
Text-based inference demo for the public SCCL reference implementation.

The script reads an unlabeled SCIL-style JSON record, tokenizes the ordered
question-answer pairs, loads the toy checkpoint produced by train.py, and
returns six-class diagnostic probabilities together with token- and
character-level NPS spans.

The bundled checkpoint and examples are for interface demonstration only.
They are not clinically validated and must not be used for patient care.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

try:  # Supports both ``python inference.py`` and ``python -m sccl.inference``.
    from .config import SCCLConfig
    from .modeling.sccl import SCCLModel
except ImportError:
    from config import SCCLConfig
    from modeling.sccl import SCCLModel

from data import (
    load_fast_tokenizer,
    token_span_to_character_span,
)

DEFAULT_DIAGNOSIS_LABELS: Tuple[str, ...] = (
    "CON",
    "SCZ",
    "BD",
    "DD",
    "ARD",
    "INS",
)
NPS_TYPES: Tuple[str, ...] = (
    "LIA", "DM", "FRA", "DC", "EG", "HW", "EM", "IE", "GO",
    "IS", "PS", "IRB", "IR", "DNS", "CAW", "DEL", "HAL", "NS",
    "PA", "ANX", "PsA", "DIS", "EMA", "MNA", "FI", "SIB", "NSI",
)
DEFAULT_BIO_LABELS: Tuple[str, ...] = ("O",) + tuple(
    tag
    for nps in NPS_TYPES
    for tag in (f"B-{nps}", f"I-{nps}")
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run SCCL inference on a synthetic SCIL-style QA record."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "examples" / "example_input.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=script_dir / "outputs" / "toy_sccl.pt",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help=(
            "Optional local XLNet directory or compatible Hugging Face "
            "identifier. By default, the path stored in the checkpoint is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the JSON prediction.",
    )
    return parser.parse_args()


def read_input(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)

    qa_pairs = record.get("qa_pairs")
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise ValueError("The input record must contain a non-empty qa_pairs list.")
    return record


def build_scil_text(record: Mapping[str, Any]) -> str:
    pieces: List[str] = []
    for pair in record["qa_pairs"]:
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        pieces.append(f"[Q] {question}\n[A] {answer}\n")
    return "".join(pieces)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run train.py first to create the "
            "toy checkpoint."
        )
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "The checkpoint does not contain model_state_dict and config."
        )
    return checkpoint





def tensor_query(
    tokenizer,
    text: str,
    *,
    max_length: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]]]:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"][0].tolist()
    ]

    query: Dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    if "token_type_ids" in encoded:
        query["token_type_ids"] = encoded["token_type_ids"].to(device)
    else:
        query["token_type_ids"] = torch.zeros_like(query["input_ids"])
    return query, offsets




def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    record = read_input(args.input)
    text = build_scil_text(record)
    checkpoint = load_checkpoint(args.checkpoint, device)

    config_dict = dict(checkpoint["config"])
    if args.backbone:
        config_dict["pretrained_model_name_or_path"] = args.backbone
    config = SCCLConfig(**config_dict)

    diagnosis_labels = tuple(
        checkpoint.get("diagnosis_labels", DEFAULT_DIAGNOSIS_LABELS)
    )
    bio_labels = tuple(checkpoint.get("bio_labels", DEFAULT_BIO_LABELS))

    if len(diagnosis_labels) != config.num_diag_classes:
        raise ValueError(
            "Diagnosis label count does not match num_diag_classes."
        )
    if len(bio_labels) != config.num_nps_labels:
        raise ValueError("BIO label count does not match num_nps_labels.")

    backbone = config.pretrained_model_name_or_path
    tokenizer = load_fast_tokenizer(backbone)
    query, offsets = tensor_query(
        tokenizer,
        text,
        max_length=config.max_length,
        device=device,
    )

    model = SCCLModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    id2tag = {index: label for index, label in enumerate(bio_labels)}

    with torch.no_grad():
        output = model(
            query=query,
            positive_sample=None,
            positive_sc=None,
            id2tag=id2tag,
            return_dict=True,
        )

    probabilities = output["diag_probs"][0].detach().cpu().tolist()
    predicted_id = int(output["diag_pred_ids"][0].item())

    diagnostic_probabilities = {
        label: round(float(probability), 6)
        for label, probability in zip(diagnosis_labels, probabilities)
    }

    token_spans = output.get("nps_spans", [[]])[0]
    character_spans: List[Dict[str, Any]] = []
    for token_span in token_spans:
        converted = token_span_to_character_span(token_span, offsets, text)
        if converted is not None:
            character_spans.append(converted)

    result = {
        "input_id": record.get("id"),
        "synthetic_input": bool(record.get("synthetic", False)),
        "predicted_diagnosis": diagnosis_labels[predicted_id],
        "diagnosis_probabilities": diagnostic_probabilities,
        "nps_spans": character_spans,
        "notice": (
            "Prediction generated by a toy-data checkpoint for interface "
            "demonstration only; not for clinical use."
        ),
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()

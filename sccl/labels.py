"""
Central label definitions for the public SCCL reference implementation.

The diagnostic categories and 27 neuropsychiatric symptom (NPS) labels follow
the manuscript and Supplementary Table S2. NPS tagging uses BIO encoding:
one outside tag (O) plus B- and I- tags for each of the 27 NPS categories,
yielding 55 token-level labels.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Primary diagnostic categories.
DIAGNOSIS_LABELS: Tuple[str, ...] = (
    "CON",
    "SCZ",
    "BD",
    "DD",
    "ARD",
    "INS",
)

DIAGNOSIS_NAMES: Dict[str, str] = {
    "CON": "Control",
    "SCZ": "Schizophrenia",
    "BD": "Bipolar disorder",
    "DD": "Depressive disorder",
    "ARD": "Anxiety-related disorder",
    "INS": "Insomnia",
}

DIAGNOSIS_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(DIAGNOSIS_LABELS)
}
ID_TO_DIAGNOSIS: Dict[int, str] = {
    index: label for label, index in DIAGNOSIS_TO_ID.items()
}


# The order follows Supplementary Table S2.
NPS_TYPES: Tuple[str, ...] = (
    "LIA",
    "DM",
    "FRA",
    "DC",
    "EG",
    "HW",
    "EM",
    "IE",
    "GO",
    "IS",
    "PS",
    "IRB",
    "IR",
    "DNS",
    "CAW",
    "DEL",
    "HAL",
    "NS",
    "PA",
    "ANX",
    "PsA",
    "DIS",
    "EMA",
    "MNA",
    "FI",
    "SIB",
    "NSI",
)

NPS_NAMES: Dict[str, str] = {
    "LIA": "Loss of interest or anhedonia",
    "DM": "Depressed mood",
    "FRA": "Fatigue or reduced activity",
    "DC": "Difficulty concentrating",
    "EG": "Excessive guilt",
    "HW": "Hopelessness or worthlessness",
    "EM": "Elevated or expansive mood",
    "IE": "Increased energy or goal-directed activity",
    "GO": "Grandiosity or overconfidence",
    "IS": "Increased or inappropriate sociability",
    "PS": "Pressured speech and/or flight of ideas",
    "IRB": "Impulsivity or risky behaviors",
    "IR": "Irritability or anger",
    "DNS": "Decreased need for sleep",
    "CAW": "Change in appetite or weight",
    "DEL": "Delusions",
    "HAL": "Hallucinations",
    "NS": "Negative symptoms",
    "PA": "Panic attack experiences",
    "ANX": "Psychic anxiety",
    "PsA": "Physical anxiety",
    "DIS": "Difficulty initiating sleep",
    "EMA": "Early-morning awakening",
    "MNA": "Middle-of-night awakenings",
    "FI": "Functional impairment in social or occupational functioning",
    "SIB": "Suicidal ideation or behavior",
    "NSI": "Non-suicidal self-injury",
}


# BIO tag space: O + 27 B-tags + 27 I-tags = 55 labels.
BIO_LABELS: Tuple[str, ...] = ("O",) + tuple(
    tag
    for nps_type in NPS_TYPES
    for tag in (f"B-{nps_type}", f"I-{nps_type}")
)

BIO_TO_ID: Dict[str, int] = {
    label: index for index, label in enumerate(BIO_LABELS)
}
ID_TO_BIO: Dict[int, str] = {
    index: label for label, index in BIO_TO_ID.items()
}


NUM_DIAGNOSIS_CLASSES = len(DIAGNOSIS_LABELS)
NUM_NPS_TYPES = len(NPS_TYPES)
NUM_BIO_LABELS = len(BIO_LABELS)


def validate_label_definitions() -> None:
    """Raise an error if the public label schema is internally inconsistent."""
    if len(set(DIAGNOSIS_LABELS)) != NUM_DIAGNOSIS_CLASSES:
        raise RuntimeError("Duplicate diagnostic labels were detected.")
    if len(set(NPS_TYPES)) != NUM_NPS_TYPES:
        raise RuntimeError("Duplicate NPS labels were detected.")
    if set(NPS_NAMES) != set(NPS_TYPES):
        raise RuntimeError("NPS_NAMES does not match NPS_TYPES.")
    if NUM_NPS_TYPES != 27:
        raise RuntimeError(f"Expected 27 NPS categories, found {NUM_NPS_TYPES}.")
    if NUM_BIO_LABELS != 55:
        raise RuntimeError(f"Expected 55 BIO labels, found {NUM_BIO_LABELS}.")
    if BIO_LABELS[0] != "O":
        raise RuntimeError("The first BIO label must be O.")


validate_label_definitions()


__all__ = [
    "DIAGNOSIS_LABELS",
    "DIAGNOSIS_NAMES",
    "DIAGNOSIS_TO_ID",
    "ID_TO_DIAGNOSIS",
    "NPS_TYPES",
    "NPS_NAMES",
    "BIO_LABELS",
    "BIO_TO_ID",
    "ID_TO_BIO",
    "NUM_DIAGNOSIS_CLASSES",
    "NUM_NPS_TYPES",
    "NUM_BIO_LABELS",
    "validate_label_definitions",
]

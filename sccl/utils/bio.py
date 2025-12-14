from __future__ import annotations
from typing import Any, Dict, List


def bio_tags_to_spans(tags: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a BIO tag sequence into entity spans.

    Span format:
      - type: entity type string (e.g., "ANX")
      - start: inclusive token index
      - end: exclusive token index (Python slicing convention)

    Notes:
      - Robust to malformed sequences: an "I-X" without a preceding "B-X" is treated
        as starting a new entity.
      - This utility intentionally does NOT map token indices back to character offsets.
        That is tokenizer- and application-dependent.
    """
    spans: List[Dict[str, Any]] = []
    cur_type = None
    cur_start = None

    def close(i: int) -> None:
        nonlocal cur_type, cur_start
        if cur_type is not None and cur_start is not None and i > cur_start:
            spans.append({"type": cur_type, "start": cur_start, "end": i})
        cur_type, cur_start = None, None

    for i, tag in enumerate(tags):
        if tag is None or tag == "O":
            close(i)
            continue

        if tag.startswith("B-"):
            close(i)
            cur_type = tag[2:]
            cur_start = i
            continue

        if tag.startswith("I-"):
            t = tag[2:]
            if cur_type is None or cur_type != t:
                close(i)
                cur_type = t
                cur_start = i
            continue

        # Unknown tag format: close and ignore.
        close(i)

    close(len(tags))
    return spans

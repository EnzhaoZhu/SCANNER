# Privacy-preserving toy examples

All records in this directory are manually authored synthetic examples. They are not copied,
translated, paraphrased, or derived from any study participant or clinical record.

## Files

- `toy_scil.jsonl`: SCIL-style question-answer records for six diagnostic categories.
- `toy_dspa.jsonl`: synthetic doctor-summarized psychiatric anamnesis records used to
  illustrate diagnosis-only pre-adaptation.
- `toy_standard_cases.jsonl`: synthetic standard-case records for the five target disorders.
- `example_input.json`: an unlabeled SCIL-style record for inference demonstration.

## Span convention

For `toy_scil.jsonl`, `nps_spans.start` is inclusive and `nps_spans.end` is exclusive.
Offsets are calculated within the `answer` field identified by `qa_index`. The `text` value
must satisfy:

```python
qa_pairs[qa_index]["answer"][start:end] == text
```

`has_nps_labels=false` and `nps_spans=null` indicate that token-level NPS annotations are
unavailable for that sample. An empty list with `has_nps_labels=true` indicates that the
sample was reviewed but no current NPS span was annotated.

Negated, hypothetical, uncertain, or exclusively historical symptoms that are not present at
the index assessment are not annotated as current NPS.

## Intended use

These files illustrate input schemas, preprocessing, partial-label masking, queue construction,
and inference interfaces. They are not a clinical benchmark and cannot reproduce the results
reported in the manuscript.

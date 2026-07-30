# SCCL: Standard-case Contrastive Learning

This repository provides a **public reference implementation of the core SCCL computational workflow** used in our study. It is intended to support inspection and execution of the model architecture and toy-data workflow, while excluding protected clinical datasets, interview question banks, trained study weights, and deployment-specific components.

---

## Overview

SCCL is a multi-task neural architecture designed for psychiatric diagnostic modeling from Structured Clinical Interview for Deep Learning (SCIL).

Core characteristics:

- A **shared language encoder backbone** based on Chinese XLNet
- **Two prediction heads**:
  - Diagnosis classification head for sample-level prediction
  - Neuropsychiatric symptom (NPS) tagging head using token-level BIO labels
- **Two contrastive learning objectives** implemented with MoCo-style queues:
  - Standard-case (SC) contrastive learning
  - Historical-sample (HIS) contrastive learning
- **Partial-label training support** for NPS:
  - Samples without token-level NPS annotations are masked out of the NPS loss
- **Joint inference** of diagnosis probabilities and NPS predictions
- Privacy-preserving **toy examples** illustrating the expected input schemas and executable workflow

This repository provides an architecture-complete and trainable reference implementation for method-level inspection and execution. The included toy examples are not a clinical benchmark and cannot reproduce the performance reported in the manuscript without the protected study datasets.

---

## Repository Structure

```text
sccl/
├── examples/
│   ├── README.md
│   ├── toy_scil.jsonl
│   ├── toy_dspa.jsonl
│   ├── toy_standard_cases.jsonl
│   └── example_input.json
│
├── modeling/
│   ├── backbone_xlnet.py
│   ├── heads.py
│   ├── contrastive.py
│   ├── moco_queue.py
│   ├── losses.py
│   └── sccl.py
│
├── utils/
│   ├── bio.py
│   └── masking.py
│
├── config.py
├── labels.py
├── data.py
├── inference.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Backbone

A shared **XLNet encoder** produces:

- Token-level representations for NPS tagging
- Pooled sequence representations for diagnosis classification and contrastive learning

The implementation supports either a local Chinese XLNet checkpoint, such as `../xlnet_base`, or a compatible Hugging Face checkpoint identifier.

### Prediction Heads

1. **Diagnosis Head**
   - Six-class classification over CON, SCZ, BD, DD, ARD, and INS
   - Trained with cross-entropy or focal loss

2. **NPS Tagging Head**
   - Token-level BIO tagging for 27 NPS categories
   - Optional CRF layer:
     - When enabled, training uses sequence-level negative log-likelihood
     - Decoding enforces valid BIO transitions

### Contrastive Learning

Two contrastive objectives are optimized jointly:

- **Standard-case contrastive learning**
- **Historical-sample contrastive learning**

Both use:

- Momentum-updated key encoders
- Fixed-size feature queues
- Temperature-scaled InfoNCE loss
- Label-aware positive and negative sampling

---

## Partial-label Training for NPS

Token-level NPS annotations are available only for a subset of samples. The implementation supports this setting by:

- Applying sample-level masking
- Applying token-level masking
- Excluding unannotated samples from the NPS loss

---

## Toy Examples

All records in `examples/` were manually authored for demonstration and are not copied, translated, paraphrased, or derived from any study participant or clinical record.

The examples illustrate:

- SCIL-style question-answer inputs
- DSPA-style diagnosis-only text inputs
- Standard-case inputs for contrastive learning
- Character-level NPS spans and BIO conversion
- Partial-label masking
- Text-based inference

They are intended only to demonstrate data schemas and execution of the computational workflow.

---

## Installation

```bash
pip install -r sccl/requirements.txt
```

The default workflow expects the local XLNet checkpoint directory at:

```text
SCANNER/xlnet_base/
```

A compatible checkpoint path can also be supplied through the command-line arguments.

---

## Usage

Run the commands from the `SCANNER` project root.

### Toy Training

```bash
python -m sccl.train
```

The script performs:

1. Diagnosis-only pre-adaptation using the toy DSPA records
2. Multitask SCIL training with NPS supervision and two contrastive queues
3. Saving of a toy checkpoint to `sccl/outputs/toy_sccl.pt`

### Toy Inference

```bash
python -m sccl.inference
```

The script reads `sccl/examples/example_input.json` and returns six-class diagnosis probabilities and predicted NPS spans.

---

## Scope and Limitations

This repository does not include:

- Patient-level SCIL records
- The retrospective DSPA dataset
- The curated clinical standard-case dataset
- Trained weights used for the reported study results
- The deployed clinical interface and institution-specific components

The public implementation supports method-level inspection and execution, but not independent reproduction of the reported clinical performance without access to the protected study data.

---

## License

See the repository `LICENSE` file.

# SCCL: Standard-case Contrastive Learning

This repository provides the **reference implementation of the SCCL model architecture** used in our study.
It is intended to demonstrate the **complete and executable model structure**, while deliberately excluding
private datasets, interview question banks, and deployment-specific components.


---

## Overview

SCCL is a multi-task neural architecture designed for
psychiatric diagnostic modeling from Structured Clinical Interview for Deep Learning (SCIL).

Core characteristics:

- A **shared language encoder backbone** (default: Chinese XLNet base)
- **Two prediction heads**:
  - Diagnosis classification head (sample-level)
  - Neuropsychiatric symptom (NPS) tagging head (token-level BIO tagging)
- **Two contrastive learning objectives** implemented with MoCo-style queues:
  - Standard-case (SC) contrastive learning
  - Historical-sample (HIS) contrastive learning
- **Partial-label training support** for NPS:
  - Samples without token-level NPS annotations are automatically masked out of the NPS loss
- **Joint inference** of diagnosis probabilities and NPS predictions

This repository provides an **architecture-complete and trainable reference**, suitable for inspection,
extension, and method-level reproducibility.

---

## Repository Structure

```
sccl/
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
├── constants.py
├── inference.py
└── train.py
```

---

## Model Architecture

### Backbone

- A shared **XLNet encoder** is used to produce:
  - Token-level representations for NPS tagging
  - Pooled sequence representations for diagnosis classification and contrastive learning
- Default pretrained checkpoint:
  - `hfl/chinese-xlnet-base`

### Prediction Heads

1. **Diagnosis Head**
   - Multi-class classification over major diagnostic categories
   - Trained with cross-entropy or focal loss

2. **NPS Tagging Head**
   - Token-level BIO tagging
   - Optional CRF layer:
     - When enabled, training uses **sequence-level negative log-likelihood**
     - Decoding enforces valid BIO transitions

### Contrastive Learning

Two contrastive objectives are optimized jointly:

- **Standard-case (SC) contrastive learning**
- **Historical-sample (HIS) contrastive learning**

Both are implemented using:
- Momentum-updated key encoders
- Fixed-size feature queues
- Temperature-scaled InfoNCE loss
- Label-aware positive/negative sampling

---

## Partial-label Training for NPS

Token-level NPS annotations are available only for a subset of samples.

The implementation supports this setting by:

- Applying **sample-level masking**
- Applying **token-level masking**
- Ensuring CRF constraints are satisfied even with partially annotated sequences

---

## Examples

### Inference

```bash
python inference.py
```

### Training (Architecture Demo)

```bash
python train.py
```

---

## License

This code is released for academic and research use.

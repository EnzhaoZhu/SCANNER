"""
SCCL: Standard-case Contrastive Clinical Learning (model-architecture only).

This package intentionally exposes ONLY the model architecture components:
- shared encoder backbone
- diagnosis head + NPS head
- two-stage contrastive learning (SC queue + HIS queue)
- partial label masking for NPS
"""

from .config import SCCLConfig
from .modeling.sccl import SCCLModel

__all__ = ["SCCLConfig", "SCCLModel"]

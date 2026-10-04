"""
perception — Member 1 perception layer for Temporal Vision.

Public surface for downstream consumers:
  - PerceptionFrame, DetectedObject  (types)
  - PerceptionConfig                  (configuration)
  - PerceptionPipeline                (main pipeline)
"""

from .config import PerceptionConfig
from .pipeline import PerceptionPipeline
from .types import DetectedObject, PerceptionFrame

__all__ = [
    "PerceptionConfig",
    "PerceptionPipeline",
    "DetectedObject",
    "PerceptionFrame",
]

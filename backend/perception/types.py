"""
perception/types.py
-------------------
Canonical data types shared between perception sub-modules and downstream consumers.
Other team members should only import from here — never from model-specific modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


@dataclass
class DetectedObject:
    """A single detected/tracked object in one video frame."""

    # Stable, human-readable ID that persists across frames, e.g. "person_01"
    id: str

    # Class label as produced by the detector, e.g. "person"
    cls: str

    # Detector confidence in [0, 1]
    confidence: float

    # Axis-aligned bounding box in original image pixel coordinates [x1, y1, x2, y2]
    bbox: List[int]

    # Pixel coordinates of the bounding-box centre [cx, cy]
    centroid: List[int]

    # Relative path to the saved binary mask PNG (None if mask unavailable)
    mask_path: Optional[str] = None

    # Metric depth (metres) at the object centroid; None when Depth Pro is disabled
    depth: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # rename cls -> class for the JSON schema agreed with the team
        d["class"] = d.pop("cls")
        return d


@dataclass
class PerceptionFrame:
    """Standardized per-frame perception result consumed by the reasoning layer."""

    frame_index: int
    timestamp: float  # seconds from video start
    objects: List[DetectedObject] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "objects": [obj.to_dict() for obj in self.objects],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

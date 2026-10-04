"""
perception/config.py
--------------------
All configuration for the perception pipeline lives here.
No constants are hard-coded elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Paths – resolved relative to project root (two levels up from this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
_CHECKPOINTS = _PROJECT_ROOT / "checkpoints"


@dataclass
class PerceptionConfig:
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    input_video: str = ""           # path to input video file
    output_dir: str = "output"      # base output directory

    # ------------------------------------------------------------------
    # Detection – Grounding DINO
    # ------------------------------------------------------------------
    gdino_config: str = str(
        _CHECKPOINTS / "GroundingDINO_SwinB_cfg.py"
    )
    gdino_checkpoint: str = str(
        _CHECKPOINTS / "groundingdino_swinb_cogcoor.pth"
    )
    detection_prompt: str = "person . laptop . table ."
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    # ------------------------------------------------------------------
    # Tracking – SAM 2.1
    # ------------------------------------------------------------------
    sam2_checkpoint: str = str(
        _CHECKPOINTS / "sam2.1_hiera_base_plus.pt"
    )
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    # ------------------------------------------------------------------
    # Depth – Depth Pro (optional)
    # ------------------------------------------------------------------
    enable_depth: bool = False
    depth_checkpoint: str = str(
        _CHECKPOINTS / "depth_pro.pt"
    )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    frame_stride: int = 3           # process every Nth frame
    device: str = "auto"            # "auto" | "cuda" | "mps" | "cpu"

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    debug: bool = False             # write annotated frames

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def output_path(self, *parts: str) -> Path:
        return Path(self.output_dir).joinpath(*parts)

    def masks_dir(self) -> Path:
        return self.output_path("masks")

    def debug_dir(self) -> Path:
        return self.output_path("debug_frames")


# Convenience singleton — callers may mutate fields before passing to pipeline
DEFAULT_CONFIG = PerceptionConfig()

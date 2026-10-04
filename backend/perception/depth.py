"""
perception/depth.py
--------------------
Depth Pro optional adapter.

Depth Pro provides metric monocular depth estimation.
If unavailable, all depth fields are returned as None and a single warning is logged.

Apple ml-depth-pro repo: https://github.com/apple/ml-depth-pro
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
_DEPTH_WARNED = False  # log unavailability only once


class DepthProAdapter:
    """
    Thin adapter for Apple Depth Pro.

    Usage
    -----
    adapter = DepthProAdapter(config)
    depth_map, focal = adapter.infer(frame_bgr)   # None, None if unavailable
    depth_at = adapter.depth_at(depth_map, cx, cy)
    depth_mean = adapter.depth_mean(depth_map, mask)
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._transform = None
        self._available: Optional[bool] = None  # None = not yet checked

    # ------------------------------------------------------------------
    def _check_and_load(self) -> bool:
        """Try to load Depth Pro. Returns True if successful."""
        global _DEPTH_WARNED

        if self._available is not None:
            return self._available

        ckpt = self.config.depth_checkpoint
        if not Path(ckpt).exists():
            if not _DEPTH_WARNED:
                log.warning(
                    "Depth Pro checkpoint not found at %s. "
                    "Depth will be null. "
                    "Place depth_pro.pt in checkpoints/ to enable depth.",
                    ckpt,
                )
                _DEPTH_WARNED = True
            self._available = False
            return False

        try:
            import depth_pro
            self._model, self._transform = depth_pro.create_model_and_transforms(
                checkpoint_uri=ckpt,
            )
            self._model.eval()
            self._available = True
            log.info("Depth Pro loaded from %s", ckpt)
        except ImportError:
            if not _DEPTH_WARNED:
                log.warning(
                    "depth_pro package not installed. Depth will be null. "
                    "Install with: pip install git+https://github.com/apple/ml-depth-pro.git"
                )
                _DEPTH_WARNED = True
            self._available = False
        except Exception as exc:
            if not _DEPTH_WARNED:
                log.warning("Depth Pro failed to load (%s). Depth will be null.", exc)
                _DEPTH_WARNED = True
            self._available = False

        return self._available

    # ------------------------------------------------------------------
    def infer(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Run Depth Pro on a BGR frame.

        Returns
        -------
        depth_map : float32 numpy array (H, W) in metres, or None
        focal_length : predicted focal length in pixels, or None
        """
        if not self._check_and_load():
            return None, None

        try:
            import torch
            from PIL import Image

            rgb = frame_bgr[:, :, ::-1].copy()
            pil_img = Image.fromarray(rgb)
            image_data = self._transform(pil_img)

            with torch.no_grad():
                prediction = self._model.infer(image_data, f_px=None)

            depth = prediction["depth"].cpu().numpy().astype(np.float32)
            focal = float(prediction.get("focallength_px", 0) or 0) or None
            return depth, focal

        except Exception as exc:
            log.warning("Depth Pro inference failed: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    @staticmethod
    def depth_at(
        depth_map: Optional[np.ndarray],
        cx: int,
        cy: int,
    ) -> Optional[float]:
        """Return depth value (metres) at pixel (cx, cy), or None."""
        if depth_map is None:
            return None
        h, w = depth_map.shape[:2]
        cy_ = max(0, min(cy, h - 1))
        cx_ = max(0, min(cx, w - 1))
        val = float(depth_map[cy_, cx_])
        return round(val, 3)

    @staticmethod
    def depth_mean(
        depth_map: Optional[np.ndarray],
        mask: Optional[np.ndarray],
    ) -> Optional[float]:
        """Return mean depth (metres) inside a boolean mask, or None."""
        if depth_map is None or mask is None:
            return None
        vals = depth_map[mask > 0]
        if len(vals) == 0:
            return None
        return round(float(vals.mean()), 3)

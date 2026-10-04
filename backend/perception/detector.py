"""
perception/detector.py
-----------------------
Grounding DINO-B adapter.

Input:  RGB numpy frame + text prompt
Output: list[DetectedObject]  (no raw model tensors exposed)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import PerceptionConfig
from .device import select_device
from .types import DetectedObject

log = logging.getLogger(__name__)


def _gdino_transform():
    """Return the standard Grounding DINO image transform."""
    import groundingdino.datasets.transforms as T
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class GroundingDINODetector:
    """
    Thin adapter around Grounding DINO.
    Load once, call detect() per frame.
    """

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.device = select_device(config.device)
        self._model = None
        self._transform = None

    # ------------------------------------------------------------------
    # Lazy load — only instantiated when first detection is requested
    # ------------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return

        cfg_path = self.config.gdino_config
        ckpt_path = self.config.gdino_checkpoint

        if not Path(cfg_path).exists():
            raise FileNotFoundError(
                f"Grounding DINO config not found: {cfg_path}\n"
                "Download GroundingDINO_SwinB_cfg.py and place it in checkpoints/"
            )
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Grounding DINO checkpoint not found: {ckpt_path}\n"
                "Download groundingdino_swinb_cogcoor.pth and place it in checkpoints/"
            )

        log.info("Loading Grounding DINO from %s", ckpt_path)
        from groundingdino.util.inference import load_model
        self._model = load_model(cfg_path, ckpt_path, device=self.device)
        self._transform = _gdino_transform()
        log.info("Grounding DINO loaded on device=%s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        frame_bgr: np.ndarray,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[DetectedObject]:
        """
        Run detection on a single BGR frame (as returned by OpenCV).

        Returns a list of DetectedObject with temporary IDs ("cls_N").
        The pipeline assigns stable IDs later.
        """
        self._load()

        import torch
        from PIL import Image
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert

        prompt = prompt or self.config.detection_prompt
        box_thr = box_threshold if box_threshold is not None else self.config.box_threshold
        txt_thr = text_threshold if text_threshold is not None else self.config.text_threshold

        # BGR → RGB PIL for the transform
        rgb = frame_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        h, w = frame_bgr.shape[:2]

        img_transformed, _ = self._transform(pil_img, None)

        boxes, logits, phrases = predict(
            model=self._model,
            image=img_transformed,
            caption=prompt,
            box_threshold=box_thr,
            text_threshold=txt_thr,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            log.debug("No detections for prompt=%r", prompt)
            return []

        # boxes are normalised [cx, cy, w, h]; convert to pixel xyxy
        boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h
        boxes_xyxy = boxes_xyxy.int().tolist()

        # Count per class to generate temporary IDs like person_01
        cls_counter: dict[str, int] = {}
        results: List[DetectedObject] = []

        for bbox, logit, phrase in zip(boxes_xyxy, logits.tolist(), phrases):
            cls_counter[phrase] = cls_counter.get(phrase, 0) + 1
            tmp_id = f"{phrase}_{cls_counter[phrase]:02d}"

            x1, y1, x2, y2 = bbox
            # Clamp to frame bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            results.append(DetectedObject(
                id=tmp_id,
                cls=phrase,
                confidence=round(float(logit), 4),
                bbox=[x1, y1, x2, y2],
                centroid=[cx, cy],
            ))

        log.debug("Detected %d objects: %s", len(results), [r.id for r in results])
        return results

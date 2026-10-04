"""
perception/tracker.py
---------------------
SAM 2.1 video tracking adapter.

Flow:
  1. Detector finds objects in a key frame.
  2. Tracker initialises SAM 2.1 with bounding-box prompts.
  3. SAM 2.1 propagates masks through the entire frame sequence.
  4. Per-frame mask data is returned as DetectedObject lists.

SAM 2.1 requires frames as JPEG files.
This module extracts them to a temp directory if needed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import PerceptionConfig
from .device import select_device
from .types import DetectedObject, PerceptionFrame

log = logging.getLogger(__name__)

# SAM 2.1 needs this env variable for Apple MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class SAM2Tracker:
    """
    Wraps SAM 2.1 video predictor for multi-object mask propagation.
    Object IDs are deterministic (sorted by class then bbox x1).
    """

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.device = select_device(config.device)
        self._predictor = None

    # ------------------------------------------------------------------
    def _load(self):
        if self._predictor is not None:
            return

        ckpt = self.config.sam2_checkpoint
        cfg = self.config.sam2_config

        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"SAM 2.1 checkpoint not found: {ckpt}\n"
                "Download sam2.1_hiera_base_plus.pt and place it in checkpoints/"
            )

        log.info("Loading SAM 2.1 from %s", ckpt)
        from sam2.build_sam import build_sam2_video_predictor
        self._predictor = build_sam2_video_predictor(cfg, ckpt, device=self.device)
        log.info("SAM 2.1 loaded on device=%s", self.device)

    # ------------------------------------------------------------------
    # Stable ID assignment
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_stable_ids(detections: List[DetectedObject]) -> Dict[int, DetectedObject]:
        """
        Given detections from the seed frame, assign deterministic stable IDs.

        Sort by (class, x1) so IDs are reproducible regardless of detector order.
        Returns mapping: sam_obj_id (int) -> DetectedObject with stable id.
        """
        sorted_dets = sorted(detections, key=lambda d: (d.cls, d.bbox[0]))
        cls_counter: Dict[str, int] = {}
        id_map: Dict[int, DetectedObject] = {}

        for sam_id, det in enumerate(sorted_dets, start=1):
            cls_counter[det.cls] = cls_counter.get(det.cls, 0) + 1
            stable_id = f"{det.cls}_{cls_counter[det.cls]:02d}"
            det.id = stable_id
            id_map[sam_id] = det

        return id_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_video(
        self,
        frame_dir: Path,
        seed_frame_idx: int,
        detections: List[DetectedObject],
        output_mask_dir: Path,
        frame_indices: List[int],
        fps: float,
    ) -> List[PerceptionFrame]:
        """
        Run SAM 2.1 on a sequence of JPEG frames.

        Parameters
        ----------
        frame_dir      : directory containing %05d.jpg frames (0-indexed)
        seed_frame_idx : frame index where we initialise from detections
        detections     : detector output for the seed frame
        output_mask_dir: where to save binary mask PNGs
        frame_indices  : sorted list of all frame indices being processed
        fps            : video frames-per-second (for timestamp calculation)

        Returns
        -------
        List of PerceptionFrame, one per frame in frame_indices.
        """
        import torch

        self._load()

        if not detections:
            log.warning("No detections on seed frame %d; skipping tracking.", seed_frame_idx)
            return [
                PerceptionFrame(frame_index=idx, timestamp=round(idx / fps, 4))
                for idx in frame_indices
            ]

        id_map = self._assign_stable_ids(detections)
        log.info("Initialising SAM 2.1 with %d objects: %s",
                 len(id_map), {v.id for v in id_map.values()})

        output_mask_dir.mkdir(parents=True, exist_ok=True)

        inference_state = self._predictor.init_state(video_path=str(frame_dir))
        self._predictor.reset_state(inference_state)

        sorted_indices = sorted(frame_indices)
        # Explicit mapping: SAM sequence index -> original video frame index
        # When frame_dir has files named 00000.jpg, 00003.jpg, ..., SAM loads them in sorted
        # order and assigns contiguous sequence indices 0, 1, 2, ..., len(sorted_indices)-1.
        sam_to_orig: Dict[int, int] = {
            sam_seq_idx: orig_idx for sam_seq_idx, orig_idx in enumerate(sorted_indices)
        }
        orig_to_sam: Dict[int, int] = {
            orig_idx: sam_seq_idx for sam_seq_idx, orig_idx in enumerate(sorted_indices)
        }

        # SAM sequence index for the seed frame
        sam_seed_idx = orig_to_sam.get(seed_frame_idx, 0)

        # Add bounding-box prompts for each detected object on the seed frame
        for sam_id, det in id_map.items():
            box = np.array(det.bbox, dtype=np.float32)  # [x1,y1,x2,y2]
            _, obj_ids, mask_logits = self._predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=sam_seed_idx,
                obj_id=sam_id,
                box=box,
            )
            log.debug("Initialised obj_id=%d (%s) on seed frame %d (sam_seq_idx=%d)",
                      sam_id, det.id, seed_frame_idx, sam_seed_idx)

        # Propagate through entire video
        frame_results: Dict[int, List[DetectedObject]] = {idx: [] for idx in sorted_indices}

        for sam_frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(inference_state):
            orig_frame_idx = sam_to_orig.get(sam_frame_idx)
            if orig_frame_idx is None:
                continue

            masks = (mask_logits > 0.0).cpu().numpy()  # shape: (N, 1, H, W)

            for i, sam_id in enumerate(obj_ids):
                if sam_id not in id_map:
                    continue

                ref_det = id_map[sam_id]
                mask = masks[i, 0]  # (H, W) bool

                if not mask.any():
                    # Object not visible in this frame
                    log.debug("Object %s absent in frame %d (orig_frame=%d)",
                              ref_det.id, sam_frame_idx, orig_frame_idx)
                    continue

                # Compute bbox and centroid from mask
                ys, xs = np.where(mask)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Save mask PNG using original frame index in filename/path
                frame_mask_dir = output_mask_dir / f"frame_{orig_frame_idx:05d}"
                frame_mask_dir.mkdir(parents=True, exist_ok=True)
                mask_rel = f"masks/frame_{orig_frame_idx:05d}/{ref_det.id}.png"
                mask_abs = output_mask_dir.parent / mask_rel
                mask_abs.parent.mkdir(parents=True, exist_ok=True)
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(str(mask_abs))

                frame_results[orig_frame_idx].append(DetectedObject(
                    id=ref_det.id,
                    cls=ref_det.cls,
                    confidence=ref_det.confidence,  # carry detector confidence
                    bbox=[x1, y1, x2, y2],
                    centroid=[cx, cy],
                    mask_path=mask_rel,
                ))

        # Build PerceptionFrame list
        perception_frames = []
        for idx in sorted_indices:
            perception_frames.append(PerceptionFrame(
                frame_index=idx,
                timestamp=round(idx / fps, 4),
                objects=frame_results.get(idx, []),
            ))

        return perception_frames


# ------------------------------------------------------------------
# Helper: extract video frames to a temp directory for SAM 2.1
# ------------------------------------------------------------------

def extract_frames(
    video_path: str,
    stride: int,
    out_dir: Path,
) -> Tuple[List[int], float, int, int]:
    """
    Extract every `stride`-th frame from a video as JPEG files.

    Returns (frame_indices, fps, width, height).
    Files are named %05d.jpg (0-indexed original frame number).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %dx%d @ %.2f fps, %d total frames", width, height, fps, total)

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = []
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % stride == 0:
            jpg_path = out_dir / f"{frame_no:05d}.jpg"
            cv2.imwrite(str(jpg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_indices.append(frame_no)
        frame_no += 1

    cap.release()
    log.info("Extracted %d frames (stride=%d)", len(frame_indices), stride)
    return frame_indices, fps, width, height

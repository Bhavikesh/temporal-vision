"""
perception/pipeline.py
-----------------------
Orchestrates the full perception pipeline:

  video → detect (GDino) → track (SAM 2.1) → depth (DepthPro) → output JSON

Usage
-----
from backend.perception.pipeline import PerceptionPipeline
from backend.perception.config import PerceptionConfig

cfg = PerceptionConfig(input_video="my_video.mp4", output_dir="output")
pipeline = PerceptionPipeline(cfg)
results = pipeline.run()
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import PerceptionConfig
from .debug import render_frame, write_debug_video
from .depth import DepthProAdapter
from .detector import GroundingDINODetector
from .tracker import SAM2Tracker, extract_frames
from .types import DetectedObject, PerceptionFrame

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    End-to-end perception pipeline.

    Parameters
    ----------
    config : PerceptionConfig — all knobs live here.
    """

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.detector = GroundingDINODetector(config)
        self.tracker = SAM2Tracker(config)
        self.depth_adapter = DepthProAdapter(config) if config.enable_depth else None

    # ------------------------------------------------------------------
    def run(self) -> List[PerceptionFrame]:
        """
        Process the configured video.

        Returns
        -------
        List of PerceptionFrame (one per sampled frame).
        Also writes:
          - output/perception_output.json
          - output/masks/frame_XXXXX/<id>.png   (binary masks)
          - output/debug_frames/frame_XXXXX.jpg  (if debug=True)
          - output/debug_video.mp4               (if debug=True)
        """
        cfg = self.config
        video_path = cfg.input_video

        if not video_path:
            raise ValueError("PerceptionConfig.input_video must be set.")
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = out_dir / "masks"
        debug_dir = out_dir / "debug_frames"

        # ----------------------------------------------------------
        # 1. Extract frames to a temp directory (SAM 2.1 needs JPEGs)
        # ----------------------------------------------------------
        frame_tmp = Path(tempfile.mkdtemp(prefix="temporal_vision_frames_"))
        log.info("Extracting frames to %s", frame_tmp)

        try:
            frame_indices, fps, width, height = extract_frames(
                video_path, cfg.frame_stride, frame_tmp
            )

            if not frame_indices:
                raise RuntimeError("No frames extracted from video.")

            # ----------------------------------------------------------
            # 2. Detect objects in the seed frame (first sampled frame)
            # ----------------------------------------------------------
            seed_idx = frame_indices[0]
            seed_jpg = frame_tmp / f"{seed_idx:05d}.jpg"
            seed_bgr = cv2.imread(str(seed_jpg))
            if seed_bgr is None:
                raise RuntimeError(f"Could not read seed frame: {seed_jpg}")

            log.info("Running detection on seed frame %d", seed_idx)
            detections = self.detector.detect(seed_bgr, cfg.detection_prompt)
            log.info("Detected %d objects: %s", len(detections), [d.id for d in detections])

            # ----------------------------------------------------------
            # 3. SAM 2.1 tracking
            # ----------------------------------------------------------
            log.info("Starting SAM 2.1 tracking over %d frames", len(frame_indices))
            perception_frames = self.tracker.track_video(
                frame_dir=frame_tmp,
                seed_frame_idx=seed_idx,
                detections=detections,
                output_mask_dir=masks_dir,
                frame_indices=frame_indices,
                fps=fps,
            )

            # ----------------------------------------------------------
            # 4. Depth Pro (optional)
            # ----------------------------------------------------------
            if cfg.enable_depth and self.depth_adapter is not None:
                log.info("Running Depth Pro on %d frames", len(frame_indices))
                for pframe in perception_frames:
                    jpg = frame_tmp / f"{pframe.frame_index:05d}.jpg"
                    bgr = cv2.imread(str(jpg))
                    if bgr is None:
                        continue
                    depth_map, _ = self.depth_adapter.infer(bgr)
                    for obj in pframe.objects:
                        obj.depth = self.depth_adapter.depth_at(
                            depth_map, obj.centroid[0], obj.centroid[1]
                        )

            # ----------------------------------------------------------
            # 5. Debug visualisation
            # ----------------------------------------------------------
            if cfg.debug:
                log.info("Rendering debug frames")
                for pframe in perception_frames:
                    jpg = frame_tmp / f"{pframe.frame_index:05d}.jpg"
                    bgr = cv2.imread(str(jpg))
                    if bgr is None:
                        continue
                    render_frame(bgr, pframe, show_mask=True, output_dir=debug_dir)

                debug_video = out_dir / "debug_video.mp4"
                write_debug_video(debug_dir, debug_video, fps=fps / cfg.frame_stride)
                log.info("Debug video: %s", debug_video)

        finally:
            shutil.rmtree(frame_tmp, ignore_errors=True)

        # ----------------------------------------------------------
        # 6. Save perception output JSON
        # ----------------------------------------------------------
        output_json = out_dir / "perception_output.json"
        payload = [pf.to_dict() for pf in perception_frames]
        output_json.write_text(json.dumps(payload, indent=2))
        log.info("Perception output saved: %s  (%d frames)", output_json, len(perception_frames))

        return perception_frames

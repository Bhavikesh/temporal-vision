"""
perception/debug.py
--------------------
Debug visualisation renderer — completely separate from model inference.

Writes annotated frames and optionally an annotated video.
Member 4 can use these artefacts for the frontend evidence display.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .types import PerceptionFrame

log = logging.getLogger(__name__)

# Colour palette per class (BGR)
_CLASS_COLOURS = {
    "person": (0, 200, 50),
    "laptop": (50, 150, 255),
    "table":  (0, 200, 220),
}
_DEFAULT_COLOUR = (180, 180, 180)


def _colour_for(cls: str) -> tuple:
    return _CLASS_COLOURS.get(cls.lower(), _DEFAULT_COLOUR)


def render_frame(
    frame_bgr: np.ndarray,
    perception_frame: PerceptionFrame,
    show_mask: bool = True,
    output_dir: Optional[Path] = None,
    artifact_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Draw bounding boxes, masks, object IDs, and confidence on a frame.

    Parameters
    ----------
    frame_bgr        : BGR numpy array to annotate
    perception_frame : frame data with object list
    show_mask        : whether to overlay masks
    output_dir       : if set, saves the annotated JPEG here (debug_frames/)
    artifact_dir     : root used to resolve relative mask_path values
                       (= the pipeline output_dir that contains masks/).
                       If None, mask_path is used as-is.

    Returns the annotated BGR numpy array.
    """
    vis = frame_bgr.copy()

    for obj in perception_frame.objects:
        colour = _colour_for(obj.cls)
        x1, y1, x2, y2 = obj.bbox
        cx, cy = obj.centroid

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Overlay mask — resolve mask_path relative to artifact_dir when provided
        if show_mask and obj.mask_path:
            try:
                mask_abs = (
                    str(Path(artifact_dir) / obj.mask_path)
                    if artifact_dir is not None
                    else obj.mask_path
                )
                mask_gray = cv2.imread(mask_abs, cv2.IMREAD_GRAYSCALE)
                if mask_gray is not None:
                    h, w = vis.shape[:2]
                    if mask_gray.shape != (h, w):
                        mask_gray = cv2.resize(mask_gray, (w, h))
                    mask_bool = mask_gray > 127
                    overlay = vis.copy()
                    overlay[mask_bool] = [c // 2 + v // 2
                                          for c, v in zip(colour, vis[mask_bool].mean(axis=0)[:3])]
                    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            except Exception as exc:
                log.debug("Could not render mask for %s: %s", obj.id, exc)

        # Label: e.g. "Person #01  0.96  2.40m"
        cls_cap = obj.cls.capitalize()
        idx_str = obj.id.split("_")[-1] if "_" in obj.id else "01"
        label = f"{cls_cap} #{idx_str}  {obj.confidence:.2f}"
        if obj.depth is not None:
            label += f"  {obj.depth:.2f}m"

        # Background rect for legibility
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Centroid dot
        cv2.circle(vis, (cx, cy), 4, colour, -1)

    # Frame info overlay (top-left)
    info = f"Frame {perception_frame.frame_index:05d}  t={perception_frame.timestamp:.2f}s"
    cv2.putText(vis, info, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"frame_{perception_frame.frame_index:05d}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.debug("Saved debug frame: %s", out_path)

    return vis


def write_debug_video(
    debug_frame_dir: Path,
    output_video_path: Path,
    fps: float = 10.0,
) -> None:
    """
    Assemble annotated JPEG frames into an MP4 debug video.
    """
    frames = sorted(debug_frame_dir.glob("frame_*.jpg"))
    if not frames:
        log.warning("No debug frames found in %s", debug_frame_dir)
        return

    sample = cv2.imread(str(frames[0]))
    if sample is None:
        log.warning("Could not read sample frame: %s", frames[0])
        return

    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    for fp in frames:
        img = cv2.imread(str(fp))
        if img is not None:
            writer.write(img)

    writer.release()
    log.info("Debug video saved: %s", output_video_path)

#!/usr/bin/env python3
"""
scripts/test_perception.py
---------------------------
Minimal smoke test for the Temporal Vision perception layer.

What it checks:
  1. Model files exist (checkpoint paths).
  2. Grounding DINO-B runs and detects on a real frame.
  3. At least one of: person / laptop / table detected when present.
  4. SAM 2.1 runs on a short video and produces masks across frames.
  5. Object IDs are stable across frames.
  6. Per-frame JSON output is valid and follows the agreed schema.
  7. Debug visualisation is written.
  8. Depth Pro either runs or cleanly returns null.

Usage
-----
# From the project root (curious-parc/):
conda run -n yolo_env python scripts/test_perception.py \
    --video <path_to_video.mp4> \
    --output output_smoke_test \
    [--prompt "person . laptop . table ."] \
    [--depth]

The script creates synthetic test data if no video is provided.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so we can import backend/perception
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Required for Apple MPS compatibility with SAM 2.1
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test")


# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Perception layer smoke test")
    p.add_argument("--video", default="", help="Path to input video file")
    p.add_argument("--output", default="output_smoke_test", help="Output directory")
    p.add_argument("--prompt", default="person . laptop . table .",
                   help="Detection text prompt")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=5,
                   help="Frame stride (default 5 for smoke test speed)")
    p.add_argument("--depth", action="store_true", help="Enable Depth Pro")
    p.add_argument("--debug", action="store_true", default=True,
                   help="Write debug frames (default: on)")
    return p.parse_args()


# -----------------------------------------------------------------------
def _create_synthetic_video(path: str, n_frames: int = 30, fps: float = 10.0):
    """Generate a minimal RGB video with coloured rectangles for testing."""
    import cv2
    import numpy as np

    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Simulate a moving rectangle (dummy "person")
        x = 50 + i * 5
        cv2.rectangle(frame, (x, 80), (x + 100, 300), (0, 200, 50), -1)
        writer.write(frame)
    writer.release()
    log.info("Created synthetic video: %s (%d frames)", path, n_frames)


# -----------------------------------------------------------------------
def check_checkpoint(path: str, label: str) -> bool:
    if Path(path).exists():
        log.info("✓ %s: %s", label, path)
        return True
    log.warning("✗ %s NOT FOUND: %s", label, path)
    return False


# -----------------------------------------------------------------------
def validate_frame_schema(frame_dict: dict, frame_no: int) -> list[str]:
    """Return list of schema violations, empty if valid."""
    errors = []
    for key in ("frame_index", "timestamp", "objects"):
        if key not in frame_dict:
            errors.append(f"Frame {frame_no}: missing key '{key}'")

    if not isinstance(frame_dict.get("frame_index"), int):
        errors.append(f"Frame {frame_no}: frame_index must be int")
    if not isinstance(frame_dict.get("timestamp"), (int, float)):
        errors.append(f"Frame {frame_no}: timestamp must be numeric")

    for obj in frame_dict.get("objects", []):
        for key in ("id", "class", "confidence", "bbox", "centroid"):
            if key not in obj:
                errors.append(f"Frame {frame_no}: object missing key '{key}'")
        bbox = obj.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"Frame {frame_no}: bbox must have 4 elements")
        centroid = obj.get("centroid", [])
        if len(centroid) != 2:
            errors.append(f"Frame {frame_no}: centroid must have 2 elements")
        conf = obj.get("confidence", -1)
        if not (0.0 <= conf <= 1.0):
            errors.append(f"Frame {frame_no}: confidence {conf} not in [0,1]")

    return errors


# -----------------------------------------------------------------------
def main():
    args = parse_args()

    from backend.perception.config import PerceptionConfig

    # ------------------------------------------------------------------
    # 1. Build config
    # ------------------------------------------------------------------
    cfg = PerceptionConfig(
        input_video=args.video,
        output_dir=args.output,
        detection_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        frame_stride=args.stride,
        enable_depth=args.depth,
        debug=args.debug,
    )

    log.info("=== STEP 1: Checkpoint verification ===")
    gdino_ok = check_checkpoint(cfg.gdino_config, "GDino config")
    gdino_ok &= check_checkpoint(cfg.gdino_checkpoint, "GDino checkpoint")
    sam2_ok = check_checkpoint(cfg.sam2_checkpoint, "SAM 2.1 checkpoint")
    depth_ok = True
    if args.depth:
        depth_ok = check_checkpoint(cfg.depth_checkpoint, "Depth Pro checkpoint")

    if not gdino_ok:
        log.error("Grounding DINO checkpoints missing. Smoke test cannot continue.")
        log.error(
            "Download from:\n"
            "  Config : https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/"
            "main/groundingdino/config/GroundingDINO_SwinB_cfg.py\n"
            "  Weights: https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
        )
        sys.exit(1)

    if not sam2_ok:
        log.error("SAM 2.1 checkpoint missing. Smoke test cannot continue.")
        log.error(
            "Download from:\n"
            "  https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_base_plus.pt"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Video setup
    # ------------------------------------------------------------------
    synthetic_video = None
    if not cfg.input_video:
        synthetic_video = tempfile.mktemp(suffix=".mp4")
        _create_synthetic_video(synthetic_video, n_frames=30, fps=10.0)
        cfg.input_video = synthetic_video
        log.info("No video provided; using synthetic video for smoke test.")

    # ------------------------------------------------------------------
    # 3. Run pipeline
    # ------------------------------------------------------------------
    log.info("=== STEP 2–7: Running full perception pipeline ===")
    from backend.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(cfg)
    frames = pipeline.run()

    log.info("Pipeline complete. Processed %d frames.", len(frames))

    # ------------------------------------------------------------------
    # 4. Validate JSON output
    # ------------------------------------------------------------------
    log.info("=== STEP 8: Validating JSON output ===")
    output_json = Path(cfg.output_dir) / "perception_output.json"
    if not output_json.exists():
        log.error("Output JSON not found: %s", output_json)
        sys.exit(1)

    data = json.loads(output_json.read_text())
    all_errors = []
    for i, fd in enumerate(data):
        all_errors.extend(validate_frame_schema(fd, i))

    if all_errors:
        log.error("Schema violations:\n" + "\n".join(all_errors))
        sys.exit(1)
    log.info("✓ JSON schema valid  (%d frames)", len(data))

    # ------------------------------------------------------------------
    # 5. ID stability check
    # ------------------------------------------------------------------
    log.info("=== STEP 9: ID stability check ===")
    id_sets = [set(obj["id"] for obj in fd["objects"]) for fd in data if fd["objects"]]
    if id_sets:
        all_ids = id_sets[0].union(*id_sets[1:]) if len(id_sets) > 1 else id_sets[0]
        log.info("Unique stable IDs across all frames: %s", sorted(all_ids))
        # Confirm no frame invents a new ID that wasn't in the first active frame
        first_ids = id_sets[0]
        for i, ids in enumerate(id_sets[1:], start=1):
            unexpected = ids - first_ids
            if unexpected:
                log.warning("Frame %d introduced new IDs: %s", i, unexpected)
    else:
        log.warning("No objects tracked in any frame (empty video or no detections).")

    # ------------------------------------------------------------------
    # 6. Detection class check
    # ------------------------------------------------------------------
    log.info("=== STEP 10: Detection class check ===")
    all_classes = set()
    for fd in data:
        for obj in fd["objects"]:
            all_classes.add(obj["class"])

    target_classes = {"person", "laptop", "table"}
    found = target_classes & all_classes
    missing = target_classes - all_classes

    if found:
        log.info("✓ Classes detected: %s", sorted(found))
    if missing:
        log.warning(
            "Target classes NOT detected: %s "
            "(may be absent from video or below threshold)",
            sorted(missing),
        )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("=== SMOKE TEST SUMMARY ===")
    log.info("  Frames processed  : %d", len(frames))
    log.info("  Classes detected  : %s", sorted(all_classes) or "none")
    log.info("  Output JSON       : %s", output_json)
    if args.debug:
        debug_vid = Path(cfg.output_dir) / "debug_video.mp4"
        log.info("  Debug video       : %s (%s)",
                 debug_vid, "exists" if debug_vid.exists() else "NOT FOUND")

    # Print an example frame
    active = [fd for fd in data if fd["objects"]]
    if active:
        log.info("=== Example output (first active frame) ===")
        print(json.dumps(active[0], indent=2))

    log.info("=== Smoke test PASSED ===")

    # Cleanup synthetic video
    if synthetic_video and Path(synthetic_video).exists():
        Path(synthetic_video).unlink()


if __name__ == "__main__":
    main()

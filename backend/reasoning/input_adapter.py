"""
reasoning/input_adapter.py
--------------------------
Perception output adapter and contract validator (Member 2).

Consumes and validates raw JSON perception observations according to the agreed
team contract in docs/DATA_MODEL.md:
  - Validates frame-level fields (frame_index, timestamp, objects)
  - Validates object-level fields (id, class, confidence, bbox, centroid, mask_path, depth)
  - Preserves nullable depth without hallucination
  - Constructs clean ObjectObservation models for downstream reasoning
  - Compatible with ObjectStateManager.update()

Strictly decoupled from Member 1 perception implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .state import ObjectObservation


class PerceptionValidationError(ValueError):
    """Raised when a perception frame or object fails schema validation."""
    pass


@dataclass
class ParsedObject:
    """
    Validated perception object containing identity, class, and an ObjectObservation.
    Exposes attributes compatible with ObjectStateManager.
    """

    id: str
    cls: str
    observation: ObjectObservation

    @property
    def confidence(self) -> float:
        return self.observation.confidence

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.observation.bbox

    @property
    def centroid(self) -> Tuple[int, int]:
        return self.observation.centroid

    @property
    def mask_path(self) -> Optional[str]:
        return self.observation.mask_path

    @property
    def depth(self) -> Optional[float]:
        return self.observation.depth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.cls,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "mask_path": self.mask_path,
            "depth": round(self.depth, 4) if self.depth is not None else None,
        }


@dataclass
class ParsedFrame:
    """
    Validated perception frame holding metadata and parsed objects.
    Directly consumable by ObjectStateManager.update().
    """

    frame_index: int
    timestamp: float
    objects: List[ParsedObject]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "objects": [obj.to_dict() for obj in self.objects],
        }


class PerceptionInputAdapter:
    """
    Parser and schema validator for Member 1 perception output.
    """

    @classmethod
    def parse_object(
        cls,
        raw_obj: Any,
        frame_index: int,
        timestamp: float,
    ) -> ParsedObject:
        """
        Validate and convert a raw perception object dictionary into a ParsedObject.
        """
        if not isinstance(raw_obj, dict):
            raise PerceptionValidationError(
                f"Object must be a dictionary, got {type(raw_obj).__name__}: {raw_obj!r}"
            )

        # 1. Validate 'id'
        if "id" not in raw_obj or not isinstance(raw_obj["id"], str) or not raw_obj["id"].strip():
            raise PerceptionValidationError(
                f"Object missing required non-empty string field 'id': {raw_obj}"
            )
        obj_id = raw_obj["id"].strip()

        # 2. Validate 'class' (also accept 'cls' if provided)
        cls_name = raw_obj.get("class") or raw_obj.get("cls")
        if not cls_name or not isinstance(cls_name, str) or not cls_name.strip():
            raise PerceptionValidationError(
                f"Object {obj_id!r} missing required non-empty string field 'class': {raw_obj}"
            )
        obj_class = cls_name.strip()

        # 3. Validate 'confidence'
        if "confidence" not in raw_obj:
            raise PerceptionValidationError(
                f"Object {obj_id!r} missing required field 'confidence': {raw_obj}"
            )
        try:
            confidence = float(raw_obj["confidence"])
        except (ValueError, TypeError):
            raise PerceptionValidationError(
                f"Object {obj_id!r} confidence must be a float, got {raw_obj['confidence']!r}"
            )
        if not (0.0 <= confidence <= 1.0):
            raise PerceptionValidationError(
                f"Object {obj_id!r} confidence must be in range [0.0, 1.0], got {confidence}"
            )

        # 4. Validate 'bbox'
        if "bbox" not in raw_obj:
            raise PerceptionValidationError(
                f"Object {obj_id!r} missing required field 'bbox': {raw_obj}"
            )
        raw_bbox = raw_obj["bbox"]
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise PerceptionValidationError(
                f"Object {obj_id!r} bbox must be a 4-element sequence [x1, y1, x2, y2], got {raw_bbox!r}"
            )
        try:
            bbox = (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))
        except (ValueError, TypeError):
            raise PerceptionValidationError(
                f"Object {obj_id!r} bbox coordinates must be integers, got {raw_bbox!r}"
            )
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise PerceptionValidationError(
                f"Object {obj_id!r} bbox invalid (x1 > x2 or y1 > y2): {bbox}"
            )

        # 5. Validate 'centroid'
        if "centroid" not in raw_obj:
            raise PerceptionValidationError(
                f"Object {obj_id!r} missing required field 'centroid': {raw_obj}"
            )
        raw_centroid = raw_obj["centroid"]
        if not isinstance(raw_centroid, (list, tuple)) or len(raw_centroid) != 2:
            raise PerceptionValidationError(
                f"Object {obj_id!r} centroid must be a 2-element sequence [cx, cy], got {raw_centroid!r}"
            )
        try:
            centroid = (int(raw_centroid[0]), int(raw_centroid[1]))
        except (ValueError, TypeError):
            raise PerceptionValidationError(
                f"Object {obj_id!r} centroid coordinates must be integers, got {raw_centroid!r}"
            )

        # 6. Validate 'mask_path' (nullable)
        mask_path = raw_obj.get("mask_path")
        if mask_path is not None and not isinstance(mask_path, str):
            raise PerceptionValidationError(
                f"Object {obj_id!r} mask_path must be string or null, got {type(mask_path).__name__}"
            )

        # 7. Validate 'depth' (nullable)
        raw_depth = raw_obj.get("depth")
        depth: Optional[float] = None
        if raw_depth is not None:
            try:
                depth = float(raw_depth)
            except (ValueError, TypeError):
                raise PerceptionValidationError(
                    f"Object {obj_id!r} depth must be a float or null, got {raw_depth!r}"
                )
            if depth < 0.0:
                raise PerceptionValidationError(
                    f"Object {obj_id!r} depth must be non-negative, got {depth}"
                )

        obs = ObjectObservation(
            frame_index=frame_index,
            timestamp=timestamp,
            centroid=centroid,
            bbox=bbox,
            confidence=confidence,
            depth=depth,
            mask_path=mask_path,
        )

        return ParsedObject(id=obj_id, cls=obj_class, observation=obs)

    @classmethod
    def parse_frame(cls, raw_frame: Any) -> ParsedFrame:
        """
        Validate and convert a raw perception frame dictionary into a ParsedFrame.
        """
        if not isinstance(raw_frame, dict):
            raise PerceptionValidationError(
                f"Perception frame must be a dictionary, got {type(raw_frame).__name__}"
            )

        # 1. Validate 'frame_index'
        if "frame_index" not in raw_frame:
            raise PerceptionValidationError(
                f"Frame missing required field 'frame_index': {raw_frame}"
            )
        try:
            frame_idx = int(raw_frame["frame_index"])
        except (ValueError, TypeError):
            raise PerceptionValidationError(
                f"frame_index must be an integer, got {raw_frame['frame_index']!r}"
            )
        if frame_idx < 0:
            raise PerceptionValidationError(
                f"frame_index must be non-negative, got {frame_idx}"
            )

        # 2. Validate 'timestamp'
        if "timestamp" not in raw_frame:
            raise PerceptionValidationError(
                f"Frame {frame_idx} missing required field 'timestamp': {raw_frame}"
            )
        try:
            timestamp = float(raw_frame["timestamp"])
        except (ValueError, TypeError):
            raise PerceptionValidationError(
                f"timestamp must be a float, got {raw_frame['timestamp']!r}"
            )
        if timestamp < 0.0:
            raise PerceptionValidationError(
                f"timestamp must be non-negative, got {timestamp}"
            )

        # 3. Validate 'objects'
        if "objects" not in raw_frame:
            raise PerceptionValidationError(
                f"Frame {frame_idx} missing required field 'objects': {raw_frame}"
            )
        raw_objects = raw_frame["objects"]
        if not isinstance(raw_objects, list):
            raise PerceptionValidationError(
                f"Frame {frame_idx} 'objects' must be a list, got {type(raw_objects).__name__}"
            )

        parsed_objects = [
            cls.parse_object(raw_obj, frame_index=frame_idx, timestamp=timestamp)
            for raw_obj in raw_objects
        ]

        return ParsedFrame(
            frame_index=frame_idx,
            timestamp=timestamp,
            objects=parsed_objects,
        )

    @classmethod
    def parse_sequence(
        cls,
        raw_input: Union[str, Path, Sequence[Dict[str, Any]]],
    ) -> List[ParsedFrame]:
        """
        Parse a JSON array of frames from a list of dicts, JSON string, or file path.
        Preserves original frame ordering.
        """
        if isinstance(raw_input, (str, Path)):
            input_path = Path(raw_input)
            if input_path.exists() and input_path.is_file():
                content = input_path.read_text(encoding="utf-8")
                raw_list = json.loads(content)
            else:
                # Attempt to parse as raw JSON string
                raw_list = json.loads(str(raw_input))
        else:
            raw_list = raw_input

        if not isinstance(raw_list, list):
            raise PerceptionValidationError(
                f"Perception sequence must be a JSON array (list), got {type(raw_list).__name__}"
            )

        parsed_frames = [cls.parse_frame(rf) for rf in raw_list]
        return parsed_frames

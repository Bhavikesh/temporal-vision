"""
reasoning/state.py
------------------
Temporal state tracking and object history manager (Member 2).

Converts per-frame perception observations (DetectedObject lists) into persistent,
multi-frame temporal states for each unique object ID. Computes 2D displacements,
velocities, depth deltas, and maintains observation histories across video frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ObjectObservation:
    """
    A single point-in-time observation of an object in a frame.
    """

    frame_index: int
    timestamp: float
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    depth: Optional[float] = None
    mask_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "centroid": list(self.centroid),
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "depth": round(self.depth, 4) if self.depth is not None else None,
            "mask_path": self.mask_path,
        }


@dataclass
class ObjectTemporalState:
    """
    Persistent temporal state for an object tracked across multiple frames.
    """

    id: str
    cls: str

    # Current observation state
    current_frame_index: int
    current_timestamp: float
    current_centroid: Tuple[int, int]
    current_bbox: Tuple[int, int, int, int]
    current_confidence: float
    current_depth: Optional[float] = None
    current_mask_path: Optional[str] = None

    # Previous observation state (None if first observation)
    previous_frame_index: Optional[int] = None
    previous_timestamp: Optional[float] = None
    previous_centroid: Optional[Tuple[int, int]] = None
    previous_bbox: Optional[Tuple[int, int, int, int]] = None
    previous_depth: Optional[float] = None

    # Motion and kinematics
    displacement_px: Tuple[float, float] = (0.0, 0.0)  # (dx, dy)
    distance_px: float = 0.0  # Euclidean distance moved since previous observation
    elapsed_time_s: float = 0.0  # dt since previous observation
    velocity_px_s: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/sec
    speed_px_s: float = 0.0  # scalar speed in px/sec
    depth_delta_m: Optional[float] = None  # current_depth - previous_depth (metres)
    direction_normalized: Tuple[float, float] = (0.0, 0.0)  # unit vector (dx/dist, dy/dist)

    # Visibility & lifecycle
    is_present: bool = True
    first_seen_frame: int = 0
    first_seen_timestamp: float = 0.0
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    total_observations: int = 1
    consecutive_missing_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.cls,
            "is_present": self.is_present,
            "current_frame_index": self.current_frame_index,
            "current_timestamp": round(self.current_timestamp, 4),
            "current_centroid": list(self.current_centroid),
            "current_bbox": list(self.current_bbox),
            "current_confidence": round(self.current_confidence, 4),
            "current_depth": round(self.current_depth, 4) if self.current_depth is not None else None,
            "previous_frame_index": self.previous_frame_index,
            "previous_timestamp": round(self.previous_timestamp, 4) if self.previous_timestamp is not None else None,
            "previous_centroid": list(self.previous_centroid) if self.previous_centroid is not None else None,
            "displacement_px": [round(v, 2) for v in self.displacement_px],
            "distance_px": round(self.distance_px, 2),
            "elapsed_time_s": round(self.elapsed_time_s, 4),
            "velocity_px_s": [round(v, 2) for v in self.velocity_px_s],
            "speed_px_s": round(self.speed_px_s, 2),
            "depth_delta_m": round(self.depth_delta_m, 4) if self.depth_delta_m is not None else None,
            "direction_normalized": [round(v, 4) for v in self.direction_normalized],
            "total_observations": self.total_observations,
            "consecutive_missing_frames": self.consecutive_missing_frames,
        }


class ObjectStateManager:
    """
    Maintains persistent temporal state and history for all tracked objects
    across sequential video frames.
    """

    def __init__(self, max_history_per_object: Optional[int] = 100) -> None:
        """
        Parameters
        ----------
        max_history_per_object : Optional[int]
            Maximum number of observations to retain in history per object ID.
            If None, history is unbounded.
        """
        self.max_history = max_history_per_object
        self._states: Dict[str, ObjectTemporalState] = {}
        self._histories: Dict[str, List[ObjectObservation]] = {}
        self.current_frame_index: Optional[int] = None
        self.current_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear all tracked objects and history."""
        self._states.clear()
        self._histories.clear()
        self.current_frame_index = None
        self.current_timestamp = None

    def update(self, frame: Union[Dict[str, Any], Any]) -> Dict[str, ObjectTemporalState]:
        """
        Process a new perception frame and update the temporal state of all objects.

        Parameters
        ----------
        frame : Union[Dict[str, Any], Any]
            PerceptionFrame instance or equivalent dictionary conforming to
            docs/DATA_MODEL.md Section 2.

        Returns
        -------
        Dict[str, ObjectTemporalState]
            Mapping of object ID -> updated temporal state for all tracked objects.
        """
        frame_idx, timestamp, raw_objects = self._parse_frame(frame)
        self.current_frame_index = frame_idx
        self.current_timestamp = timestamp

        seen_in_this_frame = set()

        for raw_obj in raw_objects:
            obj_id, cls_name, conf, bbox, centroid, mask_path, depth = self._parse_object(raw_obj)
            seen_in_this_frame.add(obj_id)

            obs = ObjectObservation(
                frame_index=frame_idx,
                timestamp=timestamp,
                centroid=centroid,
                bbox=bbox,
                confidence=conf,
                depth=depth,
                mask_path=mask_path,
            )

            # Record observation in history
            if obj_id not in self._histories:
                self._histories[obj_id] = []
            hist = self._histories[obj_id]
            hist.append(obs)
            if self.max_history is not None and len(hist) > self.max_history:
                hist.pop(0)

            # Update or initialize state
            if obj_id not in self._states:
                # First observation
                self._states[obj_id] = ObjectTemporalState(
                    id=obj_id,
                    cls=cls_name,
                    current_frame_index=frame_idx,
                    current_timestamp=timestamp,
                    current_centroid=centroid,
                    current_bbox=bbox,
                    current_confidence=conf,
                    current_depth=depth,
                    current_mask_path=mask_path,
                    is_present=True,
                    first_seen_frame=frame_idx,
                    first_seen_timestamp=timestamp,
                    last_seen_frame=frame_idx,
                    last_seen_timestamp=timestamp,
                    total_observations=1,
                    consecutive_missing_frames=0,
                )
            else:
                # Subsequent observation (or reappearance after absence)
                st = self._states[obj_id]

                prev_cx, prev_cy = st.current_centroid
                curr_cx, curr_cy = centroid
                dx = float(curr_cx - prev_cx)
                dy = float(curr_cy - prev_cy)
                dist = math.hypot(dx, dy)

                dt = timestamp - st.current_timestamp
                # Guard against non-positive or near-zero dt
                if dt > 1e-6:
                    vx = dx / dt
                    vy = dy / dt
                    speed = dist / dt
                else:
                    dt = 0.0
                    vx = 0.0
                    vy = 0.0
                    speed = 0.0

                if dist > 1e-6:
                    dir_norm = (dx / dist, dy / dist)
                else:
                    dir_norm = (0.0, 0.0)

                depth_delta = None
                if depth is not None and st.current_depth is not None:
                    depth_delta = depth - st.current_depth

                # Shift current to previous
                st.previous_frame_index = st.current_frame_index
                st.previous_timestamp = st.current_timestamp
                st.previous_centroid = st.current_centroid
                st.previous_bbox = st.current_bbox
                st.previous_depth = st.current_depth

                # Update current
                st.current_frame_index = frame_idx
                st.current_timestamp = timestamp
                st.current_centroid = centroid
                st.current_bbox = bbox
                st.current_confidence = conf
                st.current_depth = depth
                st.current_mask_path = mask_path

                # Kinematics
                st.displacement_px = (dx, dy)
                st.distance_px = dist
                st.elapsed_time_s = dt
                st.velocity_px_s = (vx, vy)
                st.speed_px_s = speed
                st.depth_delta_m = depth_delta
                st.direction_normalized = dir_norm

                # Presence
                st.is_present = True
                st.last_seen_frame = frame_idx
                st.last_seen_timestamp = timestamp
                st.total_observations += 1
                st.consecutive_missing_frames = 0

        # Handle objects missing in this frame
        for obj_id, st in self._states.items():
            if obj_id not in seen_in_this_frame:
                st.is_present = False
                st.consecutive_missing_frames += 1
                # When missing in this frame, instantaneous movement is zero
                st.displacement_px = (0.0, 0.0)
                st.distance_px = 0.0
                st.velocity_px_s = (0.0, 0.0)
                st.speed_px_s = 0.0
                st.direction_normalized = (0.0, 0.0)
                st.depth_delta_m = None

        return self._states

    def get_state(self, object_id: str) -> Optional[ObjectTemporalState]:
        """Return the temporal state for a given object ID, or None if unknown."""
        return self._states.get(object_id)

    def get_history(
        self, object_id: str, limit: Optional[int] = None
    ) -> List[ObjectObservation]:
        """
        Return the observation history for an object ID in chronological order.
        """
        hist = self._histories.get(object_id, [])
        if limit is not None and limit > 0:
            return hist[-limit:]
        return list(hist)

    def get_all_states(
        self, only_present: bool = False
    ) -> Dict[str, ObjectTemporalState]:
        """
        Return all tracked states.
        If only_present is True, filters to objects detected in the latest frame.
        """
        if only_present:
            return {
                oid: st for oid, st in self._states.items() if st.is_present
            }
        return dict(self._states)

    @property
    def tracked_ids(self) -> List[str]:
        """List of all object IDs tracked across the session."""
        return list(self._states.keys())

    @property
    def present_ids(self) -> List[str]:
        """List of object IDs present in the most recently processed frame."""
        return [oid for oid, st in self._states.items() if st.is_present]

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_frame(
        frame: Union[Dict[str, Any], Any]
    ) -> Tuple[int, float, Sequence[Any]]:
        """Extract (frame_index, timestamp, objects) safely from dict or object."""
        if isinstance(frame, dict):
            frame_idx = int(frame.get("frame_index", 0))
            timestamp = float(frame.get("timestamp", 0.0))
            objects = frame.get("objects", [])
        else:
            frame_idx = int(getattr(frame, "frame_index", 0))
            timestamp = float(getattr(frame, "timestamp", 0.0))
            objects = getattr(frame, "objects", [])
        return frame_idx, timestamp, objects

    @staticmethod
    def _parse_object(
        obj: Union[Dict[str, Any], Any]
    ) -> Tuple[
        str,
        str,
        float,
        Tuple[int, int, int, int],
        Tuple[int, int],
        Optional[str],
        Optional[float],
    ]:
        """Extract standard attributes from a detected object dict or instance."""
        if isinstance(obj, dict):
            obj_id = str(obj["id"])
            cls_name = str(obj.get("class") or obj.get("cls") or "unknown")
            conf = float(obj.get("confidence", 1.0))
            bbox_raw = obj.get("bbox", [0, 0, 0, 0])
            centroid_raw = obj.get("centroid", [0, 0])
            mask_path = obj.get("mask_path")
            depth_raw = obj.get("depth")
        else:
            obj_id = str(getattr(obj, "id"))
            cls_name = str(
                getattr(obj, "cls", getattr(obj, "class_", getattr(obj, "class", "unknown")))
            )
            conf = float(getattr(obj, "confidence", 1.0))
            bbox_raw = getattr(obj, "bbox", [0, 0, 0, 0])
            centroid_raw = getattr(obj, "centroid", [0, 0])
            mask_path = getattr(obj, "mask_path", None)
            depth_raw = getattr(obj, "depth", None)

        bbox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
        centroid = (int(centroid_raw[0]), int(centroid_raw[1]))
        depth = float(depth_raw) if depth_raw is not None else None

        return obj_id, cls_name, conf, bbox, centroid, mask_path, depth

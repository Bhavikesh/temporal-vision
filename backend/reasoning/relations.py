"""
reasoning/relations.py
----------------------
Deterministic spatial and temporal relation computation (Member 2).

Evaluates directed spatial and kinematic relations between pairs of tracked objects
using the persistent state maintained by ObjectStateManager.

Supported relation vocabulary per docs/DATA_MODEL.md Section 3:
  - near           (spatial proximity threshold in 3D metres and 2D pixels)
  - on             (spatial support / resting configuration e.g. laptop on table)
  - moving-toward  (decreasing distance + consistent motion direction)
  - moving-away    (increasing distance + receding motion direction)
  - holding        (close proximity + synchronized multi-frame motion)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from .state import ObjectStateManager, ObjectTemporalState
from .types import RelationType, SpatialRelation


@dataclass
class RelationConfig:
    """
    Configurable thresholds for relation reasoning.
    All distance thresholds have explicit units (pixels or metres).
    """

    # Proximity thresholds ('near')
    proximity_threshold_px: float = 150.0
    proximity_threshold_m: float = 1.2
    max_depth_diff_near_m: float = 0.8
    nominal_focal_length_px: float = 1000.0

    # Support surface thresholds ('on')
    on_horizontal_padding_px: float = 30.0
    on_vertical_tolerance_px: float = 50.0
    on_max_depth_diff_m: float = 0.5
    on_max_subject_speed_px_s: float = 20.0

    # Motion direction & kinematics ('moving-toward', 'moving-away')
    min_motion_distance_px: float = 2.0
    min_motion_speed_px_s: float = 5.0
    min_direction_cosine: float = 0.3

    # Synchronized motion ('holding')
    holding_proximity_px: float = 120.0
    holding_proximity_m: float = 1.0
    holding_min_speed_px_s: float = 8.0
    holding_min_sync_cosine: float = 0.65
    holding_max_dist_drift_px: float = 30.0

    # Resolution scaling support (Part A)
    scale_factor: float = 1.0
    frame_resolution: Optional[Tuple[int, int]] = None
    reference_resolution: Tuple[int, int] = (1280, 720)

    @classmethod
    def for_resolution(
        cls,
        width: int,
        height: int,
        reference_resolution: Tuple[int, int] = (1280, 720),
        **kwargs: Any,
    ) -> RelationConfig:
        """Create a RelationConfig scaled for a specific video resolution."""
        scale = math.hypot(width, height) / math.hypot(
            reference_resolution[0], reference_resolution[1]
        )
        return cls(
            scale_factor=scale,
            frame_resolution=(width, height),
            reference_resolution=reference_resolution,
            **kwargs,
        )

    @property
    def effective_proximity_threshold_px(self) -> float:
        return self.proximity_threshold_px * self.scale_factor

    @property
    def effective_holding_proximity_px(self) -> float:
        return self.holding_proximity_px * self.scale_factor

    @property
    def effective_holding_max_dist_drift_px(self) -> float:
        return self.holding_max_dist_drift_px * self.scale_factor

    @property
    def effective_min_motion_distance_px(self) -> float:
        return self.min_motion_distance_px * self.scale_factor

    @property
    def effective_on_horizontal_padding_px(self) -> float:
        return self.on_horizontal_padding_px * self.scale_factor

    @property
    def effective_on_vertical_tolerance_px(self) -> float:
        return self.on_vertical_tolerance_px * self.scale_factor


class RelationCalculator:
    """
    Computes deterministic spatial and temporal relations for object pairs.
    """

    def __init__(self, config: Optional[RelationConfig] = None) -> None:
        self.config = config or RelationConfig()

    def compute(
        self,
        source: Union[ObjectStateManager, Dict[str, ObjectTemporalState]],
        frame_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[SpatialRelation]:
        """
        Compute all active spatial relations across present objects.

        Parameters
        ----------
        source : Union[ObjectStateManager, Dict[str, ObjectTemporalState]]
            Active ObjectStateManager instance or dict of ObjectTemporalState.
        frame_index : Optional[int]
            Frame number (required if source is dict and not manager).
        timestamp : Optional[float]
            Timestamp in seconds (required if source is dict and not manager).

        Returns
        -------
        List[SpatialRelation]
            List of detected directed relations for this frame.
        """
        if isinstance(source, ObjectStateManager):
            states = source.get_all_states(only_present=True)
            f_idx = source.current_frame_index if source.current_frame_index is not None else 0
            ts = source.current_timestamp if source.current_timestamp is not None else 0.0
        else:
            states = {oid: st for oid, st in source.items() if st.is_present}
            f_idx = frame_index if frame_index is not None else 0
            ts = timestamp if timestamp is not None else 0.0

        relations: List[SpatialRelation] = []
        present_objects = list(states.values())

        for i, subj in enumerate(present_objects):
            for j, obj in enumerate(present_objects):
                if i == j or subj.id == obj.id:
                    continue
                pair_rels = self.compute_pair_relations(subj, obj, f_idx, ts)
                relations.extend(pair_rels)

        return relations

    def compute_pair_relations(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
        frame_index: int,
        timestamp: float,
    ) -> List[SpatialRelation]:
        """
        Evaluate all candidate relations directed from subject to object.
        """
        if not subject.is_present or not obj.is_present:
            return []

        dist_m = self.estimate_distance_m(subject, obj)
        results: List[SpatialRelation] = []

        # 1. 'near'
        if self.check_near(subject, obj, dist_m):
            results.append(
                SpatialRelation(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    subject_id=subject.id,
                    relation=RelationType.NEAR,
                    object_id=obj.id,
                    distance_m=dist_m,
                )
            )

        # 2. 'on'
        if self.check_on(subject, obj):
            results.append(
                SpatialRelation(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    subject_id=subject.id,
                    relation=RelationType.ON,
                    object_id=obj.id,
                    distance_m=dist_m,
                )
            )

        # 3. 'moving-toward'
        if self.check_moving_toward(subject, obj):
            results.append(
                SpatialRelation(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    subject_id=subject.id,
                    relation=RelationType.MOVING_TOWARD,
                    object_id=obj.id,
                    distance_m=dist_m,
                )
            )

        # 4. 'moving-away'
        if self.check_moving_away(subject, obj):
            results.append(
                SpatialRelation(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    subject_id=subject.id,
                    relation=RelationType.MOVING_AWAY,
                    object_id=obj.id,
                    distance_m=dist_m,
                )
            )

        # 5. 'holding'
        if self.check_holding(subject, obj, dist_m):
            results.append(
                SpatialRelation(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    subject_id=subject.id,
                    relation=RelationType.HOLDING,
                    object_id=obj.id,
                    distance_m=dist_m,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Individual Relation Checkers
    # ------------------------------------------------------------------
    def check_near(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
        dist_m: Optional[float] = None,
    ) -> bool:
        """
        Check if subject and object are spatially proximate.
        Evaluates 2D centroid Euclidean distance AND 3D metric distance / depth
        plane alignment when depth is available.
        """
        dx = float(subject.current_centroid[0] - obj.current_centroid[0])
        dy = float(subject.current_centroid[1] - obj.current_centroid[1])
        dist_px = math.hypot(dx, dy)

        # 2D proximity boundary
        if dist_px > self.config.effective_proximity_threshold_px:
            return False

        # If 3D metric depth is available, enforce depth consistency
        if dist_m is not None:
            depth_diff = abs(
                (subject.current_depth or 0.0) - (obj.current_depth or 0.0)
            )
            if (
                dist_m > self.config.proximity_threshold_m
                or depth_diff > self.config.max_depth_diff_near_m
            ):
                return False

        return True

    def check_on(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
    ) -> bool:
        """
        Check if subject is resting on top of object (e.g. laptop on table).
        Evaluates horizontal containment, resting vertical contact, and depth alignment.
        """
        sx1, sy1, sx2, sy2 = subject.current_bbox
        ox1, oy1, ox2, oy2 = obj.current_bbox

        # Validate bounding boxes (must be non-empty)
        if (sx2 - sx1) <= 0 or (sy2 - sy1) <= 0 or (ox2 - ox1) <= 0 or (oy2 - oy1) <= 0:
            return False

        # Subject must be above or at the top of the surface in image space
        # (y increases downwards)
        scx, scy = subject.current_centroid
        ocx, ocy = obj.current_centroid

        # Centroid of supported object must be vertically above surface centroid
        if scy >= ocy:
            return False

        # Horizontal alignment: subject centroid must be within object horizontal span
        pad = self.config.effective_on_horizontal_padding_px
        if scx < (ox1 - pad) or scx > (ox2 + pad):
            return False

        # Vertical resting alignment: subject bottom edge (sy2) should be near
        # the upper region of the surface (oy1), not far above it or completely submerged
        v_tol = self.config.effective_on_vertical_tolerance_px
        vertical_contact = (sy2 >= oy1 - v_tol) and (sy2 <= oy2) and (sy1 <= oy1 + v_tol)
        if not vertical_contact:
            return False

        # Depth alignment if both depths available
        if subject.current_depth is not None and obj.current_depth is not None:
            if abs(subject.current_depth - obj.current_depth) > self.config.on_max_depth_diff_m:
                return False

        # Stability: supported object should not be moving rapidly across the frame
        if subject.speed_px_s > self.config.on_max_subject_speed_px_s:
            return False

        return True

    def check_moving_toward(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
    ) -> bool:
        """
        Check if subject is moving toward object over time.
        Requires decreasing separation and positive directional alignment.
        """
        if subject.previous_centroid is None:
            return False

        # Subject must have non-negligible motion
        if (
            subject.distance_px < self.config.min_motion_distance_px
            or subject.speed_px_s < self.config.min_motion_speed_px_s
        ):
            return False

        curr_s = subject.current_centroid
        prev_s = subject.previous_centroid
        curr_o = obj.current_centroid
        prev_o = obj.previous_centroid if obj.previous_centroid is not None else curr_o

        dist_curr = math.hypot(curr_s[0] - curr_o[0], curr_s[1] - curr_o[1])
        dist_prev = math.hypot(prev_s[0] - prev_o[0], prev_s[1] - prev_o[1])

        # Distance must decrease
        if dist_curr >= dist_prev - self.config.min_motion_distance_px:
            return False

        # Direction vector from subject's previous position toward object
        vec_to_obj = (float(prev_o[0] - prev_s[0]), float(prev_o[1] - prev_s[1]))
        norm_to_obj = math.hypot(vec_to_obj[0], vec_to_obj[1])

        # Subject displacement vector
        disp_s = (float(curr_s[0] - prev_s[0]), float(curr_s[1] - prev_s[1]))
        norm_disp = math.hypot(disp_s[0], disp_s[1])

        if norm_to_obj <= 1e-6 or norm_disp <= 1e-6:
            return False

        # Cosine similarity
        cosine = (disp_s[0] * vec_to_obj[0] + disp_s[1] * vec_to_obj[1]) / (
            norm_disp * norm_to_obj
        )

        return cosine >= self.config.min_direction_cosine

    def check_moving_away(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
    ) -> bool:
        """
        Check if subject is moving away from object over time.
        Requires increasing separation and receding directional alignment.
        """
        if subject.previous_centroid is None:
            return False

        if (
            subject.distance_px < self.config.min_motion_distance_px
            or subject.speed_px_s < self.config.min_motion_speed_px_s
        ):
            return False

        curr_s = subject.current_centroid
        prev_s = subject.previous_centroid
        curr_o = obj.current_centroid
        prev_o = obj.previous_centroid if obj.previous_centroid is not None else curr_o

        dist_curr = math.hypot(curr_s[0] - curr_o[0], curr_s[1] - curr_o[1])
        dist_prev = math.hypot(prev_s[0] - prev_o[0], prev_s[1] - prev_o[1])

        # Distance must increase
        if dist_curr <= dist_prev + self.config.min_motion_distance_px:
            return False

        # Vector from subject toward object
        vec_to_obj = (float(prev_o[0] - prev_s[0]), float(prev_o[1] - prev_s[1]))
        norm_to_obj = math.hypot(vec_to_obj[0], vec_to_obj[1])

        # Subject displacement vector
        disp_s = (float(curr_s[0] - prev_s[0]), float(curr_s[1] - prev_s[1]))
        norm_disp = math.hypot(disp_s[0], disp_s[1])

        if norm_to_obj <= 1e-6 or norm_disp <= 1e-6:
            return False

        cosine = (disp_s[0] * vec_to_obj[0] + disp_s[1] * vec_to_obj[1]) / (
            norm_disp * norm_to_obj
        )

        # Negative cosine indicates motion heading away from the target
        return cosine <= -self.config.min_direction_cosine

    def check_holding(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
        dist_m: Optional[float] = None,
    ) -> bool:
        """
        Check if subject and object are moving in synchronized contact ('holding').
        Requires:
          1. Spatial proximity (close centroids).
          2. Both objects are in active motion.
          3. Strong directional alignment (high velocity cosine).
          4. Stable relative separation across frames (rigid body motion).
        """
        if subject.previous_centroid is None or obj.previous_centroid is None:
            return False

        # 1. Proximity check
        dx = float(subject.current_centroid[0] - obj.current_centroid[0])
        dy = float(subject.current_centroid[1] - obj.current_centroid[1])
        dist_px = math.hypot(dx, dy)

        if dist_px > self.config.effective_holding_proximity_px:
            return False
        if dist_m is not None and dist_m > self.config.holding_proximity_m:
            return False

        # 2. Both objects must be moving
        if (
            subject.speed_px_s < self.config.holding_min_speed_px_s
            or obj.speed_px_s < self.config.holding_min_speed_px_s
            or subject.distance_px < self.config.min_motion_distance_px
            or obj.distance_px < self.config.min_motion_distance_px
        ):
            return False

        # 3. Synchronized motion direction (velocity vector alignment)
        vx_s, vy_s = subject.velocity_px_s
        vx_o, vy_o = obj.velocity_px_s
        norm_s = math.hypot(vx_s, vy_s)
        norm_o = math.hypot(vx_o, vy_o)

        if norm_s <= 1e-6 or norm_o <= 1e-6:
            return False

        sync_cosine = (vx_s * vx_o + vy_s * vy_o) / (norm_s * norm_o)
        if sync_cosine < self.config.holding_min_sync_cosine:
            return False

        # 4. Relative distance stability (they travel together without drifting apart)
        prev_dx = float(subject.previous_centroid[0] - obj.previous_centroid[0])
        prev_dy = float(subject.previous_centroid[1] - obj.previous_centroid[1])
        prev_dist_px = math.hypot(prev_dx, prev_dy)

        dist_drift = abs(dist_px - prev_dist_px)
        if dist_drift > self.config.effective_holding_max_dist_drift_px:
            return False

        return True

    # ------------------------------------------------------------------
    # Distance Utilities
    # ------------------------------------------------------------------
    def estimate_distance_m(
        self,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
    ) -> Optional[float]:
        """
        Estimate 3D Euclidean distance in metres if metric depth is available
        for both objects, using a standard pinhole projection model.
        Returns None if depth is unavailable on either object.
        """
        zs = subject.current_depth
        zo = obj.current_depth
        if zs is None or zo is None:
            return None

        f = self.config.nominal_focal_length_px
        scx, scy = subject.current_centroid
        ocx, ocy = obj.current_centroid

        # 3D unprojection relative to principal axis
        dx_m = (scx * zs - ocx * zo) / f
        dy_m = (scy * zs - ocy * zo) / f
        dz_m = zs - zo

        dist_3d = math.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
        return round(dist_3d, 4)

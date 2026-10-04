"""
reasoning/evidence.py
---------------------
Evidence collection and validation engine (Member 2).

Given temporal state, spatial relations, and event context, collects and validates
observable visual evidence supporting an already-detected event.

Produces the standard Evidence dataclass defined in docs/DATA_MODEL.md:
  - frame_index        (frame where evidence was observed)
  - distance_m         (metric 3D distance if depth available, else None)
  - position_change    (observed positional change flag or metric displacement)
  - synchronized_motion (whether subject and object moved in sync)
  - depth_delta_m      (signed metric depth change if depth available, else None)

Does NOT perform event detection itself.
Does NOT invent metric distance from pixel coordinates when depth is unavailable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

from .relations import RelationCalculator
from .state import ObjectTemporalState
from .types import EventType, Evidence


class EvidenceValidationError(ValueError):
    """Raised when evidence metrics fail numerical or structural validation."""
    pass


class EvidenceCollector:
    """
    Collects, derives, and validates structured evidence for detected events.
    """

    def __init__(self, relation_calculator: Optional[RelationCalculator] = None) -> None:
        self.rel_calc = relation_calculator or RelationCalculator()

    # ------------------------------------------------------------------
    # Core Evidence Collection
    # ------------------------------------------------------------------
    def collect(
        self,
        event_type: Union[EventType, str],
        frame_index: int,
        subject: ObjectTemporalState,
        obj: Optional[ObjectTemporalState] = None,
        synchronized_motion: Optional[bool] = None,
        distance_m: Optional[float] = None,
        depth_delta_m: Optional[float] = None,
        position_change: Optional[Union[bool, float]] = None,
    ) -> Evidence:
        """
        Derive and validate an Evidence instance for an already-detected event.

        Parameters
        ----------
        event_type : Union[EventType, str]
            The detected event (APPROACH, REACH, PICK_UP, CARRY).
        frame_index : int
            Video frame where the event evidence was observed.
        subject : ObjectTemporalState
            Temporal state of the acting subject.
        obj : Optional[ObjectTemporalState]
            Temporal state of the target object, if applicable.
        synchronized_motion : Optional[bool]
            Explicit synchronized motion flag (derived if None).
        distance_m : Optional[float]
            Explicit metric distance in metres (derived via depth if None).
        depth_delta_m : Optional[float]
            Explicit signed depth delta in metres (derived from subject if None).
        position_change : Optional[Union[bool, float]]
            Explicit position change flag/metric (derived from kinematics if None).

        Returns
        -------
        Evidence
            Validated Evidence dataclass instance.
        """
        evt_str = event_type.value if hasattr(event_type, "value") else str(event_type)

        # 1. Derive distance_m (strictly from depth, never invented from pixels)
        derived_dist_m: Optional[float] = distance_m
        if derived_dist_m is None and obj is not None:
            derived_dist_m = self.rel_calc.estimate_distance_m(subject, obj)

        # 2. Derive position_change
        derived_pos_change: Optional[Union[bool, float]] = position_change
        if derived_pos_change is None:
            if evt_str in (EventType.PICK_UP.value, EventType.CARRY.value) and obj is not None:
                # For pickup/carry, evidence focuses on the target object moving
                derived_pos_change = bool(obj.distance_px >= 2.0 or obj.speed_px_s >= 5.0)
            else:
                derived_pos_change = bool(subject.distance_px >= 2.0 or subject.speed_px_s >= 5.0)

        # 3. Derive synchronized_motion
        derived_sync_motion: Optional[bool] = synchronized_motion
        if derived_sync_motion is None:
            if evt_str in (EventType.PICK_UP.value, EventType.CARRY.value) and obj is not None:
                derived_sync_motion = self.rel_calc.check_holding(subject, obj, dist_m=derived_dist_m)
            elif evt_str in (EventType.APPROACH.value, EventType.REACH.value):
                derived_sync_motion = False

        # 4. Derive depth_delta_m (signed metric change)
        derived_depth_delta: Optional[float] = depth_delta_m
        if derived_depth_delta is None:
            derived_depth_delta = subject.depth_delta_m

        # Build and validate
        evidence = Evidence(
            frame_index=frame_index,
            distance_m=derived_dist_m,
            position_change=derived_pos_change,
            synchronized_motion=derived_sync_motion,
            depth_delta_m=derived_depth_delta,
        )

        return self.validate(evidence)

    # ------------------------------------------------------------------
    # Direct Builder & Validation
    # ------------------------------------------------------------------
    def build_evidence(
        self,
        frame_index: int,
        distance_m: Optional[float] = None,
        position_change: Optional[Union[bool, float]] = None,
        synchronized_motion: Optional[bool] = None,
        depth_delta_m: Optional[float] = None,
    ) -> Evidence:
        """Construct an Evidence instance directly with validation."""
        ev = Evidence(
            frame_index=frame_index,
            distance_m=distance_m,
            position_change=position_change,
            synchronized_motion=synchronized_motion,
            depth_delta_m=depth_delta_m,
        )
        return self.validate(ev)

    def validate(self, evidence: Evidence) -> Evidence:
        """
        Validate an Evidence dataclass instance according to numerical and schema rules.
        Raises EvidenceValidationError if constraints are violated.
        """
        # 1. frame_index
        if isinstance(evidence.frame_index, bool) or not isinstance(evidence.frame_index, int):
            raise EvidenceValidationError(
                f"frame_index must be an integer, got {type(evidence.frame_index).__name__}: {evidence.frame_index!r}"
            )
        if evidence.frame_index < 0:
            raise EvidenceValidationError(
                f"frame_index must be non-negative, got {evidence.frame_index}"
            )

        # 2. distance_m (nullable, non-negative, finite)
        if evidence.distance_m is not None:
            if isinstance(evidence.distance_m, bool) or not isinstance(evidence.distance_m, (int, float)):
                raise EvidenceValidationError(
                    f"distance_m must be a float or None, got {type(evidence.distance_m).__name__}"
                )
            if not math.isfinite(evidence.distance_m):
                raise EvidenceValidationError(
                    f"distance_m must be finite, got {evidence.distance_m!r}"
                )
            if evidence.distance_m < 0.0:
                raise EvidenceValidationError(
                    f"distance_m must be non-negative, got {evidence.distance_m}"
                )

        # 3. depth_delta_m (nullable, signed, finite)
        if evidence.depth_delta_m is not None:
            if isinstance(evidence.depth_delta_m, bool) or not isinstance(evidence.depth_delta_m, (int, float)):
                raise EvidenceValidationError(
                    f"depth_delta_m must be a float or None, got {type(evidence.depth_delta_m).__name__}"
                )
            if not math.isfinite(evidence.depth_delta_m):
                raise EvidenceValidationError(
                    f"depth_delta_m must be finite, got {evidence.depth_delta_m!r}"
                )

        # 4. position_change (nullable, boolean or finite numeric)
        if evidence.position_change is not None:
            if isinstance(evidence.position_change, bool):
                pass
            elif isinstance(evidence.position_change, (int, float)):
                if not math.isfinite(evidence.position_change):
                    raise EvidenceValidationError(
                        f"position_change must be finite when numeric, got {evidence.position_change!r}"
                    )
            else:
                raise EvidenceValidationError(
                    f"position_change must be boolean, numeric, or None, got {type(evidence.position_change).__name__}"
                )

        # 5. synchronized_motion (nullable, boolean)
        if evidence.synchronized_motion is not None:
            if not isinstance(evidence.synchronized_motion, bool):
                raise EvidenceValidationError(
                    f"synchronized_motion must be a boolean or None, got {type(evidence.synchronized_motion).__name__}"
                )

        return evidence

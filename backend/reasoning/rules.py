"""
reasoning/rules.py
------------------
Deterministic temporal event reasoning engine (Member 2).

Detects temporal event transitions across video frames:
  - APPROACH  (subject moves toward target across sequential frames)
  - REACH     (subject arrives at target following an approach)
  - PICK_UP   (stationary target begins moving in synchronized contact with subject)
  - CARRY     (sustained multi-frame synchronized motion post-pickup)

Maintains per-pair interaction stages to prevent duplicate event spam and ensure
events are emitted at true temporal state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .relations import RelationCalculator, RelationConfig
from .state import ObjectStateManager, ObjectTemporalState
from .types import Event, EventType, Evidence, RelationType, SpatialRelation


class PairStage(str, Enum):
    """
    Interaction lifecycle stages for a (subject, object) pair.
    """
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    REACHED = "REACHED"
    PICKED_UP = "PICKED_UP"
    CARRYING = "CARRYING"


@dataclass
class EventRuleConfig:
    """
    Configurable thresholds and parameters for temporal event rules.
    Supports active subject filtering (Part B) and resolution-aware scaling (Part A).
    """

    # Candidate subject classes (Part B: restricted to active agents by default)
    candidate_subject_classes: Optional[Set[str]] = field(
        default_factory=lambda: {"person"}
    )

    # Resolution scaling support (Part A)
    scale_factor: float = 1.0
    frame_resolution: Optional[Tuple[int, int]] = None
    reference_resolution: Tuple[int, int] = (1280, 720)

    # APPROACH parameters
    approach_min_frames: int = 1
    approach_max_start_distance_px: float = 600.0

    # REACH parameters
    reach_proximity_threshold_px: float = 160.0
    reach_proximity_threshold_m: float = 1.4

    # PICK_UP parameters
    pickup_contact_proximity_px: float = 140.0
    pickup_contact_proximity_m: float = 1.0
    pickup_stationary_speed_threshold_px_s: float = 6.0
    pickup_min_object_speed_px_s: float = 8.0
    pickup_min_subject_speed_px_s: float = 8.0
    pickup_min_sync_cosine: float = 0.65
    pickup_max_dist_drift_px: float = 30.0

    # CARRY parameters
    carry_proximity_px: float = 140.0
    carry_proximity_m: float = 1.0
    carry_min_speed_px_s: float = 8.0
    carry_min_sync_cosine: float = 0.65
    carry_max_dist_drift_px: float = 35.0

    @classmethod
    def for_resolution(
        cls,
        width: int,
        height: int,
        reference_resolution: Tuple[int, int] = (1280, 720),
        **kwargs: Any,
    ) -> EventRuleConfig:
        """Create an EventRuleConfig scaled for a specific video resolution."""
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
    def effective_approach_max_start_distance_px(self) -> float:
        return self.approach_max_start_distance_px * self.scale_factor

    @property
    def effective_reach_proximity_threshold_px(self) -> float:
        return self.reach_proximity_threshold_px * self.scale_factor

    @property
    def effective_pickup_contact_proximity_px(self) -> float:
        return self.pickup_contact_proximity_px * self.scale_factor

    @property
    def effective_carry_proximity_px(self) -> float:
        return self.carry_proximity_px * self.scale_factor

    @property
    def effective_pickup_max_dist_drift_px(self) -> float:
        return self.pickup_max_dist_drift_px * self.scale_factor

    @property
    def effective_carry_max_dist_drift_px(self) -> float:
        return self.carry_max_dist_drift_px * self.scale_factor


@dataclass
class PairTracker:
    """
    Tracks the temporal interaction state and event lifecycle for a specific
    (subject_id, object_id) pair.
    """

    subject_id: str
    object_id: str
    stage: PairStage = PairStage.IDLE
    consecutive_approach_frames: int = 0
    consecutive_reach_frames: int = 0
    was_target_stationary: bool = True
    pickup_frame_index: Optional[int] = None
    events_emitted: Set[EventType] = field(default_factory=set)


class TemporalEventEngine:
    """
    Deterministic temporal reasoning engine for event detection.
    """

    def __init__(
        self,
        config: Optional[EventRuleConfig] = None,
        relation_calculator: Optional[RelationCalculator] = None,
    ) -> None:
        self.config = config or EventRuleConfig()
        self.rel_calc = relation_calculator or RelationCalculator()
        self._trackers: Dict[Tuple[str, str], PairTracker] = {}

    def reset(self) -> None:
        """Reset all tracked pair interaction states."""
        self._trackers.clear()

    def process_frame(
        self,
        manager: ObjectStateManager,
        relations: Optional[List[SpatialRelation]] = None,
    ) -> List[Event]:
        """
        Process the latest frame from the state manager and return newly
        triggered events for this frame.

        Parameters
        ----------
        manager : ObjectStateManager
            State manager after calling update(frame).
        relations : Optional[List[SpatialRelation]]
            Precomputed relations for this frame (computed if None).

        Returns
        -------
        List[Event]
            Any newly triggered events in this frame.
        """
        if manager.current_frame_index is None or manager.current_timestamp is None:
            return []

        frame_idx = manager.current_frame_index
        timestamp = manager.current_timestamp

        active_states = manager.get_all_states(only_present=True)
        if len(active_states) < 2:
            return []

        if relations is None:
            relations = self.rel_calc.compute(manager)

        # Map active relations by (subject_id, object_id) -> Set[RelationType]
        rel_map: Dict[Tuple[str, str], Set[Union[RelationType, str]]] = {}
        for r in relations:
            key = (r.subject_id, r.object_id)
            if key not in rel_map:
                rel_map[key] = set()
            rel_map[key].add(r.relation)

        new_events: List[Event] = []

        present_objects = list(active_states.values())
        for subj in present_objects:
            if (
                self.config.candidate_subject_classes is not None
                and subj.cls not in self.config.candidate_subject_classes
            ):
                continue

            for obj in present_objects:
                if subj.id == obj.id:
                    continue

                pair_key = (subj.id, obj.id)
                tracker = self._get_or_create_tracker(subj.id, obj.id)

                pair_rels = rel_map.get(pair_key, set())
                events = self._evaluate_pair(subj, obj, tracker, pair_rels, frame_idx, timestamp)
                new_events.extend(events)

        return new_events

    def process_sequence(
        self,
        frames: Sequence[Union[Dict[str, Any], Any]],
    ) -> List[Event]:
        """
        Process a complete sequential list of perception frames and return
        all detected events in chronological order.
        """
        self.reset()
        manager = ObjectStateManager()
        all_events: List[Event] = []

        for frame in frames:
            manager.update(frame)
            events = self.process_frame(manager)
            all_events.extend(events)

        return all_events

    # ------------------------------------------------------------------
    # Pair Evaluation
    # ------------------------------------------------------------------
    def _get_or_create_tracker(self, subject_id: str, object_id: str) -> PairTracker:
        key = (subject_id, object_id)
        if key not in self._trackers:
            self._trackers[key] = PairTracker(subject_id=subject_id, object_id=object_id)
        return self._trackers[key]

    def _evaluate_pair(
        self,
        subj: ObjectTemporalState,
        obj: ObjectTemporalState,
        tracker: PairTracker,
        pair_rels: Set[Union[RelationType, str]],
        frame_idx: int,
        timestamp: float,
    ) -> List[Event]:
        """
        Evaluate temporal state transitions for a single (subject, object) pair.
        Transitions move sequentially across frames (one stage transition per frame).
        """
        emitted: List[Event] = []

        # Cannot evaluate temporal transitions without at least two observations
        if subj.previous_centroid is None:
            return emitted

        dist_m = self.rel_calc.estimate_distance_m(subj, obj)
        scx, scy = subj.current_centroid
        ocx, ocy = obj.current_centroid
        dist_px = math.hypot(scx - ocx, scy - ocy)

        # Track stationary history of target object
        if obj.speed_px_s < self.config.pickup_stationary_speed_threshold_px_s:
            tracker.was_target_stationary = True

        # --------------------------------------------------------------
        # 1. APPROACH Transition: IDLE -> APPROACHING
        # --------------------------------------------------------------
        if tracker.stage == PairStage.IDLE:
            is_moving_toward = (
                RelationType.MOVING_TOWARD in pair_rels
                or self.rel_calc.check_moving_toward(subj, obj)
            )

            if is_moving_toward and dist_px <= self.config.effective_approach_max_start_distance_px:
                tracker.consecutive_approach_frames += 1
                if tracker.consecutive_approach_frames >= self.config.approach_min_frames:
                    if EventType.APPROACH not in tracker.events_emitted:
                        conf = self._calculate_confidence(
                            event_type=EventType.APPROACH,
                            subject=subj,
                            obj=obj,
                            dist_m=dist_m,
                            dist_px=dist_px,
                            persistence_count=tracker.consecutive_approach_frames,
                        )
                        ev = Evidence(
                            frame_index=frame_idx,
                            distance_m=dist_m,
                            position_change=subj.distance_px > 2.0,
                            synchronized_motion=False,
                            depth_delta_m=subj.depth_delta_m,
                        )
                        emitted.append(
                            Event(
                                event=EventType.APPROACH,
                                timestamp=timestamp,
                                subject=subj.id,
                                object=obj.id,
                                confidence=conf,
                                evidence=ev,
                            )
                        )
                        tracker.events_emitted.add(EventType.APPROACH)
                        tracker.stage = PairStage.APPROACHING
            else:
                tracker.consecutive_approach_frames = 0

        # --------------------------------------------------------------
        # 2. REACH Transition: APPROACHING -> REACHED
        # --------------------------------------------------------------
        elif tracker.stage == PairStage.APPROACHING:
            is_near = (
                RelationType.NEAR in pair_rels
                or self.rel_calc.check_near(subj, obj, dist_m)
                or dist_px <= self.config.effective_reach_proximity_threshold_px
            )

            if is_near:
                tracker.consecutive_reach_frames += 1
                if EventType.REACH not in tracker.events_emitted:
                    conf = self._calculate_confidence(
                        event_type=EventType.REACH,
                        subject=subj,
                        obj=obj,
                        dist_m=dist_m,
                        dist_px=dist_px,
                        persistence_count=tracker.consecutive_reach_frames,
                    )
                    ev = Evidence(
                        frame_index=frame_idx,
                        distance_m=dist_m,
                        position_change=subj.distance_px > 2.0,
                        synchronized_motion=False,
                        depth_delta_m=subj.depth_delta_m,
                    )
                    emitted.append(
                        Event(
                            event=EventType.REACH,
                            timestamp=timestamp,
                            subject=subj.id,
                            object=obj.id,
                            confidence=conf,
                            evidence=ev,
                        )
                    )
                    tracker.events_emitted.add(EventType.REACH)
                    tracker.stage = PairStage.REACHED

        # --------------------------------------------------------------
        # 3. PICK_UP Transition: REACHED -> PICKED_UP
        # --------------------------------------------------------------
        elif tracker.stage == PairStage.REACHED:
            close_contact = dist_px <= self.config.effective_pickup_contact_proximity_px
            if dist_m is not None:
                close_contact = close_contact and (dist_m <= self.config.pickup_contact_proximity_m)

            obj_now_moving = (
                obj.speed_px_s >= self.config.pickup_min_object_speed_px_s
                and obj.distance_px >= 2.0
            )
            subj_now_moving = (
                subj.speed_px_s >= self.config.pickup_min_subject_speed_px_s
                and subj.distance_px >= 2.0
            )

            sync_motion = False
            if obj_now_moving and subj_now_moving:
                sync_motion = self._check_motion_synchronization(
                    subj, obj, self.config.pickup_min_sync_cosine, self.config.effective_pickup_max_dist_drift_px
                )

            if (
                close_contact
                and tracker.was_target_stationary
                and obj_now_moving
                and subj_now_moving
                and sync_motion
            ):
                if EventType.PICK_UP not in tracker.events_emitted:
                    conf = self._calculate_confidence(
                        event_type=EventType.PICK_UP,
                        subject=subj,
                        obj=obj,
                        dist_m=dist_m,
                        dist_px=dist_px,
                        sync_motion=True,
                    )
                    ev = Evidence(
                        frame_index=frame_idx,
                        distance_m=dist_m,
                        position_change=True,
                        synchronized_motion=True,
                        depth_delta_m=subj.depth_delta_m,
                    )
                    emitted.append(
                        Event(
                            event=EventType.PICK_UP,
                            timestamp=timestamp,
                            subject=subj.id,
                            object=obj.id,
                            confidence=conf,
                            evidence=ev,
                        )
                    )
                    tracker.events_emitted.add(EventType.PICK_UP)
                    tracker.stage = PairStage.PICKED_UP
                    tracker.pickup_frame_index = frame_idx

        # --------------------------------------------------------------
        # 4. CARRY Transition: PICKED_UP -> CARRYING
        # --------------------------------------------------------------
        elif tracker.stage == PairStage.PICKED_UP:
            if tracker.pickup_frame_index is not None and frame_idx > tracker.pickup_frame_index:
                close_contact = dist_px <= self.config.effective_carry_proximity_px
                if dist_m is not None:
                    close_contact = close_contact and (dist_m <= self.config.carry_proximity_m)

                both_moving = (
                    subj.speed_px_s >= self.config.carry_min_speed_px_s
                    and obj.speed_px_s >= self.config.carry_min_speed_px_s
                )

                sync_motion = False
                if both_moving:
                    sync_motion = self._check_motion_synchronization(
                        subj, obj, self.config.carry_min_sync_cosine, self.config.effective_carry_max_dist_drift_px
                    )

                if close_contact and both_moving and sync_motion:
                    if EventType.CARRY not in tracker.events_emitted:
                        conf = self._calculate_confidence(
                            event_type=EventType.CARRY,
                            subject=subj,
                            obj=obj,
                            dist_m=dist_m,
                            dist_px=dist_px,
                            sync_motion=True,
                        )
                        ev = Evidence(
                            frame_index=frame_idx,
                            distance_m=dist_m,
                            position_change=True,
                            synchronized_motion=True,
                            depth_delta_m=subj.depth_delta_m,
                        )
                        emitted.append(
                            Event(
                                event=EventType.CARRY,
                                timestamp=timestamp,
                                subject=subj.id,
                                object=obj.id,
                                confidence=conf,
                                evidence=ev,
                            )
                        )
                        tracker.events_emitted.add(EventType.CARRY)
                        tracker.stage = PairStage.CARRYING

        return emitted

    # ------------------------------------------------------------------
    # Helper Computations
    # ------------------------------------------------------------------
    @staticmethod
    def _check_motion_synchronization(
        subj: ObjectTemporalState,
        obj: ObjectTemporalState,
        min_cosine: float,
        max_dist_drift_px: float,
    ) -> bool:
        """Check if both objects share velocity direction and rigid distance stability."""
        vx_s, vy_s = subj.velocity_px_s
        vx_o, vy_o = obj.velocity_px_s
        norm_s = math.hypot(vx_s, vy_s)
        norm_o = math.hypot(vx_o, vy_o)

        if norm_s <= 1e-6 or norm_o <= 1e-6:
            return False

        cosine = (vx_s * vx_o + vy_s * vy_o) / (norm_s * norm_o)
        if cosine < min_cosine:
            return False

        # Relative separation stability
        if subj.previous_centroid is not None and obj.previous_centroid is not None:
            curr_dx = float(subj.current_centroid[0] - obj.current_centroid[0])
            curr_dy = float(subj.current_centroid[1] - obj.current_centroid[1])
            prev_dx = float(subj.previous_centroid[0] - obj.previous_centroid[0])
            prev_dy = float(subj.previous_centroid[1] - obj.previous_centroid[1])
            drift = abs(math.hypot(curr_dx, curr_dy) - math.hypot(prev_dx, prev_dy))
            if drift > max_dist_drift_px:
                return False

        return True

    @staticmethod
    def _calculate_confidence(
        event_type: EventType,
        subject: ObjectTemporalState,
        obj: ObjectTemporalState,
        dist_m: Optional[float],
        dist_px: float,
        persistence_count: int = 1,
        sync_motion: bool = False,
    ) -> float:
        """
        Deterministic, explainable confidence calculation based on signal agreement.
        Base score: 0.65.
        Bonuses for:
          - High detector confidence of subject & object (+0.04 to +0.08)
          - Metric depth availability and consistency (+0.08)
          - Multi-frame persistence (+0.05)
          - Motion synchronization (+0.10)
        Clamped to [0.50, 0.98].
        """
        score = 0.65

        # Detector confidence weighting
        mean_det_conf = (subject.current_confidence + obj.current_confidence) / 2.0
        if mean_det_conf >= 0.85:
            score += 0.08
        elif mean_det_conf >= 0.70:
            score += 0.04

        # 3D Depth signal agreement
        if dist_m is not None:
            score += 0.08

        # Multi-frame persistence
        if persistence_count >= 2:
            score += 0.05

        # Motion synchronization (critical for PICK_UP and CARRY)
        if sync_motion:
            score += 0.10

        return round(min(0.98, max(0.50, score)), 2)

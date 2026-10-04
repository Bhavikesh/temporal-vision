"""
reasoning/engine.py
-------------------
Integrated Temporal Reasoning Engine (Member 2).

Orchestrates the complete Member 2 temporal reasoning pipeline:
  perception_output.json
          ↓
  PerceptionInputAdapter
          ↓
  ObjectStateManager
          ↓
  RelationCalculator
          ↓
  TemporalEventEngine
          ↓
  EvidenceCollector
          ↓
  EventExplainer
          ↓
  ExplanationOutput

Exposes clean, batch and stream-oriented APIs for downstream integration
(Member 3 backend). Zero perception model or external service dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .evidence import EvidenceCollector
from .explainer import EventExplainer
from .input_adapter import ParsedFrame, PerceptionInputAdapter, PerceptionValidationError
from .relations import RelationCalculator, RelationConfig
from .rules import EventRuleConfig, TemporalEventEngine
from .state import ObjectStateManager
from .types import Event, ExplanationOutput


class TemporalReasoningEngine:
    """
    Unified orchestrator for temporal state tracking, spatial relation evaluation,
    deterministic event detection, evidence collection, and natural-language narration.
    """

    def __init__(
        self,
        event_config: Optional[EventRuleConfig] = None,
        relation_config: Optional[RelationConfig] = None,
        max_history_per_object: Optional[int] = 100,
        frame_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.adapter = PerceptionInputAdapter()
        self.state_manager = ObjectStateManager(max_history_per_object=max_history_per_object)

        if frame_resolution is not None:
            relation_config = relation_config or RelationConfig.for_resolution(*frame_resolution)
            event_config = event_config or EventRuleConfig.for_resolution(*frame_resolution)

        self.rel_calc = RelationCalculator(relation_config)
        self.event_engine = TemporalEventEngine(event_config, relation_calculator=self.rel_calc)
        self.evidence_collector = EvidenceCollector(relation_calculator=self.rel_calc)
        self.explainer = EventExplainer()
        self._user_frame_resolution = frame_resolution
        self._auto_scaled = False

    def _apply_resolution(self, resolution: Tuple[int, int]) -> None:
        width, height = resolution
        scale = math.hypot(width, height) / math.hypot(1280, 720)
        self.rel_calc.config.scale_factor = scale
        self.rel_calc.config.frame_resolution = resolution
        self.event_engine.config.scale_factor = scale
        self.event_engine.config.frame_resolution = resolution

    def reset(self) -> None:
        """Reset internal tracking and event state across all components."""
        self.state_manager.reset()
        self.event_engine.reset()
        if self._user_frame_resolution is not None:
            self._apply_resolution(self._user_frame_resolution)
        elif self._auto_scaled:
            self.rel_calc.config.scale_factor = 1.0
            self.rel_calc.config.frame_resolution = None
            self.event_engine.config.scale_factor = 1.0
            self.event_engine.config.frame_resolution = None
            self._auto_scaled = False

    def process_frames(
        self,
        frames: Union[Sequence[Dict[str, Any]], str, Path, Sequence[ParsedFrame]],
        video_path: Optional[str] = None,
        processed_at: Optional[str] = None,
        frame_resolution: Optional[Tuple[int, int]] = None,
    ) -> ExplanationOutput:
        """
        Process a sequence of perception frames, sort chronologically, execute the
        reasoning pipeline, and produce the final ExplanationOutput contract.

        Parameters
        ----------
        frames : Union[Sequence[Dict[str, Any]], str, Path, Sequence[ParsedFrame]]
            Perception JSON list, JSON string, file path, or pre-parsed frames.
        video_path : Optional[str]
            Path to the source video (defaults to 'input/demo.mp4' or metadata).
        processed_at : Optional[str]
            ISO 8601 processing timestamp string (auto-generated if None).
        frame_resolution : Optional[Tuple[int, int]]
            Optional (width, height) of the video frames to scale pixel thresholds.

        Returns
        -------
        ExplanationOutput
            Complete result containing video_path, processed_at, events, and explanation.
        """
        # 1. Parse and validate frames via PerceptionInputAdapter
        if isinstance(frames, (str, Path)):
            parsed_frames = self.adapter.parse_sequence(frames)
        elif len(frames) > 0 and isinstance(frames[0], ParsedFrame):
            parsed_frames = list(frames)  # already parsed
        else:
            parsed_frames = [
                self.adapter.parse_frame(f) if not isinstance(f, ParsedFrame) else f
                for f in frames
            ]

        # 2. Ensure strictly chronological processing by frame_index
        parsed_frames.sort(key=lambda f: f.frame_index)

        # 3. Reset orchestrator state for fresh sequence run
        self.reset()

        # Apply explicit or auto-detected resolution scaling
        eff_res = frame_resolution or self._user_frame_resolution
        if eff_res is not None:
            self._apply_resolution(eff_res)
            self._auto_scaled = (frame_resolution is not None and frame_resolution != self._user_frame_resolution)
        elif (
            self.event_engine.config.frame_resolution is None
            and self.event_engine.config.scale_factor == 1.0
        ):
            max_x = max((o.bbox[2] for f in parsed_frames for o in f.objects), default=0)
            max_y = max((o.bbox[3] for f in parsed_frames for o in f.objects), default=0)
            if max_x > 2560 or max_y > 1440:
                self._apply_resolution((3840, 2160))
                self._auto_scaled = True
            elif max_x > 1920 or max_y > 1080:
                self._apply_resolution((2560, 1440))
                self._auto_scaled = True
            elif max_x > 1280 or max_y > 720:
                self._apply_resolution((1920, 1080))
                self._auto_scaled = True

        all_events: List[Event] = []

        # 4. Pipeline execution across frames
        for frame in parsed_frames:
            events_in_frame = self.process_frame(frame)
            all_events.extend(events_in_frame)

        # 5. Metadata handling
        effective_video_path = video_path or "input/demo.mp4"
        effective_processed_at = processed_at or datetime.now(timezone.utc).isoformat()

        # 6. Generate grounded narrative and package output
        return self.explainer.build_output(
            video_path=effective_video_path,
            processed_at=effective_processed_at,
            events=all_events,
        )

    def process_frame(
        self,
        frame: Union[Dict[str, Any], ParsedFrame],
    ) -> List[Event]:
        """
        Process a single perception frame in the stream:
          Adapter/Validation → StateManager → RelationCalculator → EventEngine → EvidenceCollector
        """
        parsed_frame = frame if isinstance(frame, ParsedFrame) else self.adapter.parse_frame(frame)

        # Update persistent temporal kinematics
        self.state_manager.update(parsed_frame)

        # Compute active spatial relations
        relations = self.rel_calc.compute(self.state_manager)

        # Detect temporal events
        raw_events = self.event_engine.process_frame(self.state_manager, relations=relations)

        # Enrich and validate evidence blocks
        enriched_events: List[Event] = []
        for event in raw_events:
            subj = self.state_manager.get_state(event.subject)
            obj = self.state_manager.get_state(event.object) if event.object else None

            if subj is not None:
                validated_ev = self.evidence_collector.collect(
                    event_type=event.event,
                    frame_index=event.evidence.frame_index,
                    subject=subj,
                    obj=obj,
                    synchronized_motion=event.evidence.synchronized_motion,
                    distance_m=event.evidence.distance_m,
                    depth_delta_m=event.evidence.depth_delta_m,
                    position_change=event.evidence.position_change,
                )
                event.evidence = validated_ev

            enriched_events.append(event)

        return enriched_events

    def process_json_file(
        self,
        json_path: Union[str, Path],
        video_path: Optional[str] = None,
        processed_at: Optional[str] = None,
    ) -> ExplanationOutput:
        """Convenience method to process an entire perception_output.json file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Perception output JSON file not found: {json_path}")
        return self.process_frames(
            frames=path,
            video_path=video_path or str(path),
            processed_at=processed_at,
        )

"""
reasoning/explainer.py
----------------------
Deterministic natural-language explanation engine (Member 2).

Generates grounded, human-readable explanations from already-detected Event and
Evidence objects:
  - Formats timestamps deterministically (e.g. 4.2s)
  - References observed evidence (frame numbers, distances, motion sync, position changes)
  - Strictly avoids inventing or hallucinating metrics when evidence is None
  - Produces ExplanationOutput data contracts conforming to docs/DATA_MODEL.md

Zero external LLM or network dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from .types import Event, EventType, Evidence, ExplanationOutput


class EventExplainer:
    """
    Deterministic explainer translating structured events and evidence
    into grounded natural language summaries.
    """

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format timestamp into a clean, human-readable seconds string.
        Examples: 4.2s, 0.0s, 12.5s
        """
        return f"{round(seconds, 1):.1f}s"

    def explain_event(self, event: Event, has_prior_pickup: bool = False) -> str:
        """
        Generate a concise, evidence-grounded explanation for a single event.

        Parameters
        ----------
        event : Event
            The detected event instance.
        has_prior_pickup : bool
            Whether a PICK_UP event on this target object was previously observed.

        Returns
        -------
        str
            A grounded natural-language sentence.
        """
        evt_type = event.event.value if hasattr(event.event, "value") else str(event.event)
        t_str = self.format_timestamp(event.timestamp)
        subj = event.subject
        obj = event.object or "target object"
        ev = event.evidence

        evidence_clauses: List[str] = []

        # 1. Frame reference
        if ev and ev.frame_index is not None:
            evidence_clauses.append(f"at frame {ev.frame_index}")

        # 2. Metric distance (only when non-None)
        if ev and ev.distance_m is not None:
            evidence_clauses.append(f"distance {ev.distance_m:.2f}m")

        # 3. Position change (only when non-None)
        if ev and ev.position_change is not None:
            if isinstance(ev.position_change, bool):
                if ev.position_change:
                    evidence_clauses.append("position change observed")
            elif isinstance(ev.position_change, (int, float)):
                evidence_clauses.append(f"displacement {ev.position_change:.1f}px")

        # 4. Synchronized motion (only when strictly True)
        if ev and ev.synchronized_motion is True:
            evidence_clauses.append("synchronized motion confirmed")

        ev_suffix = f" ({', '.join(evidence_clauses)})" if evidence_clauses else ""

        if evt_type == EventType.APPROACH.value:
            return f"At {t_str}, {subj} approached {obj}{ev_suffix}."
        elif evt_type == EventType.REACH.value:
            return f"At {t_str}, {subj} reached {obj}{ev_suffix}."
        elif evt_type == EventType.PICK_UP.value:
            return f"At {t_str}, {subj} picked up {obj}{ev_suffix}."
        elif evt_type == EventType.CARRY.value:
            if has_prior_pickup:
                return f"After picking up {obj}, {subj} carried it while moving together{ev_suffix}."
            return f"At {t_str}, {subj} carried {obj} while moving together{ev_suffix}."
        else:
            return f"At {t_str}, {subj} performed {evt_type} on {obj}{ev_suffix}."

    def explain_events(self, events: Sequence[Event]) -> str:
        """
        Generate a complete chronological natural-language summary over a sequence of events.
        Preserves original sequence; sorts by timestamp if events are out of order.
        Never invents intermediate missing events.
        """
        if not events:
            return "No events were detected in the video."

        # Preserve chronological ordering
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        sentences: List[str] = []
        prior_pickup_objects = set()

        for ev in sorted_events:
            evt_type = ev.event.value if hasattr(ev.event, "value") else str(ev.event)
            has_prior_pickup = ev.object in prior_pickup_objects if ev.object else False
            if evt_type == EventType.PICK_UP.value and ev.object:
                prior_pickup_objects.add(ev.object)

            sentence = self.explain_event(ev, has_prior_pickup=has_prior_pickup)
            sentences.append(sentence)

        return " ".join(sentences)

    def build_output(
        self,
        video_path: str,
        processed_at: str,
        events: Sequence[Event],
    ) -> ExplanationOutput:
        """
        Package detected events and natural-language narrative into an ExplanationOutput contract.
        """
        explanation_narrative = self.explain_events(events)
        return ExplanationOutput(
            video_path=video_path,
            processed_at=processed_at,
            events=list(events),
            explanation=explanation_narrative,
        )

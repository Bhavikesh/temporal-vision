#!/usr/bin/env python3
"""
scripts/test_reasoning_explainer.py
-------------------------------------
Unit tests for EventExplainer and natural language explanation generation (Member 2).
Covers all 18 required test points:
  1. APPROACH explanation
  2. REACH explanation
  3. PICK_UP explanation
  4. CARRY explanation
  5. Complete chronological event sequence
  6. Missing intermediate event (no hallucinated events)
  7. Evidence with distance_m
  8. Evidence with distance_m=None
  9. Evidence with synchronized_motion=True
 10. synchronized_motion=None
 11. Evidence with position_change
 12. Evidence with position_change=None
 13. Multiple events
 14. Empty event list
 15. Timestamp formatting
 16. ExplanationOutput construction
 17. Serialization compatibility
 18. No invented evidence claims
"""

import json
import os
import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.reasoning.explainer import EventExplainer
from backend.reasoning.types import Event, EventType, Evidence, ExplanationOutput


class TestEventExplainer(unittest.TestCase):
    def setUp(self):
        self.explainer = EventExplainer()

    def test_01_approach_explanation(self):
        ev = Evidence(frame_index=12, distance_m=1.2, position_change=True)
        event = Event(
            event=EventType.APPROACH,
            timestamp=4.2,
            subject="person_01",
            object="table_01",
            confidence=0.85,
            evidence=ev,
        )
        text = self.explainer.explain_event(event)
        self.assertIn("At 4.2s, person_01 approached table_01", text)
        self.assertIn("at frame 12", text)
        self.assertIn("distance 1.20m", text)

    def test_02_reach_explanation(self):
        ev = Evidence(frame_index=20, distance_m=0.8, position_change=True)
        event = Event(
            event=EventType.REACH,
            timestamp=5.1,
            subject="person_01",
            object="table_01",
            confidence=0.88,
            evidence=ev,
        )
        text = self.explainer.explain_event(event)
        self.assertIn("At 5.1s, person_01 reached table_01", text)
        self.assertIn("distance 0.80m", text)

    def test_03_pickup_explanation(self):
        ev = Evidence(frame_index=35, distance_m=0.42, position_change=True, synchronized_motion=True)
        event = Event(
            event=EventType.PICK_UP,
            timestamp=5.7,
            subject="person_01",
            object="laptop_01",
            confidence=0.94,
            evidence=ev,
        )
        text = self.explainer.explain_event(event)
        self.assertIn("At 5.7s, person_01 picked up laptop_01", text)
        self.assertIn("synchronized motion confirmed", text)
        self.assertIn("position change observed", text)

    def test_04_carry_explanation(self):
        ev = Evidence(frame_index=50, distance_m=0.45, position_change=True, synchronized_motion=True)
        event = Event(
            event=EventType.CARRY,
            timestamp=6.8,
            subject="person_01",
            object="laptop_01",
            confidence=0.92,
            evidence=ev,
        )
        # Without prior pickup
        text_standalone = self.explainer.explain_event(event, has_prior_pickup=False)
        self.assertIn("At 6.8s, person_01 carried laptop_01 while moving together", text_standalone)

        # With prior pickup
        text_with_pickup = self.explainer.explain_event(event, has_prior_pickup=True)
        self.assertIn("After picking up laptop_01, person_01 carried it while moving together", text_with_pickup)

    def test_05_complete_chronological_sequence(self):
        events = [
            Event(event=EventType.APPROACH, timestamp=2.0, subject="person_01", object="laptop_01", confidence=0.8, evidence=Evidence(frame_index=10)),
            Event(event=EventType.REACH, timestamp=3.0, subject="person_01", object="laptop_01", confidence=0.85, evidence=Evidence(frame_index=20)),
            Event(event=EventType.PICK_UP, timestamp=5.0, subject="person_01", object="laptop_01", confidence=0.9, evidence=Evidence(frame_index=30, synchronized_motion=True)),
            Event(event=EventType.CARRY, timestamp=8.0, subject="person_01", object="laptop_01", confidence=0.92, evidence=Evidence(frame_index=45, synchronized_motion=True)),
        ]
        summary = self.explainer.explain_events(events)
        self.assertIn("approached", summary)
        self.assertIn("reached", summary)
        self.assertIn("picked up", summary)
        self.assertIn("carried", summary)

        # Verify ordering
        idx_app = summary.index("approached")
        idx_rch = summary.index("reached")
        idx_pck = summary.index("picked up")
        idx_cry = summary.index("carried")
        self.assertTrue(idx_app < idx_rch < idx_pck < idx_cry)

    def test_06_missing_intermediate_event(self):
        # Sequence has only APPROACH and PICK_UP (no REACH)
        events = [
            Event(event=EventType.APPROACH, timestamp=2.0, subject="person_01", object="laptop_01", confidence=0.8, evidence=Evidence(frame_index=10)),
            Event(event=EventType.PICK_UP, timestamp=5.0, subject="person_01", object="laptop_01", confidence=0.9, evidence=Evidence(frame_index=30)),
        ]
        summary = self.explainer.explain_events(events)
        self.assertIn("approached", summary)
        self.assertIn("picked up", summary)
        self.assertNotIn("reached", summary)

    def test_07_evidence_with_distance_m(self):
        event = Event(
            event=EventType.APPROACH, timestamp=1.5, subject="p1", object="l1", confidence=0.8,
            evidence=Evidence(frame_index=5, distance_m=0.85)
        )
        text = self.explainer.explain_event(event)
        self.assertIn("distance 0.85m", text)

    def test_08_evidence_with_distance_m_none(self):
        event = Event(
            event=EventType.APPROACH, timestamp=1.5, subject="p1", object="l1", confidence=0.8,
            evidence=Evidence(frame_index=5, distance_m=None)
        )
        text = self.explainer.explain_event(event)
        self.assertNotIn("distance", text)
        self.assertNotIn("0.85m", text)
        self.assertNotIn("m)", text)

    def test_09_evidence_with_synchronized_motion_true(self):
        event = Event(
            event=EventType.PICK_UP, timestamp=5.0, subject="p1", object="l1", confidence=0.9,
            evidence=Evidence(frame_index=25, synchronized_motion=True)
        )
        text = self.explainer.explain_event(event)
        self.assertIn("synchronized motion confirmed", text)

    def test_10_synchronized_motion_none(self):
        event = Event(
            event=EventType.PICK_UP, timestamp=5.0, subject="p1", object="l1", confidence=0.9,
            evidence=Evidence(frame_index=25, synchronized_motion=None)
        )
        text = self.explainer.explain_event(event)
        self.assertNotIn("synchronized motion", text)

    def test_11_evidence_with_position_change(self):
        event = Event(
            event=EventType.PICK_UP, timestamp=5.0, subject="p1", object="l1", confidence=0.9,
            evidence=Evidence(frame_index=25, position_change=True)
        )
        text = self.explainer.explain_event(event)
        self.assertIn("position change observed", text)

    def test_12_evidence_with_position_change_none(self):
        event = Event(
            event=EventType.PICK_UP, timestamp=5.0, subject="p1", object="l1", confidence=0.9,
            evidence=Evidence(frame_index=25, position_change=None)
        )
        text = self.explainer.explain_event(event)
        self.assertNotIn("position change", text)
        self.assertNotIn("displacement", text)

    def test_13_multiple_events(self):
        # Unsorted input timestamps: explainer should sort chronologically
        events = [
            Event(event=EventType.PICK_UP, timestamp=5.0, subject="p1", object="l1", confidence=0.9, evidence=Evidence(frame_index=30)),
            Event(event=EventType.APPROACH, timestamp=2.0, subject="p1", object="l1", confidence=0.8, evidence=Evidence(frame_index=10)),
        ]
        summary = self.explainer.explain_events(events)
        idx_app = summary.index("approached")
        idx_pck = summary.index("picked up")
        self.assertLess(idx_app, idx_pck)

    def test_14_empty_event_list(self):
        summary = self.explainer.explain_events([])
        self.assertEqual(summary, "No events were detected in the video.")

    def test_15_timestamp_formatting(self):
        self.assertEqual(EventExplainer.format_timestamp(4.234), "4.2s")
        self.assertEqual(EventExplainer.format_timestamp(0.0), "0.0s")
        self.assertEqual(EventExplainer.format_timestamp(12.78), "12.8s")
        self.assertEqual(EventExplainer.format_timestamp(5.0), "5.0s")

    def test_16_explanation_output_construction(self):
        events = [
            Event(event=EventType.APPROACH, timestamp=1.0, subject="p1", object="l1", confidence=0.8, evidence=Evidence(frame_index=5))
        ]
        out = self.explainer.build_output(
            video_path="input/demo.mp4",
            processed_at="2026-10-04T14:30:00",
            events=events,
        )
        self.assertIsInstance(out, ExplanationOutput)
        self.assertEqual(out.video_path, "input/demo.mp4")
        self.assertEqual(out.processed_at, "2026-10-04T14:30:00")
        self.assertEqual(len(out.events), 1)
        self.assertEqual(out.events[0], events[0])
        self.assertIn("approached", out.explanation)

    def test_17_serialization_compatibility(self):
        events = [
            Event(event=EventType.APPROACH, timestamp=1.0, subject="p1", object="l1", confidence=0.8, evidence=Evidence(frame_index=5, distance_m=1.2))
        ]
        out = self.explainer.build_output(
            video_path="input/demo.mp4",
            processed_at="2026-10-04T14:30:00",
            events=events,
        )
        d = out.to_dict()
        self.assertIn("video_path", d)
        self.assertIn("processed_at", d)
        self.assertIn("events", d)
        self.assertIn("explanation", d)

        # JSON round-trip
        json_str = out.to_json()
        loaded_dict = json.loads(json_str)
        restored = ExplanationOutput.from_dict(loaded_dict)
        self.assertEqual(restored.video_path, out.video_path)
        self.assertEqual(restored.explanation, out.explanation)
        self.assertEqual(len(restored.events), 1)
        self.assertEqual(restored.events[0].event, EventType.APPROACH)

    def test_18_no_invented_evidence_claims(self):
        # Event with zero evidence fields (all None)
        event = Event(
            event=EventType.APPROACH,
            timestamp=3.0,
            subject="person_01",
            object="laptop_01",
            confidence=0.8,
            evidence=Evidence(frame_index=15),
        )
        text = self.explainer.explain_event(event)
        # Must only claim frame, never distance, sync, or position change
        self.assertIn("At 3.0s, person_01 approached laptop_01 (at frame 15).", text)
        self.assertNotIn("distance", text)
        self.assertNotIn("synchronized", text)
        self.assertNotIn("position change", text)


if __name__ == "__main__":
    unittest.main()

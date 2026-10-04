#!/usr/bin/env python3
"""
scripts/test_reasoning_evidence.py
------------------------------------
Unit tests for EvidenceCollector and validation rules (Member 2).
Covers all 18 required test points:
  1. APPROACH evidence
  2. REACH evidence
  3. PICK_UP evidence
  4. CARRY evidence
  5. null depth
  6. non-null depth
  7. position change
  8. synchronized motion true
  9. synchronized motion false
 10. missing/None synchronized motion
 11. signed depth delta
 12. invalid frame index
 13. invalid distance
 14. invalid position change
 15. invalid/non-finite numeric values
 16. Evidence.to_dict()/serialization compatibility
 17. Evidence produced from realistic ObjectTemporalState values
 18. No metric distance invented when depth is unavailable
"""

import math
import os
import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.reasoning.evidence import EvidenceCollector, EvidenceValidationError
from backend.reasoning.state import ObjectStateManager, ObjectTemporalState
from backend.reasoning.types import EventType, Evidence


class TestEvidenceCollector(unittest.TestCase):
    def setUp(self):
        self.collector = EvidenceCollector()
        self.mgr = ObjectStateManager()

    def _setup_realistic_states(self, depth: bool = True):
        d_p = 2.0 if depth else None
        d_l = 2.1 if depth else None
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [150, 200], "depth": d_p},
                {"id": "l1", "class": "laptop", "confidence": 0.92, "bbox": [250, 180, 350, 240], "centroid": [300, 210], "depth": d_l},
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [140, 100, 240, 300], "centroid": [190, 200], "depth": d_p + 0.1 if depth else None},
                {"id": "l1", "class": "laptop", "confidence": 0.92, "bbox": [250, 180, 350, 240], "centroid": [300, 210], "depth": d_l},
            ],
        }
        self.mgr.reset()
        self.mgr.update(f0)
        self.mgr.update(f1)
        return self.mgr.get_state("p1"), self.mgr.get_state("l1")

    def test_01_approach_evidence(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.APPROACH,
            frame_index=5,
            subject=p,
            obj=l,
        )
        self.assertEqual(ev.frame_index, 5)
        self.assertIsNotNone(ev.distance_m)
        self.assertTrue(ev.position_change)
        self.assertFalse(ev.synchronized_motion)
        self.assertAlmostEqual(ev.depth_delta_m, 0.1, places=4)

    def test_02_reach_evidence(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.REACH,
            frame_index=10,
            subject=p,
            obj=l,
        )
        self.assertEqual(ev.frame_index, 10)
        self.assertIsNotNone(ev.distance_m)
        self.assertFalse(ev.synchronized_motion)

    def test_03_pickup_evidence(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.PICK_UP,
            frame_index=15,
            subject=p,
            obj=l,
            synchronized_motion=True,
            position_change=True,
        )
        self.assertEqual(ev.frame_index, 15)
        self.assertTrue(ev.synchronized_motion)
        self.assertTrue(ev.position_change)

    def test_04_carry_evidence(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.CARRY,
            frame_index=20,
            subject=p,
            obj=l,
            synchronized_motion=True,
            position_change=True,
        )
        self.assertEqual(ev.frame_index, 20)
        self.assertTrue(ev.synchronized_motion)

    def test_05_null_depth(self):
        p, l = self._setup_realistic_states(depth=False)
        ev = self.collector.collect(
            event_type=EventType.APPROACH,
            frame_index=5,
            subject=p,
            obj=l,
        )
        self.assertIsNone(ev.distance_m)
        self.assertIsNone(ev.depth_delta_m)

    def test_06_non_null_depth(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.APPROACH,
            frame_index=5,
            subject=p,
            obj=l,
        )
        self.assertIsNotNone(ev.distance_m)
        self.assertGreater(ev.distance_m, 0.0)

    def test_07_position_change(self):
        ev_bool = self.collector.build_evidence(frame_index=0, position_change=True)
        self.assertTrue(ev_bool.position_change)

        ev_false = self.collector.build_evidence(frame_index=0, position_change=False)
        self.assertFalse(ev_false.position_change)

        ev_num = self.collector.build_evidence(frame_index=0, position_change=42.5)
        self.assertEqual(ev_num.position_change, 42.5)

    def test_08_synchronized_motion_true(self):
        ev = self.collector.build_evidence(frame_index=0, synchronized_motion=True)
        self.assertTrue(ev.synchronized_motion)

    def test_09_synchronized_motion_false(self):
        ev = self.collector.build_evidence(frame_index=0, synchronized_motion=False)
        self.assertFalse(ev.synchronized_motion)

    def test_10_missing_none_synchronized_motion(self):
        ev = self.collector.build_evidence(frame_index=0, synchronized_motion=None)
        self.assertIsNone(ev.synchronized_motion)

    def test_11_signed_depth_delta(self):
        # Positive depth delta (moving farther away)
        ev_pos = self.collector.build_evidence(frame_index=0, depth_delta_m=0.35)
        self.assertEqual(ev_pos.depth_delta_m, 0.35)

        # Negative depth delta (moving closer to camera)
        ev_neg = self.collector.build_evidence(frame_index=0, depth_delta_m=-0.50)
        self.assertEqual(ev_neg.depth_delta_m, -0.50)

        # Zero depth delta
        ev_zero = self.collector.build_evidence(frame_index=0, depth_delta_m=0.0)
        self.assertEqual(ev_zero.depth_delta_m, 0.0)

    def test_12_invalid_frame_index(self):
        # Negative frame index
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=-1)

        # Non-int
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index="0")

        # Bool disguised as int
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=True)

    def test_13_invalid_distance(self):
        # Negative distance
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, distance_m=-1.5)

        # Non-numeric
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, distance_m="close")

        # Bool distance
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, distance_m=True)

    def test_14_invalid_position_change(self):
        # String
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, position_change="moved")

        # List
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, position_change=[1, 2])

    def test_15_invalid_non_finite_numeric_values(self):
        # NaN distance
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, distance_m=float("nan"))

        # Inf distance
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, distance_m=float("inf"))

        # NaN depth delta
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, depth_delta_m=float("nan"))

        # Inf position change
        with self.assertRaises(EvidenceValidationError):
            self.collector.build_evidence(frame_index=0, position_change=float("-inf"))

    def test_16_serialization_compatibility(self):
        ev = self.collector.build_evidence(
            frame_index=156,
            distance_m=0.42,
            position_change=True,
            synchronized_motion=True,
            depth_delta_m=None,
        )
        d = ev.to_dict()
        self.assertEqual(d["frame_index"], 156)
        self.assertEqual(d["distance_m"], 0.42)
        self.assertTrue(d["position_change"])
        self.assertTrue(d["synchronized_motion"])
        self.assertIsNone(d["depth_delta_m"])

        # Round trip
        restored = Evidence.from_dict(d)
        self.assertEqual(restored, ev)

    def test_17_evidence_produced_from_realistic_states(self):
        p, l = self._setup_realistic_states(depth=True)
        ev = self.collector.collect(
            event_type=EventType.APPROACH,
            frame_index=5,
            subject=p,
            obj=l,
        )
        self.assertIsInstance(ev, Evidence)
        self.assertEqual(ev.frame_index, 5)
        self.assertIsNotNone(ev.distance_m)
        self.assertAlmostEqual(ev.depth_delta_m, 0.1, places=4)
        self.assertTrue(ev.position_change)

    def test_18_no_metric_distance_invented_when_depth_unavailable(self):
        p, l = self._setup_realistic_states(depth=False)
        # Even though 2D pixel centroids exist, distance_m must strictly remain None
        ev = self.collector.collect(
            event_type=EventType.APPROACH,
            frame_index=5,
            subject=p,
            obj=l,
        )
        self.assertIsNone(ev.distance_m)


if __name__ == "__main__":
    unittest.main()

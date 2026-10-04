#!/usr/bin/env python3
"""
scripts/test_reasoning_engine.py
----------------------------------
Integration tests for the complete TemporalReasoningEngine pipeline (Member 2).
Exercises the end-to-end flow:
  Perception JSON → Adapter → StateManager → Relations → Rules → Evidence → Explainer
Covers all 18 integration test requirements using realistic perception JSON contracts.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.reasoning.engine import TemporalReasoningEngine
from backend.reasoning.input_adapter import PerceptionValidationError
from backend.reasoning.types import EventType, ExplanationOutput


class TestTemporalReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TemporalReasoningEngine()

    def test_01_single_frame_person_table(self):
        """Single frame should process cleanly with 0 events emitted."""
        raw = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [150, 200], "mask_path": None, "depth": None},
                    {"id": "table_01", "class": "table", "confidence": 0.92, "bbox": [300, 250, 600, 500], "centroid": [450, 375], "mask_path": None, "depth": None},
                ],
            }
        ]
        out = self.engine.process_frames(raw)
        self.assertIsInstance(out, ExplanationOutput)
        self.assertEqual(len(out.events), 0)
        self.assertEqual(out.explanation, "No events were detected in the video.")

    def test_02_multiple_chronological_frames(self):
        raw = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [150, 200], "mask_path": None, "depth": None},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.88, "bbox": [380, 180, 440, 220], "centroid": [410, 200], "mask_path": None, "depth": None},
                ],
            },
            {
                "frame_index": 5,
                "timestamp": 0.5,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [150, 100, 250, 300], "centroid": [200, 200], "mask_path": None, "depth": None},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.88, "bbox": [380, 180, 440, 220], "centroid": [410, 200], "mask_path": None, "depth": None},
                ],
            },
        ]
        out = self.engine.process_frames(raw)
        self.assertGreater(len(out.events), 0)
        self.assertEqual(out.events[0].event, EventType.APPROACH)

    def test_03_out_of_order_frames_handled_correctly(self):
        """Unsorted frame indices in input JSON must be processed in chronological order."""
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [100, 300], "mask_path": None, "depth": None},
                {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 280, 350, 320], "centroid": [325, 300], "mask_path": None, "depth": None},
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [170, 100, 270, 300], "centroid": [170, 300], "mask_path": None, "depth": None},
                {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 280, 350, 320], "centroid": [325, 300], "mask_path": None, "depth": None},
            ],
        }
        # Pass out of order: f1 then f0
        out = self.engine.process_frames([f1, f0])
        # Engine internally sorts by frame_index -> correctly detects approach
        self.assertEqual(len(out.events), 1)
        self.assertEqual(out.events[0].event, EventType.APPROACH)

    def test_04_empty_sequence(self):
        out = self.engine.process_frames([])
        self.assertEqual(len(out.events), 0)
        self.assertEqual(out.explanation, "No events were detected in the video.")

    def test_05_frame_with_zero_objects(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": []},
            {"frame_index": 5, "timestamp": 0.5, "objects": []},
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual(len(out.events), 0)

    def test_06_persistent_object_ids_across_frames(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [{"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [10, 10, 30, 30], "centroid": [20, 20], "mask_path": None, "depth": None}]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [{"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [10, 10, 30, 30], "centroid": [20, 20], "mask_path": None, "depth": None}]},
        ]
        self.engine.process_frames(raw)
        st = self.engine.state_manager.get_state("laptop_01")
        self.assertIsNotNone(st)
        self.assertEqual(st.id, "laptop_01")
        self.assertEqual(st.total_observations, 2)

    def test_07_depth_null_throughout(self):
        raw = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [100, 200], "mask_path": None, "depth": None},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 180, 350, 220], "centroid": [325, 200], "mask_path": None, "depth": None},
                ],
            },
            {
                "frame_index": 5,
                "timestamp": 0.5,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [180, 100, 280, 300], "centroid": [180, 200], "mask_path": None, "depth": None},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 180, 350, 220], "centroid": [325, 200], "mask_path": None, "depth": None},
                ],
            },
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual(len(out.events), 1)
        self.assertEqual(out.events[0].event, EventType.APPROACH)
        self.assertIsNone(out.events[0].evidence.distance_m)

    def test_08_non_null_depth(self):
        raw = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [100, 200], "mask_path": None, "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 180, 350, 220], "centroid": [325, 200], "mask_path": None, "depth": 2.05},
                ],
            },
            {
                "frame_index": 5,
                "timestamp": 0.5,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [180, 100, 280, 300], "centroid": [180, 200], "mask_path": None, "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [300, 180, 350, 220], "centroid": [325, 200], "mask_path": None, "depth": 2.05},
                ],
            },
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual(len(out.events), 1)
        self.assertIsNotNone(out.events[0].evidence.distance_m)
        self.assertGreater(out.events[0].evidence.distance_m, 0.0)

    def test_09_complete_sequence_realistic_perception_json(self):
        """
        Complete end-to-end integration test from realistic perception JSON:
        APPROACH -> REACH -> PICK_UP -> CARRY.
        """
        raw = [
            # Frame 0: Person far away, laptop resting stationary
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [50, 100, 150, 350], "centroid": [100, 250], "mask_path": "masks/f0/p.png", "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [280, 220, 340, 280], "centroid": [310, 250], "mask_path": "masks/f0/l.png", "depth": 2.0},
                ],
            },
            # Frame 1: Person moves toward laptop -> APPROACH
            {
                "frame_index": 5,
                "timestamp": 0.5,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [120, 100, 220, 350], "centroid": [170, 250], "mask_path": "masks/f5/p.png", "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [280, 220, 340, 280], "centroid": [310, 250], "mask_path": "masks/f5/l.png", "depth": 2.0},
                ],
            },
            # Frame 2: Person arrives at laptop -> REACH
            {
                "frame_index": 10,
                "timestamp": 1.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [190, 100, 290, 350], "centroid": [240, 250], "mask_path": "masks/f10/p.png", "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [280, 220, 340, 280], "centroid": [310, 250], "mask_path": "masks/f10/l.png", "depth": 2.0},
                ],
            },
            # Frame 3: Both move together in synchronized velocity -> PICK_UP
            {
                "frame_index": 15,
                "timestamp": 1.5,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [230, 100, 330, 350], "centroid": [280, 250], "mask_path": "masks/f15/p.png", "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [320, 220, 380, 280], "centroid": [350, 250], "mask_path": "masks/f15/l.png", "depth": 2.0},
                ],
            },
            # Frame 4: Sustained synchronized motion -> CARRY
            {
                "frame_index": 20,
                "timestamp": 2.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [270, 100, 370, 350], "centroid": [320, 250], "mask_path": "masks/f20/p.png", "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [360, 220, 420, 280], "centroid": [390, 250], "mask_path": "masks/f20/l.png", "depth": 2.0},
                ],
            },
        ]
        out = self.engine.process_frames(raw, video_path="demo.mp4")
        self.assertEqual(len(out.events), 4)
        types = [e.event for e in out.events]
        self.assertEqual(types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP, EventType.CARRY])
        self.assertIn("approached", out.explanation)
        self.assertIn("reached", out.explanation)
        self.assertIn("picked up", out.explanation)
        self.assertIn("carried", out.explanation)

    def test_10_approach_without_pickup(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 10, 90, 50], "centroid": [70, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual([e.event for e in out.events], [EventType.APPROACH])

    def test_11_reach_without_pickup(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 10, 90, 50], "centroid": [70, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [90, 10, 130, 50], "centroid": [110, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        types = [e.event for e in out.events]
        self.assertEqual(types, [EventType.APPROACH, EventType.REACH])

    def test_12_independent_object_movement_no_pickup(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [300, 300, 350, 350], "centroid": [325, 325], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                # Laptop shifts far away on its own
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [350, 300, 400, 350], "centroid": [375, 325], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual(len(out.events), 0)

    def test_13_duplicate_events_not_emitted(self):
        """Continuous approach frames should yield exactly one APPROACH event."""
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "centroid": [5, 5], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [300, 0, 320, 20], "centroid": [310, 10], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 0, 60, 10], "centroid": [55, 5], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [300, 0, 320, 20], "centroid": [310, 10], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [100, 0, 110, 10], "centroid": [105, 5], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [300, 0, 320, 20], "centroid": [310, 10], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        app_events = [e for e in out.events if e.event == EventType.APPROACH]
        self.assertEqual(len(app_events), 1)

    def test_14_disappearing_reappearing_object(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "centroid": [5, 5], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [200, 0, 220, 20], "centroid": [210, 10], "mask_path": None, "depth": None},
            ]},
            # Laptop missing in frame 5
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 0, 60, 10], "centroid": [55, 5], "mask_path": None, "depth": None},
            ]},
            # Laptop reappears in frame 10
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [100, 0, 110, 10], "centroid": [105, 5], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [200, 0, 220, 20], "centroid": [210, 10], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        self.assertIsInstance(out, ExplanationOutput)

    def test_15_malformed_input_propagates_validation_error(self):
        bad_json = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 1.5, "bbox": [0, 0, 10, 10], "centroid": [5, 5]}  # confidence > 1.0
            ]}
        ]
        with self.assertRaises(PerceptionValidationError):
            self.engine.process_frames(bad_json)

    def test_16_final_output_contains_structured_events(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 10, 90, 50], "centroid": [70, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw, video_path="demo.mp4")
        self.assertEqual(out.video_path, "demo.mp4")
        self.assertIsNotNone(out.processed_at)
        self.assertEqual(len(out.events), 1)
        self.assertEqual(out.events[0].subject, "p1")
        self.assertEqual(out.events[0].object, "l1")
        self.assertIsNotNone(out.events[0].evidence)

    def test_17_explanation_generated_from_same_events(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 10, 90, 50], "centroid": [70, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        self.assertIn("p1 approached l1", out.explanation)

    def test_18_no_invented_metric_distance_when_depth_null(self):
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [50, 10, 90, 50], "centroid": [70, 30], "mask_path": None, "depth": None},
                {"id": "l1", "class": "laptop", "confidence": 0.9, "bbox": [150, 10, 180, 40], "centroid": [165, 30], "mask_path": None, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        for ev in out.events:
            self.assertIsNone(ev.evidence.distance_m)
            self.assertIsNone(ev.evidence.depth_delta_m)

    def test_19_process_json_file_convenience(self):
        """Test process_json_file reading from an actual JSON file on disk."""
        data = [
            {"frame_index": 0, "timestamp": 0.0, "objects": []},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            out = self.engine.process_json_file(temp_path)
            self.assertIsInstance(out, ExplanationOutput)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_20_4k_person_laptop_interaction_auto_scaling(self):
        """
        Verify that 4K perception JSON (with coordinate bounds > 2500x1500)
        auto-scales proximity thresholds so person-laptop contact (distance ~320px)
        correctly triggers the complete event sequence: APPROACH, REACH, PICK_UP, CARRY.
        """
        raw = [
            # Frame 0: Approach start (distance 700px)
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [900, 500, 1100, 1500], "centroid": [1000, 1000], "confidence": 0.9, "depth": None},
                {"id": "l1", "class": "laptop", "bbox": [1500, 900, 1900, 1100], "centroid": [1700, 1000], "confidence": 0.9, "depth": None},
                {"id": "t1", "class": "table", "bbox": [0, 1800, 3839, 2159], "centroid": [1000, 2000], "confidence": 0.9, "depth": None},
            ]},
            # Frame 1: Approach step (distance 500px)
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "bbox": [1100, 500, 1300, 1500], "centroid": [1200, 1000], "confidence": 0.9, "depth": None},
                {"id": "l1", "class": "laptop", "bbox": [1500, 900, 1900, 1100], "centroid": [1700, 1000], "confidence": 0.9, "depth": None},
                {"id": "t1", "class": "table", "bbox": [0, 1800, 3839, 2159], "centroid": [1000, 2000], "confidence": 0.9, "depth": None},
            ]},
            # Frame 2: Contact reach (distance 330px <= 420px scaled threshold)
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [1270, 500, 1470, 1500], "centroid": [1370, 1000], "confidence": 0.9, "depth": None},
                {"id": "l1", "class": "laptop", "bbox": [1500, 900, 1900, 1100], "centroid": [1700, 1000], "confidence": 0.9, "depth": None},
                {"id": "t1", "class": "table", "bbox": [0, 1800, 3839, 2159], "centroid": [1000, 2000], "confidence": 0.9, "depth": None},
            ]},
            # Frame 3: Both move in synchronized velocity -> PICK_UP
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "p1", "class": "person", "bbox": [1370, 500, 1570, 1500], "centroid": [1470, 1000], "confidence": 0.9, "depth": None},
                {"id": "l1", "class": "laptop", "bbox": [1600, 900, 2000, 1100], "centroid": [1800, 1000], "confidence": 0.9, "depth": None},
                {"id": "t1", "class": "table", "bbox": [0, 1800, 3839, 2159], "centroid": [1000, 2000], "confidence": 0.9, "depth": None},
            ]},
            # Frame 4: Sustained synchronized motion -> CARRY
            {"frame_index": 20, "timestamp": 2.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [1470, 500, 1670, 1500], "centroid": [1570, 1000], "confidence": 0.9, "depth": None},
                {"id": "l1", "class": "laptop", "bbox": [1700, 900, 2100, 1100], "centroid": [1900, 1000], "confidence": 0.9, "depth": None},
                {"id": "t1", "class": "table", "bbox": [0, 1800, 3839, 2159], "centroid": [1000, 2000], "confidence": 0.9, "depth": None},
            ]},
        ]
        out = self.engine.process_frames(raw)
        event_types = [e.event for e in out.events]
        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP, EventType.CARRY])
        # All events must have person as subject
        for ev in out.events:
            self.assertEqual(ev.subject, "p1")
            self.assertEqual(ev.object, "l1")

    def test_21_active_subject_filtering_integration(self):
        """Verify that passive objects (e.g. table, laptop) never emit action events."""
        raw = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "t1", "class": "table", "centroid": [500, 500], "bbox": [400, 400, 600, 600], "confidence": 0.9},
                {"id": "l1", "class": "laptop", "centroid": [100, 100], "bbox": [80, 80, 120, 120], "confidence": 0.9},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "t1", "class": "table", "centroid": [500, 500], "bbox": [400, 400, 600, 600], "confidence": 0.9},
                {"id": "l1", "class": "laptop", "centroid": [300, 300], "bbox": [280, 280, 320, 320], "confidence": 0.9},
            ]},
        ]
        out = self.engine.process_frames(raw)
        self.assertEqual(len(out.events), 0)

    def test_22_resolution_scaling_override_and_reset_isolation(self):
        """
        Verify:
          1. Explicit frame_resolution overrides auto-detection.
          2. Auto-scaling does not leak state across calls on the same engine instance.
          3. 1080p coordinate bounds correctly select 1.5x scaling.
        """
        engine = TemporalReasoningEngine()

        # Call 1: 4K coordinates -> should auto-scale to 3.0x
        frames_4k = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [0, 0, 3840, 2160], "centroid": [1920, 1080], "confidence": 0.9}
            ]}
        ]
        engine.process_frames(frames_4k)
        self.assertAlmostEqual(engine.event_engine.config.scale_factor, 3.0, places=2)

        # Call 2: Standard 720p coordinates -> reset should restore base scale 1.0x
        frames_720p = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [0, 0, 1280, 720], "centroid": [640, 360], "confidence": 0.9}
            ]}
        ]
        engine.process_frames(frames_720p)
        self.assertAlmostEqual(engine.event_engine.config.scale_factor, 1.0, places=2)

        # Call 3: Explicit frame_resolution override
        engine.process_frames(frames_720p, frame_resolution=(1920, 1080))
        self.assertAlmostEqual(engine.event_engine.config.scale_factor, 1.5, places=2)

        # Call 4: 1080p bounds without explicit resolution -> auto-scales to 1.5x
        frames_1080p = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "bbox": [0, 0, 1920, 1080], "centroid": [960, 540], "confidence": 0.9}
            ]}
        ]
        engine.process_frames(frames_1080p)
        self.assertAlmostEqual(engine.event_engine.config.scale_factor, 1.5, places=2)


if __name__ == "__main__":
    unittest.main()

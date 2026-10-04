#!/usr/bin/env python3
"""
scripts/test_reasoning_state.py
---------------------------------
Unit tests for the ObjectStateManager and temporal state calculations (Member 2).
Uses mock/synthetic perception frames to test state transitions, displacement,
velocity, reappearance, null depth, and boundary conditions without requiring
actual perception model weights.
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

from backend.reasoning.state import (
    ObjectObservation,
    ObjectStateManager,
    ObjectTemporalState,
)


class TestObjectStateManager(unittest.TestCase):
    def setUp(self):
        self.mgr = ObjectStateManager(max_history_per_object=10)

    def test_01_new_object_initialization(self):
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {
                    "id": "person_01",
                    "class": "person",
                    "confidence": 0.95,
                    "bbox": [100, 100, 200, 300],
                    "centroid": [150, 200],
                    "mask_path": "masks/frame_00000/person_01.png",
                    "depth": 2.5,
                }
            ],
        }
        states = self.mgr.update(frame)
        self.assertIn("person_01", states)
        st = self.mgr.get_state("person_01")
        self.assertIsNotNone(st)
        self.assertEqual(st.id, "person_01")
        self.assertEqual(st.cls, "person")
        self.assertEqual(st.current_centroid, (150, 200))
        self.assertEqual(st.current_bbox, (100, 100, 200, 300))
        self.assertEqual(st.current_depth, 2.5)
        self.assertTrue(st.is_present)
        self.assertEqual(st.total_observations, 1)
        self.assertEqual(st.consecutive_missing_frames, 0)
        self.assertIsNone(st.previous_centroid)
        self.assertEqual(st.distance_px, 0.0)
        self.assertEqual(st.displacement_px, (0.0, 0.0))
        self.assertEqual(st.speed_px_s, 0.0)

    def test_02_position_update_across_frames(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30]}
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "laptop_01", "class": "laptop", "confidence": 0.92, "bbox": [15, 15, 55, 55], "centroid": [35, 35]}
            ],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        st = self.mgr.get_state("laptop_01")
        self.assertEqual(st.previous_frame_index, 0)
        self.assertEqual(st.previous_timestamp, 0.0)
        self.assertEqual(st.previous_centroid, (30, 30))
        self.assertEqual(st.current_frame_index, 5)
        self.assertEqual(st.current_timestamp, 0.5)
        self.assertEqual(st.current_centroid, (35, 35))
        self.assertEqual(st.total_observations, 2)

    def test_03_movement_and_displacement_calculation(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [{"id": "person_01", "class": "person", "centroid": [100, 100]}],
        }
        f1 = {
            "frame_index": 1,
            "timestamp": 1.0,
            "objects": [{"id": "person_01", "class": "person", "centroid": [103, 104]}],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        st = self.mgr.get_state("person_01")
        # dx = 3, dy = 4 -> dist = 5
        self.assertEqual(st.displacement_px, (3.0, 4.0))
        self.assertAlmostEqual(st.distance_px, 5.0, places=4)
        self.assertAlmostEqual(st.direction_normalized[0], 0.6, places=4)
        self.assertAlmostEqual(st.direction_normalized[1], 0.8, places=4)

    def test_04_velocity_calculation(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 1.0,
            "objects": [{"id": "person_01", "class": "person", "centroid": [0, 0]}],
        }
        f1 = {
            "frame_index": 10,
            "timestamp": 1.5,  # dt = 0.5s
            "objects": [{"id": "person_01", "class": "person", "centroid": [30, 40]}],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        st = self.mgr.get_state("person_01")
        self.assertAlmostEqual(st.elapsed_time_s, 0.5, places=4)
        self.assertAlmostEqual(st.distance_px, 50.0, places=4)
        # vx = 30 / 0.5 = 60, vy = 40 / 0.5 = 80, speed = 50 / 0.5 = 100
        self.assertAlmostEqual(st.velocity_px_s[0], 60.0, places=4)
        self.assertAlmostEqual(st.velocity_px_s[1], 80.0, places=4)
        self.assertAlmostEqual(st.speed_px_s, 100.0, places=4)

    def test_05_multiple_objects(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [10, 10]},
                {"id": "laptop_01", "class": "laptop", "centroid": [50, 50]},
                {"id": "table_01", "class": "table", "centroid": [50, 60]},
            ],
        }
        self.mgr.update(f0)
        self.assertEqual(len(self.mgr.tracked_ids), 3)
        self.assertEqual(set(self.mgr.present_ids), {"person_01", "laptop_01", "table_01"})
        hist_p = self.mgr.get_history("person_01")
        hist_l = self.mgr.get_history("laptop_01")
        self.assertEqual(len(hist_p), 1)
        self.assertEqual(len(hist_l), 1)

    def test_06_missing_object_in_intermediate_frame(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [10, 10]},
                {"id": "laptop_01", "class": "laptop", "centroid": [50, 50]},
            ],
        }
        f1 = {
            "frame_index": 1,
            "timestamp": 0.1,
            "objects": [
                # laptop_01 is missing here
                {"id": "person_01", "class": "person", "centroid": [12, 10]},
            ],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        st_p = self.mgr.get_state("person_01")
        st_l = self.mgr.get_state("laptop_01")
        self.assertTrue(st_p.is_present)
        self.assertFalse(st_l.is_present)
        self.assertEqual(st_l.consecutive_missing_frames, 1)
        self.assertEqual(st_l.speed_px_s, 0.0)
        self.assertEqual(st_l.displacement_px, (0.0, 0.0))
        self.assertEqual(self.mgr.present_ids, ["person_01"])
        self.assertEqual(set(self.mgr.tracked_ids), {"person_01", "laptop_01"})

    def test_07_object_reappearing(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [{"id": "laptop_01", "class": "laptop", "centroid": [50, 50]}],
        }
        f1 = {
            "frame_index": 1,
            "timestamp": 0.1,
            "objects": [],  # missing in frame 1
        }
        f2 = {
            "frame_index": 2,
            "timestamp": 0.2,
            "objects": [{"id": "laptop_01", "class": "laptop", "centroid": [60, 50]}],  # reappears
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        self.mgr.update(f2)
        st = self.mgr.get_state("laptop_01")
        self.assertTrue(st.is_present)
        self.assertEqual(st.consecutive_missing_frames, 0)
        self.assertEqual(st.total_observations, 2)
        self.assertEqual(st.previous_frame_index, 0)
        self.assertEqual(st.previous_centroid, (50, 50))
        self.assertEqual(st.current_centroid, (60, 50))
        self.assertEqual(st.displacement_px, (10.0, 0.0))
        self.assertAlmostEqual(st.distance_px, 10.0, places=4)
        self.assertAlmostEqual(st.speed_px_s, 50.0, places=4)  # 10 px in 0.2s = 50 px/s

    def test_08_null_depth(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [{"id": "table_01", "class": "table", "centroid": [100, 100], "depth": None}],
        }
        f1 = {
            "frame_index": 1,
            "timestamp": 0.1,
            "objects": [{"id": "table_01", "class": "table", "centroid": [100, 100], "depth": None}],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        st = self.mgr.get_state("table_01")
        self.assertIsNone(st.current_depth)
        self.assertIsNone(st.previous_depth)
        self.assertIsNone(st.depth_delta_m)

        # Transition from None to float depth
        f2 = {
            "frame_index": 2,
            "timestamp": 0.2,
            "objects": [{"id": "table_01", "class": "table", "centroid": [100, 100], "depth": 1.8}],
        }
        self.mgr.update(f2)
        st = self.mgr.get_state("table_01")
        self.assertEqual(st.current_depth, 1.8)
        self.assertIsNone(st.previous_depth)
        self.assertIsNone(st.depth_delta_m)

        # Transition from float depth to float depth
        f3 = {
            "frame_index": 3,
            "timestamp": 0.3,
            "objects": [{"id": "table_01", "class": "table", "centroid": [100, 100], "depth": 1.6}],
        }
        self.mgr.update(f3)
        st = self.mgr.get_state("table_01")
        self.assertAlmostEqual(st.depth_delta_m, -0.2, places=4)

    def test_09_zero_or_small_timestamp_difference(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 1.0,
            "objects": [{"id": "p1", "class": "person", "centroid": [10, 10]}],
        }
        f1 = {
            "frame_index": 1,
            "timestamp": 1.0,  # identical timestamp
            "objects": [{"id": "p1", "class": "person", "centroid": [20, 20]}],
        }
        self.mgr.update(f0)
        # Should not raise ZeroDivisionError
        self.mgr.update(f1)
        st = self.mgr.get_state("p1")
        self.assertEqual(st.elapsed_time_s, 0.0)
        self.assertEqual(st.speed_px_s, 0.0)
        self.assertEqual(st.velocity_px_s, (0.0, 0.0))

    def test_10_reset_behavior(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [{"id": "p1", "class": "person", "centroid": [10, 10]}],
        }
        self.mgr.update(f0)
        self.assertEqual(len(self.mgr.tracked_ids), 1)
        self.mgr.reset()
        self.assertEqual(len(self.mgr.tracked_ids), 0)
        self.assertIsNone(self.mgr.get_state("p1"))
        self.assertEqual(self.mgr.get_history("p1"), [])
        self.assertIsNone(self.mgr.current_frame_index)
        self.assertIsNone(self.mgr.current_timestamp)


if __name__ == "__main__":
    unittest.main()

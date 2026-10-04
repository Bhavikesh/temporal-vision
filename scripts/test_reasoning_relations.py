#!/usr/bin/env python3
"""
scripts/test_reasoning_relations.py
------------------------------------
Unit tests for RelationCalculator and spatial relation logic (Member 2).
Tests 'near', 'on', 'moving-toward', 'moving-away', and 'holding' under both
positive and negative (false-positive rejection) conditions.
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

from backend.reasoning.relations import RelationCalculator, RelationConfig
from backend.reasoning.state import ObjectStateManager, ObjectTemporalState
from backend.reasoning.types import RelationType, SpatialRelation


class TestRelationCalculator(unittest.TestCase):
    def setUp(self):
        self.config = RelationConfig(
            proximity_threshold_px=150.0,
            proximity_threshold_m=1.5,
            min_motion_distance_px=2.0,
            min_motion_speed_px_s=5.0,
            holding_proximity_px=120.0,
            holding_min_speed_px_s=8.0,
        )
        self.calc = RelationCalculator(self.config)
        self.mgr = ObjectStateManager()

    def test_01_person_near_laptop(self):
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 200], "depth": 2.0},
                {"id": "laptop_01", "class": "laptop", "centroid": [250, 200], "depth": 2.1},
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        near_rels = [r for r in relations if r.relation == RelationType.NEAR and r.subject_id == "person_01" and r.object_id == "laptop_01"]
        self.assertEqual(len(near_rels), 1)
        self.assertIsNotNone(near_rels[0].distance_m)
        self.assertLess(near_rels[0].distance_m, 1.5)

    def test_02_person_far_from_laptop(self):
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [100, 100], "depth": 2.0},
                {"id": "laptop_01", "class": "laptop", "centroid": [600, 600], "depth": 2.0},
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        near_rels = [r for r in relations if r.relation == RelationType.NEAR]
        self.assertEqual(len(near_rels), 0)

    def test_03_person_approaching_laptop_multiple_frames(self):
        # Frame 0: Person at x=100
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [100, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [400, 300]},
            ],
        }
        # Frame 1: Person moves closer to x=150 (dx=50 in 0.5s = 100 px/s)
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [150, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [400, 300]},
            ],
        }
        self.mgr.update(f0)
        rels_f0 = self.calc.compute(self.mgr)
        # In frame 0, no previous history -> moving-toward should be False
        self.assertFalse(any(r.relation == RelationType.MOVING_TOWARD for r in rels_f0))

        self.mgr.update(f1)
        rels_f1 = self.calc.compute(self.mgr)
        moving_toward = [
            r for r in rels_f1
            if r.relation == RelationType.MOVING_TOWARD
            and r.subject_id == "person_01"
            and r.object_id == "laptop_01"
        ]
        self.assertEqual(len(moving_toward), 1)

    def test_04_person_moving_away_from_laptop(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [250, 300]},
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                # Person walks left, away from laptop
                {"id": "person_01", "class": "person", "centroid": [120, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [250, 300]},
            ],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        relations = self.calc.compute(self.mgr)
        moving_away = [
            r for r in relations
            if r.relation == RelationType.MOVING_AWAY
            and r.subject_id == "person_01"
            and r.object_id == "laptop_01"
        ]
        self.assertEqual(len(moving_away), 1)
        # Should NOT be moving-toward
        self.assertFalse(any(r.relation == RelationType.MOVING_TOWARD for r in relations))

    def test_05_laptop_on_table(self):
        # Table bbox [100, 300, 500, 600], centroid [300, 450]
        # Laptop resting on tabletop: bbox [220, 250, 340, 320], centroid [280, 285]
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {
                    "id": "table_01",
                    "class": "table",
                    "bbox": [100, 300, 500, 600],
                    "centroid": [300, 450],
                    "depth": 2.0,
                },
                {
                    "id": "laptop_01",
                    "class": "laptop",
                    "bbox": [220, 250, 340, 320],
                    "centroid": [280, 285],
                    "depth": 2.05,
                },
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        on_rels = [
            r for r in relations
            if r.relation == RelationType.ON
            and r.subject_id == "laptop_01"
            and r.object_id == "table_01"
        ]
        self.assertEqual(len(on_rels), 1)
        # Table on laptop should NOT exist
        reverse_on = [
            r for r in relations
            if r.relation == RelationType.ON
            and r.subject_id == "table_01"
            and r.object_id == "laptop_01"
        ]
        self.assertEqual(len(reverse_on), 0)

    def test_06_laptop_not_on_table(self):
        # Laptop horizontally disjoint from table
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "table_01", "class": "table", "bbox": [100, 300, 500, 600], "centroid": [300, 450]},
                {"id": "laptop_01", "class": "laptop", "bbox": [600, 250, 700, 320], "centroid": [650, 285]},
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        self.assertFalse(any(r.relation == RelationType.ON for r in relations))

    def test_07_holding_synchronized_movement(self):
        # Person and laptop both move together in frame 1
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [220, 300]},
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                # Both shift +40 px in x, +10 px in y
                {"id": "person_01", "class": "person", "centroid": [240, 310]},
                {"id": "laptop_01", "class": "laptop", "centroid": [260, 310]},
            ],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        relations = self.calc.compute(self.mgr)
        holding_rels = [
            r for r in relations
            if r.relation == RelationType.HOLDING
            and r.subject_id == "person_01"
            and r.object_id == "laptop_01"
        ]
        self.assertEqual(len(holding_rels), 1)

    def test_08_close_objects_not_holding(self):
        # Person moves, but laptop is stationary on table (NOT holding)
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [220, 300]},
            ],
        }
        f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [240, 310]},
                # Laptop remains stationary
                {"id": "laptop_01", "class": "laptop", "centroid": [220, 300]},
            ],
        }
        self.mgr.update(f0)
        self.mgr.update(f1)
        relations = self.calc.compute(self.mgr)
        self.assertFalse(any(r.relation == RelationType.HOLDING for r in relations))

        # Diverging motion (both move in opposite directions)
        f2 = {
            "frame_index": 10,
            "timestamp": 1.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [260, 300]},
            ],
        }
        self.mgr.update(f2)
        relations_div = self.calc.compute(self.mgr)
        self.assertFalse(any(r.relation == RelationType.HOLDING for r in relations_div))

    def test_09_null_depth_safety(self):
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 200], "depth": None},
                {"id": "laptop_01", "class": "laptop", "centroid": [240, 200], "depth": None},
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        near_rels = [r for r in relations if r.relation == RelationType.NEAR and r.subject_id == "person_01"]
        self.assertEqual(len(near_rels), 1)
        # distance_m must be None when depth is unavailable
        self.assertIsNone(near_rels[0].distance_m)

    def test_10_missing_previous_observation(self):
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 200]},
                {"id": "laptop_01", "class": "laptop", "centroid": [250, 200]},
            ],
        }
        self.mgr.update(frame)
        relations = self.calc.compute(self.mgr)
        # Should not produce moving-toward, moving-away, or holding with 1 observation
        motion_rels = [
            r for r in relations
            if r.relation in (RelationType.MOVING_TOWARD, RelationType.MOVING_AWAY, RelationType.HOLDING)
        ]
        self.assertEqual(len(motion_rels), 0)

    def test_11_zero_distance_safety(self):
        # Two objects with identical centroid
        frame = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "obj1", "class": "item", "centroid": [100, 100], "depth": 2.0},
                {"id": "obj2", "class": "item", "centroid": [100, 100], "depth": 2.0},
            ],
        }
        self.mgr.update(frame)
        # Must not raise ZeroDivisionError
        relations = self.calc.compute(self.mgr)
        near_rels = [r for r in relations if r.relation == RelationType.NEAR]
        self.assertEqual(len(near_rels), 2)  # obj1->obj2 and obj2->obj1

    def test_12_multiple_unrelated_objects(self):
        f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "person_01", "class": "person", "centroid": [100, 100]},
                {"id": "laptop_01", "class": "laptop", "bbox": [250, 250, 350, 320], "centroid": [300, 285]},
                {"id": "table_01", "class": "table", "bbox": [200, 300, 600, 600], "centroid": [400, 450]},
                {"id": "chair_01", "class": "chair", "centroid": [800, 800]},
            ],
        }
        self.mgr.update(f0)
        relations = self.calc.compute(self.mgr)
        # Laptop on table
        self.assertTrue(any(r.relation == RelationType.ON and r.subject_id == "laptop_01" and r.object_id == "table_01" for r in relations))
        # Chair is far away from everything
        self.assertFalse(any(r.subject_id == "chair_01" or r.object_id == "chair_01" for r in relations))


if __name__ == "__main__":
    unittest.main()

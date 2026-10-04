#!/usr/bin/env python3
"""
scripts/test_reasoning_input.py
---------------------------------
Unit tests for PerceptionInputAdapter and contract validation (Member 2).
Covers:
  - Valid single and multiple frames
  - Multiple objects and persistent IDs
  - Nullable depth vs metric depth handling
  - Malformed frame structure rejection
  - Missing and invalid object fields (id, class, confidence, bbox, centroid)
  - Empty objects list (valid frame without detections)
  - Frame ordering preservation
  - Seamless interop with ObjectStateManager
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

from backend.reasoning.input_adapter import (
    ParsedFrame,
    ParsedObject,
    PerceptionInputAdapter,
    PerceptionValidationError,
)
from backend.reasoning.state import ObjectObservation, ObjectStateManager


class TestPerceptionInputAdapter(unittest.TestCase):
    def test_01_valid_single_frame(self):
        raw = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {
                    "id": "person_01",
                    "class": "person",
                    "confidence": 0.3119,
                    "bbox": [400, 21, 480, 310],
                    "centroid": [440, 165],
                    "mask_path": "masks/frame_00000/person_01.png",
                    "depth": None,
                }
            ],
        }
        parsed = PerceptionInputAdapter.parse_frame(raw)
        self.assertIsInstance(parsed, ParsedFrame)
        self.assertEqual(parsed.frame_index, 0)
        self.assertEqual(parsed.timestamp, 0.0)
        self.assertEqual(len(parsed.objects), 1)

        obj = parsed.objects[0]
        self.assertEqual(obj.id, "person_01")
        self.assertEqual(obj.cls, "person")
        self.assertAlmostEqual(obj.confidence, 0.3119, places=4)
        self.assertEqual(obj.bbox, (400, 21, 480, 310))
        self.assertEqual(obj.centroid, (440, 165))
        self.assertEqual(obj.mask_path, "masks/frame_00000/person_01.png")
        self.assertIsNone(obj.depth)
        self.assertIsInstance(obj.observation, ObjectObservation)

    def test_02_valid_multiple_frames(self):
        raw = [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50], "centroid": [30, 30], "mask_path": None, "depth": None}
                ],
            },
            {
                "frame_index": 5,
                "timestamp": 0.3333,
                "objects": [
                    {"id": "person_01", "class": "person", "confidence": 0.92, "bbox": [15, 10, 55, 50], "centroid": [35, 30], "mask_path": None, "depth": None}
                ],
            },
        ]
        parsed_frames = PerceptionInputAdapter.parse_sequence(raw)
        self.assertEqual(len(parsed_frames), 2)
        self.assertEqual(parsed_frames[0].frame_index, 0)
        self.assertEqual(parsed_frames[1].frame_index, 5)
        self.assertAlmostEqual(parsed_frames[1].timestamp, 0.3333, places=4)

    def test_03_multiple_objects(self):
        raw = {
            "frame_index": 12,
            "timestamp": 0.8,
            "objects": [
                {"id": "person_01", "class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300], "centroid": [150, 200], "mask_path": "masks/p.png", "depth": 2.1},
                {"id": "laptop_01", "class": "laptop", "confidence": 0.88, "bbox": [250, 200, 320, 260], "centroid": [285, 230], "mask_path": "masks/l.png", "depth": 1.9},
                {"id": "table_01", "class": "table", "confidence": 0.91, "bbox": [200, 240, 500, 400], "centroid": [350, 320], "mask_path": None, "depth": 2.0},
            ],
        }
        parsed = PerceptionInputAdapter.parse_frame(raw)
        self.assertEqual(len(parsed.objects), 3)
        ids = [o.id for o in parsed.objects]
        self.assertEqual(ids, ["person_01", "laptop_01", "table_01"])

    def test_04_depth_null_handling(self):
        raw_obj = {
            "id": "p1", "class": "person", "confidence": 0.8,
            "bbox": [10, 10, 30, 30], "centroid": [20, 20],
            "mask_path": None, "depth": None,
        }
        parsed = PerceptionInputAdapter.parse_object(raw_obj, 0, 0.0)
        self.assertIsNone(parsed.depth)
        self.assertIsNone(parsed.observation.depth)
        self.assertIsNone(parsed.to_dict()["depth"])

    def test_05_non_null_metric_depth(self):
        raw_obj = {
            "id": "p1", "class": "person", "confidence": 0.8,
            "bbox": [10, 10, 30, 30], "centroid": [20, 20],
            "mask_path": None, "depth": 2.4567,
        }
        parsed = PerceptionInputAdapter.parse_object(raw_obj, 0, 0.0)
        self.assertAlmostEqual(parsed.depth, 2.4567, places=4)
        self.assertAlmostEqual(parsed.observation.depth, 2.4567, places=4)
        self.assertEqual(parsed.to_dict()["depth"], 2.4567)

    def test_06_persistent_ids_preserved(self):
        seq = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [{"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [1, 1, 2, 2], "centroid": [1, 1]}]},
            {"frame_index": 1, "timestamp": 0.1, "objects": [{"id": "laptop_01", "class": "laptop", "confidence": 0.9, "bbox": [2, 2, 3, 3], "centroid": [2, 2]}]},
        ]
        parsed = PerceptionInputAdapter.parse_sequence(seq)
        self.assertEqual(parsed[0].objects[0].id, "laptop_01")
        self.assertEqual(parsed[1].objects[0].id, "laptop_01")

    def test_07_malformed_frame_rejection(self):
        # Non-dict
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame("not_a_dict")

        # Missing frame_index
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame({"timestamp": 0.0, "objects": []})

        # Negative frame_index
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame({"frame_index": -1, "timestamp": 0.0, "objects": []})

        # Missing timestamp
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame({"frame_index": 0, "objects": []})

        # Missing objects
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame({"frame_index": 0, "timestamp": 0.0})

        # Objects is not a list
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_frame({"frame_index": 0, "timestamp": 0.0, "objects": "none"})

    def test_08_missing_object_fields(self):
        base = {"class": "person", "confidence": 0.8, "bbox": [0, 0, 10, 10], "centroid": [5, 5]}

        # Missing id
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object(base, 0, 0.0)

        # Empty id
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "id": "   "}, 0, 0.0)

        # Missing class
        no_cls = {"id": "p1", "confidence": 0.8, "bbox": [0, 0, 10, 10], "centroid": [5, 5]}
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object(no_cls, 0, 0.0)

        # Missing confidence
        no_conf = {"id": "p1", "class": "person", "bbox": [0, 0, 10, 10], "centroid": [5, 5]}
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object(no_conf, 0, 0.0)

    def test_09_invalid_bbox(self):
        base = {"id": "p1", "class": "person", "confidence": 0.8, "centroid": [5, 5]}

        # Not 4 elements
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "bbox": [0, 0, 10]}, 0, 0.0)

        # Inverted box (x1 > x2)
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "bbox": [20, 0, 10, 10]}, 0, 0.0)

        # Non-numeric coordinate
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "bbox": ["a", 0, 10, 10]}, 0, 0.0)

    def test_10_invalid_centroid(self):
        base = {"id": "p1", "class": "person", "confidence": 0.8, "bbox": [0, 0, 10, 10]}

        # Not 2 elements
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "centroid": [5]}, 0, 0.0)

        # Non-numeric
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "centroid": ["x", 5]}, 0, 0.0)

    def test_11_invalid_confidence(self):
        base = {"id": "p1", "class": "person", "bbox": [0, 0, 10, 10], "centroid": [5, 5]}

        # Confidence > 1.0
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "confidence": 1.2}, 0, 0.0)

        # Confidence < 0.0
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "confidence": -0.1}, 0, 0.0)

        # Non-numeric confidence
        with self.assertRaises(PerceptionValidationError):
            PerceptionInputAdapter.parse_object({**base, "confidence": "high"}, 0, 0.0)

    def test_12_empty_objects_list(self):
        raw = {"frame_index": 42, "timestamp": 1.4, "objects": []}
        parsed = PerceptionInputAdapter.parse_frame(raw)
        self.assertEqual(parsed.frame_index, 42)
        self.assertEqual(parsed.objects, [])
        self.assertEqual(parsed.to_dict()["objects"], [])

    def test_13_frame_ordering_preservation(self):
        raw_list = [
            {"frame_index": 10, "timestamp": 1.0, "objects": []},
            {"frame_index": 20, "timestamp": 2.0, "objects": []},
            {"frame_index": 30, "timestamp": 3.0, "objects": []},
        ]
        parsed = PerceptionInputAdapter.parse_sequence(raw_list)
        self.assertEqual([f.frame_index for f in parsed], [10, 20, 30])

    def test_14_interoperability_with_object_state_manager(self):
        """
        Verify that ParsedFrame objects plug directly into ObjectStateManager.update()
        and produce identical tracking kinematics to raw dicts.
        """
        mgr_parsed = ObjectStateManager()
        mgr_raw = ObjectStateManager()

        raw_f0 = {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {"id": "p1", "class": "person", "confidence": 0.9, "bbox": [100, 100, 200, 300], "centroid": [150, 200], "depth": 2.0}
            ],
        }
        raw_f1 = {
            "frame_index": 5,
            "timestamp": 0.5,
            "objects": [
                {"id": "p1", "class": "person", "confidence": 0.92, "bbox": [130, 100, 230, 300], "centroid": [180, 200], "depth": 2.1}
            ],
        }

        # Update manager 1 using ParsedFrame objects
        parsed_f0 = PerceptionInputAdapter.parse_frame(raw_f0)
        parsed_f1 = PerceptionInputAdapter.parse_frame(raw_f1)
        mgr_parsed.update(parsed_f0)
        st_parsed_0 = mgr_parsed.get_state("p1")
        self.assertEqual(st_parsed_0.current_centroid, (150, 200))

        mgr_parsed.update(parsed_f1)
        st_parsed_1 = mgr_parsed.get_state("p1")

        # Update manager 2 using raw dicts
        mgr_raw.update(raw_f0)
        mgr_raw.update(raw_f1)
        st_raw_1 = mgr_raw.get_state("p1")

        # Kinematics and fields must be identical
        self.assertEqual(st_parsed_1.displacement_px, st_raw_1.displacement_px)
        self.assertEqual(st_parsed_1.distance_px, st_raw_1.distance_px)
        self.assertEqual(st_parsed_1.velocity_px_s, st_raw_1.velocity_px_s)
        self.assertEqual(st_parsed_1.speed_px_s, st_raw_1.speed_px_s)
        self.assertEqual(st_parsed_1.depth_delta_m, st_raw_1.depth_delta_m)


if __name__ == "__main__":
    unittest.main()

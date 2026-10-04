#!/usr/bin/env python3
"""
scripts/test_reasoning_rules.py
---------------------------------
Unit tests for the TemporalEventEngine and deterministic event rules (Member 2).
Covers the 10 mandated scenarios:
  1. Complete sequence: APPROACH -> REACH -> PICK_UP -> CARRY
  2. Approach and reach without pickup
  3. Passing near an object without pickup
  4. Independent target object movement
  5. Close objects with diverging/non-synchronized movement
  6. Temporary missing observation tolerance
  7. Null depth robustness (pure 2D reasoning)
  8. Duplicate event prevention across multiple frames
  9. Insufficient history (single frame safety)
 10. Pickup followed by multi-frame sustained carry
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

from backend.reasoning.relations import (
    RelationCalculator,
    RelationConfig,
)
from backend.reasoning.rules import (
    EventRuleConfig,
    PairStage,
    TemporalEventEngine,
)
from backend.reasoning.state import ObjectStateManager
from backend.reasoning.types import EventType


class TestTemporalEventEngine(unittest.TestCase):
    def setUp(self):
        self.config = EventRuleConfig(
            approach_min_frames=1,
            pickup_min_object_speed_px_s=8.0,
            pickup_min_subject_speed_px_s=8.0,
            carry_min_speed_px_s=8.0,
        )
        self.engine = TemporalEventEngine(self.config)

    def test_scenario_01_complete_successful_sequence(self):
        """
        Complete canonical lifecycle:
        Person approaches laptop -> reaches laptop -> picks up laptop -> carries laptop.
        """
        frames = [
            # Frame 0: Person far away (x=100), laptop stationary at (300, 300)
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "objects": [
                    {"id": "person_01", "class": "person", "centroid": [100, 300], "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "centroid": [300, 300], "depth": 2.0},
                ],
            },
            # Frame 1: Person moves closer to x=180 (moving toward) -> APPROACH
            {
                "frame_index": 5,
                "timestamp": 0.5,
                "objects": [
                    {"id": "person_01", "class": "person", "centroid": [180, 300], "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "centroid": [300, 300], "depth": 2.0},
                ],
            },
            # Frame 2: Person arrives at x=230 (within reach proximity 70px) -> REACH
            {
                "frame_index": 10,
                "timestamp": 1.0,
                "objects": [
                    {"id": "person_01", "class": "person", "centroid": [230, 300], "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "centroid": [300, 300], "depth": 2.0},
                ],
            },
            # Frame 3: Both begin moving together in sync (+40px in x) -> PICK_UP
            {
                "frame_index": 15,
                "timestamp": 1.5,
                "objects": [
                    {"id": "person_01", "class": "person", "centroid": [270, 300], "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "centroid": [340, 300], "depth": 2.0},
                ],
            },
            # Frame 4: Both continue moving together (+40px in x) -> CARRY
            {
                "frame_index": 20,
                "timestamp": 2.0,
                "objects": [
                    {"id": "person_01", "class": "person", "centroid": [310, 300], "depth": 2.0},
                    {"id": "laptop_01", "class": "laptop", "centroid": [380, 300], "depth": 2.0},
                ],
            },
        ]

        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]

        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP, EventType.CARRY])
        # Verify evidence attached
        for e in events:
            self.assertIsNotNone(e.evidence)
            self.assertEqual(e.subject, "person_01")
            self.assertEqual(e.object, "laptop_01")
            self.assertGreater(e.confidence, 0.5)

    def test_scenario_02_approach_and_reach_without_pickup(self):
        """
        Person approaches and reaches, but leaves laptop resting stationary.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [100, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 300]},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [180, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 300]},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [230, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 300]},
            ]},
            # Frame 3: Person pauses at the laptop, laptop remains stationary
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [230, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 300]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]
        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH])
        self.assertNotIn(EventType.PICK_UP, event_types)
        self.assertNotIn(EventType.CARRY, event_types)

    def test_scenario_03_person_passes_near_laptop(self):
        """
        Person walks past the laptop without picking it up.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 100]},
                {"id": "laptop_01", "class": "laptop", "centroid": [200, 250]},
            ]},
            # Person walks straight down past laptop (y: 100 -> 240 -> 380)
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 240]},
                {"id": "laptop_01", "class": "laptop", "centroid": [200, 250]},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 380]},
                {"id": "laptop_01", "class": "laptop", "centroid": [200, 250]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]
        self.assertNotIn(EventType.PICK_UP, event_types)
        self.assertNotIn(EventType.CARRY, event_types)

    def test_scenario_04_laptop_moves_independently(self):
        """
        Laptop moves while person is far away or stationary elsewhere.
        No PICK_UP should trigger.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [50, 50]},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 300]},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [50, 50]},
                # Laptop shifts on its own
                {"id": "laptop_01", "class": "laptop", "centroid": [340, 300]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        self.assertEqual(len(events), 0)

    def test_scenario_05_close_objects_diverging_motion(self):
        """
        Person and laptop start close, but move in opposite directions.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [200, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [220, 300]},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                # Person moves left (-40 px), laptop moves right (+40 px)
                {"id": "person_01", "class": "person", "centroid": [160, 300]},
                {"id": "laptop_01", "class": "laptop", "centroid": [260, 300]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]
        self.assertNotIn(EventType.PICK_UP, event_types)
        self.assertNotIn(EventType.CARRY, event_types)

    def test_scenario_06_temporary_missing_observation(self):
        """
        Laptop briefly disappears in one frame; engine should not produce duplicate events.
        """
        manager = ObjectStateManager()
        # Frame 0: Approach start
        f0 = {"frame_index": 0, "timestamp": 0.0, "objects": [
            {"id": "p1", "class": "person", "centroid": [100, 300]},
            {"id": "l1", "class": "laptop", "centroid": [300, 300]},
        ]}
        manager.update(f0)
        e0 = self.engine.process_frame(manager)
        self.assertEqual(len(e0), 0)

        # Frame 1: Approach detected
        f1 = {"frame_index": 5, "timestamp": 0.5, "objects": [
            {"id": "p1", "class": "person", "centroid": [180, 300]},
            {"id": "l1", "class": "laptop", "centroid": [300, 300]},
        ]}
        manager.update(f1)
        e1 = self.engine.process_frame(manager)
        self.assertEqual([e.event for e in e1], [EventType.APPROACH])

        # Frame 2: Laptop missing for one frame
        f2 = {"frame_index": 10, "timestamp": 1.0, "objects": [
            {"id": "p1", "class": "person", "centroid": [200, 300]},
        ]}
        manager.update(f2)
        e2 = self.engine.process_frame(manager)
        # Should not produce spurious events while object missing
        self.assertEqual(len(e2), 0)

        # Frame 3: Laptop reappears at (300, 300), person reaches
        f3 = {"frame_index": 15, "timestamp": 1.5, "objects": [
            {"id": "p1", "class": "person", "centroid": [230, 300]},
            {"id": "l1", "class": "laptop", "centroid": [300, 300]},
        ]}
        manager.update(f3)
        e3 = self.engine.process_frame(manager)
        self.assertEqual([e.event for e in e3], [EventType.REACH])

    def test_scenario_07_null_depth_robustness(self):
        """
        Full event sequence functions purely in 2D image coordinates when depth is None.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [100, 200], "depth": None},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 200], "depth": None},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [180, 200], "depth": None},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 200], "depth": None},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "person_01", "class": "person", "centroid": [230, 200], "depth": None},
                {"id": "laptop_01", "class": "laptop", "centroid": [300, 200], "depth": None},
            ]},
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "person_01", "class": "person", "centroid": [270, 200], "depth": None},
                {"id": "laptop_01", "class": "laptop", "centroid": [340, 200], "depth": None},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]
        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP])
        for e in events:
            self.assertIsNone(e.evidence.distance_m)

    def test_scenario_08_duplicate_event_prevention(self):
        """
        Continuous approach/near frames should emit exactly one APPROACH and one REACH event.
        """
        frames = [
            # Frame 0
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [100, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
            # Frames 1-3: Continuous approach
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [150, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [200, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [250, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
            # Frames 4-6: Arrived near laptop
            {"frame_index": 20, "timestamp": 2.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [320, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
            {"frame_index": 25, "timestamp": 2.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [325, 300]},
                {"id": "l1", "class": "laptop", "centroid": [400, 300]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        approach_events = [e for e in events if e.event == EventType.APPROACH]
        reach_events = [e for e in events if e.event == EventType.REACH]

        self.assertEqual(len(approach_events), 1)
        self.assertEqual(len(reach_events), 1)

    def test_scenario_09_insufficient_history(self):
        """
        A single observation frame must never trigger an event.
        """
        f0 = {"frame_index": 0, "timestamp": 0.0, "objects": [
            {"id": "p1", "class": "person", "centroid": [200, 300]},
            {"id": "l1", "class": "laptop", "centroid": [210, 300]},
        ]}
        manager = ObjectStateManager()
        manager.update(f0)
        events = self.engine.process_frame(manager)
        self.assertEqual(len(events), 0)

    def test_scenario_10_pickup_followed_by_sustained_carry(self):
        """
        After pickup, exactly one CARRY event is emitted even if carrying persists
        across multiple consecutive frames.
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [100, 300]},
                {"id": "l1", "class": "laptop", "centroid": [250, 300]},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [160, 300]},
                {"id": "l1", "class": "laptop", "centroid": [250, 300]},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [210, 300]},
                {"id": "l1", "class": "laptop", "centroid": [250, 300]},
            ]},
            # Pickup
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [250, 300]},
                {"id": "l1", "class": "laptop", "centroid": [290, 300]},
            ]},
            # Carry frame 1
            {"frame_index": 20, "timestamp": 2.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [290, 300]},
                {"id": "l1", "class": "laptop", "centroid": [330, 300]},
            ]},
            # Carry frame 2
            {"frame_index": 25, "timestamp": 2.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [330, 300]},
                {"id": "l1", "class": "laptop", "centroid": [370, 300]},
            ]},
            # Carry frame 3
            {"frame_index": 30, "timestamp": 3.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [370, 300]},
                {"id": "l1", "class": "laptop", "centroid": [410, 300]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        event_types = [e.event for e in events]

        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP, EventType.CARRY])
        carry_events = [e for e in events if e.event == EventType.CARRY]
        pickup_events = [e for e in events if e.event == EventType.PICK_UP]
        self.assertEqual(len(carry_events), 1)
        self.assertEqual(len(pickup_events), 1)

    def test_scenario_11_active_subject_filtering(self):
        """
        Verify that passive objects (laptop, table) cannot become subjects of
        human-object action events (APPROACH, REACH, PICK_UP, CARRY).
        """
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "table_01", "class": "table", "centroid": [500, 500]},
                {"id": "laptop_01", "class": "laptop", "centroid": [100, 100]},
            ]},
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "table_01", "class": "table", "centroid": [500, 500]},
                # Laptop shifts toward table
                {"id": "laptop_01", "class": "laptop", "centroid": [200, 200]},
            ]},
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "table_01", "class": "table", "centroid": [500, 500]},
                # Laptop reaches table
                {"id": "laptop_01", "class": "laptop", "centroid": [450, 450]},
            ]},
        ]
        events = self.engine.process_sequence(frames)
        # Because neither is a person, no action events should be emitted
        self.assertEqual(len(events), 0)

    def test_scenario_12_4k_resolution_scaling(self):
        """
        Verify that 4K coordinates (3840x2160) with person-laptop centroid distance
        around 300-350px cleanly trigger REACH, PICK_UP, and CARRY when resolution scaling is used.
        """
        cfg_4k = EventRuleConfig.for_resolution(3840, 2160)
        rel_cfg_4k = RelationConfig.for_resolution(3840, 2160)
        rel_calc = RelationCalculator(rel_cfg_4k)
        engine_4k = TemporalEventEngine(cfg_4k, relation_calculator=rel_calc)

        frames = [
            # Frame 0: Approach start (distance 700px)
            {"frame_index": 0, "timestamp": 0.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [1000, 1000]},
                {"id": "l1", "class": "laptop", "centroid": [1700, 1000]},
            ]},
            # Frame 1: Approach step (distance 500px) -> APPROACH
            {"frame_index": 5, "timestamp": 0.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [1200, 1000]},
                {"id": "l1", "class": "laptop", "centroid": [1700, 1000]},
            ]},
            # Frame 2: Contact reach (distance 330px <= 420px threshold) -> REACH
            {"frame_index": 10, "timestamp": 1.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [1370, 1000]},
                {"id": "l1", "class": "laptop", "centroid": [1700, 1000]},
            ]},
            # Frame 3: Both move together in synchronized velocity -> PICK_UP
            {"frame_index": 15, "timestamp": 1.5, "objects": [
                {"id": "p1", "class": "person", "centroid": [1470, 1000]},
                {"id": "l1", "class": "laptop", "centroid": [1800, 1000]},
            ]},
            # Frame 4: Sustained synchronized motion -> CARRY
            {"frame_index": 20, "timestamp": 2.0, "objects": [
                {"id": "p1", "class": "person", "centroid": [1570, 1000]},
                {"id": "l1", "class": "laptop", "centroid": [1900, 1000]},
            ]},
        ]
        events = engine_4k.process_sequence(frames)
        event_types = [e.event for e in events]
        self.assertEqual(event_types, [EventType.APPROACH, EventType.REACH, EventType.PICK_UP, EventType.CARRY])

    def test_scenario_13_deterministic_resolution_aware_proximity(self):
        """Verify that scaling scales thresholds deterministically and preserves manual overrides."""
        cfg = EventRuleConfig.for_resolution(3840, 2160, pickup_contact_proximity_px=500.0)
        self.assertAlmostEqual(cfg.scale_factor, 3.0, places=2)
        # Manually specified pickup_contact_proximity_px is preserved and scaled
        self.assertAlmostEqual(cfg.effective_pickup_contact_proximity_px, 1500.0, places=1)
        # Default reach threshold scales by scale_factor (160 * 3.0 = 480)
        self.assertAlmostEqual(cfg.effective_reach_proximity_threshold_px, 480.0, places=1)


if __name__ == "__main__":
    unittest.main()

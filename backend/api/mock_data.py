MOCK_RESULTS = {
    "perception": [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "objects": [
                {
                    "id": "person_01",
                    "class": "person",
                    "confidence": 0.96,
                    "bbox": [120, 80, 300, 650],
                    "centroid": [210, 365],
                    "mask_path": None,
                    "depth": 2.4,
                },
                {
                    "id": "laptop_01",
                    "class": "laptop",
                    "confidence": 0.93,
                    "bbox": [430, 300, 560, 390],
                    "centroid": [495, 345],
                    "mask_path": None,
                    "depth": 1.8,
                },
                {
                    "id": "table_01",
                    "class": "table",
                    "confidence": 0.91,
                    "bbox": [350, 250, 700, 500],
                    "centroid": [525, 375],
                    "mask_path": None,
                    "depth": 2.0,
                },
            ],
        }
    ],
    "events": [
        {
            "event": "APPROACH",
            "timestamp": 2.0,
            "subject": "person_01",
            "object": "table_01",
            "confidence": 0.91,
            "evidence": {
                "frame_index": 60,
                "distance_m": 0.85,
                "position_change": True,
                "synchronized_motion": False,
                "depth_delta_m": None,
            },
        },
        {
            "event": "REACH",
            "timestamp": 3.0,
            "subject": "person_01",
            "object": "laptop_01",
            "confidence": 0.89,
            "evidence": {
                "frame_index": 90,
                "distance_m": 0.52,
                "position_change": True,
                "synchronized_motion": False,
                "depth_delta_m": None,
            },
        },
        {
            "event": "PICK_UP",
            "timestamp": 5.2,
            "subject": "person_01",
            "object": "laptop_01",
            "confidence": 0.94,
            "evidence": {
                "frame_index": 156,
                "distance_m": 0.42,
                "position_change": True,
                "synchronized_motion": True,
                "depth_delta_m": None,
            },
        },
        {
            "event": "CARRY",
            "timestamp": 8.0,
            "subject": "person_01",
            "object": "laptop_01",
            "confidence": 0.92,
            "evidence": {
                "frame_index": 240,
                "distance_m": None,
                "position_change": True,
                "synchronized_motion": True,
                "depth_delta_m": None,
            },
        },
    ],
    "explanation": {
        "video_path": "input/demo.mp4",
        "processed_at": "2026-10-04T19:00:00",
        "events": [],
        "explanation": "Person #01 approached Table #01, reached for Laptop #01, picked it up, and carried it away."
    },
}
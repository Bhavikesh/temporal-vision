#!/usr/bin/env python3
"""
scripts/test_backend_api.py
----------------------------
Integration and unit tests for the Member 3 FastAPI backend and service layer.
Tests:
  - GET /health
  - POST /api/analyze (mock mode with real perception_output.json)
  - Canonical event sequence (APPROACH -> REACH -> PICK_UP -> CARRY)
  - Passive event exclusion (no laptop_01 -> person_01, table_01 -> laptop_01)
  - Error handling: missing perception JSON (404), invalid mode (400)
  - Static file mount /output
"""

import os
from pathlib import Path
import sys
import unittest

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.api.main import app, create_app
from backend.api.service import PipelineService


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """GET /health should return 200 with status 'ok'."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_02_analyze_mock_mode_default(self):
        """POST /api/analyze with empty/default body should run on real perception output."""
        response = self.client.post("/api/analyze", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Contract verification
        self.assertEqual(data["status"], "success")
        self.assertIn("video_path", data)
        self.assertIn("processed_at", data)
        self.assertIn("events", data)
        self.assertIn("explanation", data)

        events = data["events"]
        self.assertGreater(len(events), 0)

        # Verify real person_01 -> laptop_01 interaction sequence
        pl_events = [
            e for e in events
            if e["subject"] == "person_01" and e["object"] == "laptop_01"
        ]
        pl_types = [e["event"] for e in pl_events]
        self.assertEqual(pl_types, ["APPROACH", "REACH", "PICK_UP", "CARRY"])

        # Check evidence attached to PICK_UP and CARRY
        pickup_ev = next(e for e in pl_events if e["event"] == "PICK_UP")
        self.assertTrue(pickup_ev["evidence"]["position_change"])
        self.assertTrue(pickup_ev["evidence"]["synchronized_motion"])

        carry_ev = next(e for e in pl_events if e["event"] == "CARRY")
        self.assertTrue(carry_ev["evidence"]["synchronized_motion"])

        # Explanation string should be meaningful
        self.assertIn("person_01", data["explanation"])
        self.assertIn("laptop_01", data["explanation"])

    def test_03_passive_reverse_events_filtered(self):
        """Verify no passive-object action events (e.g. laptop -> person) exist in API response."""
        response = self.client.post("/api/analyze", json={"mode": "mock"})
        self.assertEqual(response.status_code, 200)
        events = response.json()["events"]

        # Ensure no events where laptop or table is the acting subject
        passive_subjects = [e for e in events if e["subject"] in ("laptop_01", "table_01")]
        self.assertEqual(len(passive_subjects), 0)

        # Specifically check forbidden pairs
        forbidden_pairs = [
            (e["subject"], e["object"]) for e in events
            if (e["subject"] == "laptop_01" and e["object"] == "person_01")
            or (e["subject"] == "table_01" and e["object"] == "laptop_01")
        ]
        self.assertEqual(forbidden_pairs, [])

    def test_04_missing_perception_json_returns_404(self):
        """POST /api/analyze with non-existent perception file should return 404."""
        response = self.client.post(
            "/api/analyze",
            json={"mode": "mock", "perception_json": "output/non_existent_file.json"}
        )
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    def test_05_invalid_mode_returns_400(self):
        """POST /api/analyze with invalid mode should return 400."""
        response = self.client.post(
            "/api/analyze",
            json={"mode": "live_stream_not_ready"}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("unsupported mode", data["detail"].lower())

    def test_06_cors_headers_present(self):
        """Preflight OPTIONS request from localhost:5173 should return CORS headers."""
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        }
        response = self.client.options("/api/analyze", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_07_static_output_mount(self):
        """GET /output/<file> should serve files from the output directory."""
        # /output/perception_output.json exists and should be readable
        response = self.client.get("/output/perception_output.json")
        self.assertEqual(response.status_code, 200)

        # Non-existent static file returns 404
        response_missing = self.client.get("/output/does_not_exist.png")
        self.assertEqual(response_missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()

"""
backend/api/main.py
-------------------
FastAPI application for Temporal Vision (Member 3).
Provides REST endpoints for pipeline orchestration, event querying, and static assets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.api.service import PipelineService

log = logging.getLogger(__name__)

# Base project directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
_DEFAULT_PERCEPTION_JSON = _DEFAULT_OUTPUT_DIR / "perception_output.json"


class AnalyzeRequest(BaseModel):
    mode: str = Field(default="real", description="Execution mode ('real', 'live', or 'mock')")
    perception_json: Optional[str] = Field(
        default=None,
        description="Path to perception_output.json (relative to workspace or absolute)"
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Optional video path metadata"
    )


def create_app(service: Optional[PipelineService] = None) -> FastAPI:
    app = FastAPI(
        title="Temporal Vision API",
        description="REST API for temporal action detection and reasoning",
        version="0.1.0",
    )

    # 1. CORS Configuration for local frontend dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Static File Mount for Output Assets (masks, debug frames, video)
    _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/output",
        StaticFiles(directory=str(_DEFAULT_OUTPUT_DIR), check_dir=False),
        name="output",
    )

    pipeline_service = service or PipelineService(default_perception_json=_DEFAULT_PERCEPTION_JSON)

    # 3. Root and Health Endpoints
    @app.get("/", tags=["System"])
    async def root() -> Dict[str, str]:
        return {"status": "ok", "service": "temporal-vision"}

    @app.get("/health", tags=["System"])
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    # 4. Results Endpoint (Used directly by Frontend)
    @app.get("/results", tags=["Reasoning"])
    async def get_results(mock: bool = False) -> Dict[str, Any]:
        """
        Unified endpoint serving real perception detections, temporal reasoning events,
        and narrative explanation.
        """
        try:
            explanation_output = pipeline_service.run_pipeline(
                perception_json_path=_DEFAULT_PERCEPTION_JSON,
                mode="mock" if mock else "real",
            )
            output_dict = explanation_output.to_dict()

            # Load perception frames if available for frontend bounding box overlay
            perception_frames = []
            if _DEFAULT_PERCEPTION_JSON.exists():
                import json
                try:
                    with _DEFAULT_PERCEPTION_JSON.open("r", encoding="utf-8") as f:
                        perception_frames = json.load(f)
                except Exception as ex:
                    log.warning("Could not read perception JSON for frontend: %s", ex)

            return {
                "status": "success",
                "perception": perception_frames,
                "events": output_dict.get("events", []),
                "explanation": output_dict.get("explanation"),
                "video_path": output_dict.get("video_path"),
                "processed_at": output_dict.get("processed_at"),
            }
        except Exception as e:
            log.exception("Error serving /results: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # 5. Analyze Endpoint
    @app.post("/api/analyze", tags=["Reasoning"])
    async def analyze(
        request: Optional[AnalyzeRequest] = Body(default=None)
    ) -> Dict[str, Any]:
        req = request or AnalyzeRequest()

        if req.mode not in ("real", "live", "mock"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported mode: '{req.mode}'. Must be 'real', 'live', or 'mock'.",
            )

        # Resolve perception json path
        json_path = req.perception_json
        if json_path:
            p = Path(json_path)
            if not p.is_absolute():
                p = _PROJECT_ROOT / p
            resolved_path = p
        else:
            resolved_path = _DEFAULT_PERCEPTION_JSON

        try:
            explanation_output = pipeline_service.run_pipeline(
                perception_json_path=resolved_path,
                video_path=req.video_path,
                mode=req.mode,
            )
        except FileNotFoundError:
            target_name = json_path or "output/perception_output.json"
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Perception output JSON file not found: {target_name}",
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            log.exception("Unexpected error in pipeline execution: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred during temporal reasoning execution.",
            )

        output_dict = explanation_output.to_dict()
        return {
            "status": "success",
            **output_dict,
        }

    return app


app = create_app()


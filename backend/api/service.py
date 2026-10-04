"""
backend/api/service.py
----------------------
M3 Service layer orchestrating pipeline execution.
Wraps M2 TemporalReasoningEngine for consumption by API endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from backend.reasoning import ExplanationOutput, TemporalReasoningEngine


class PipelineService:
    """
    Coordinates perception input and temporal reasoning.
    Currently operates in pre-computed / mock mode using existing perception JSON.
    """

    def __init__(self, default_perception_json: Union[str, Path] = "output/perception_output.json") -> None:
        self.default_perception_json = Path(default_perception_json)
        self.engine = TemporalReasoningEngine()

    def run_pipeline(
        self,
        perception_json_path: Optional[Union[str, Path]] = None,
        video_path: Optional[str] = None,
        mode: str = "real",
    ) -> ExplanationOutput:
        """
        Execute temporal reasoning on a perception output JSON file.

        Parameters
        ----------
        perception_json_path : Optional[Union[str, Path]]
            Path to perception output JSON (defaults to default_perception_json).
        video_path : Optional[str]
            Optional path or name of source video.
        mode : str
            Execution mode ('real' | 'live' | 'mock').

        Returns
        -------
        ExplanationOutput
            Deterministic temporal events, evidence, and natural language explanation.
        """
        if mode not in ("mock", "real", "live"):
            raise ValueError(f"Unsupported pipeline mode: '{mode}'. Must be 'real', 'live', or 'mock'.")

        path = Path(perception_json_path) if perception_json_path else self.default_perception_json

        if not path.exists():
            raise FileNotFoundError(f"Perception output JSON file not found: {path}")

        # Invoke the existing M2 public interface without touching internals
        return self.engine.process_json_file(
            json_path=path,
            video_path=video_path or str(path),
        )


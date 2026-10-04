"""
reasoning/types.py
------------------
Canonical data types and output contracts for the Temporal Reasoning layer
(Member 2).

These dataclasses define the exact data structures and JSON schemas specified
in docs/DATA_MODEL.md:
  - SpatialRelation (directed relation between two objects at a given frame)
  - Evidence        (quantitative visual observations supporting an event)
  - Event           (detected temporal action with attached evidence)
  - ExplanationOutput (complete video event summary and natural language narrative)

Downstream consumers (Member 3 backend, Member 4 frontend) should import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union


class RelationType(str, Enum):
    """
    Standard spatial relation vocabulary for MVP as defined in docs/DATA_MODEL.md.
    """
    NEAR = "near"
    ON = "on"
    MOVING_TOWARD = "moving-toward"
    MOVING_AWAY = "moving-away"
    HOLDING = "holding"


class EventType(str, Enum):
    """
    Standard event types for MVP as defined in docs/DATA_MODEL.md.
    """
    APPROACH = "APPROACH"
    REACH = "REACH"
    PICK_UP = "PICK_UP"
    CARRY = "CARRY"


@dataclass
class SpatialRelation:
    """
    Represents a directed spatial relationship between two objects in a specific frame.
    Produced by Member 2 per docs/DATA_MODEL.md Section 3.
    """

    frame_index: int
    timestamp: float
    subject_id: str
    relation: Union[RelationType, str]
    object_id: str
    distance_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "subject_id": self.subject_id,
            "relation": self.relation.value if isinstance(self.relation, Enum) else str(self.relation),
            "object_id": self.object_id,
            "distance_m": round(self.distance_m, 4) if self.distance_m is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpatialRelation:
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            subject_id=str(data["subject_id"]),
            relation=data["relation"],
            object_id=str(data["object_id"]),
            distance_m=float(data["distance_m"]) if data.get("distance_m") is not None else None,
        )


@dataclass
class Evidence:
    """
    Structured quantitative observations that triggered an event decision.
    Mandatory block attached to each Event per docs/DATA_MODEL.md Section 4.
    """

    frame_index: int
    distance_m: Optional[float] = None
    position_change: Optional[bool] = None
    synchronized_motion: Optional[bool] = None
    depth_delta_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "distance_m": round(self.distance_m, 4) if self.distance_m is not None else None,
            "position_change": self.position_change,
            "synchronized_motion": self.synchronized_motion,
            "depth_delta_m": round(self.depth_delta_m, 4) if self.depth_delta_m is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        return cls(
            frame_index=int(data["frame_index"]),
            distance_m=float(data["distance_m"]) if data.get("distance_m") is not None else None,
            position_change=data.get("position_change"),
            synchronized_motion=data.get("synchronized_motion"),
            depth_delta_m=float(data["depth_delta_m"]) if data.get("depth_delta_m") is not None else None,
        )


@dataclass
class Event:
    """
    A detected temporal action with attached structured evidence.
    Produced by Member 2 per docs/DATA_MODEL.md Section 4.
    """

    event: Union[EventType, str]
    timestamp: float
    subject: str
    confidence: float
    evidence: Evidence
    object: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.value if isinstance(self.event, Enum) else str(self.event),
            "timestamp": round(self.timestamp, 4),
            "subject": self.subject,
            "object": self.object,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        evidence_data = data["evidence"]
        evidence_obj = (
            evidence_data
            if isinstance(evidence_data, Evidence)
            else Evidence.from_dict(evidence_data)
        )
        return cls(
            event=data["event"],
            timestamp=float(data["timestamp"]),
            subject=str(data["subject"]),
            object=data.get("object"),
            confidence=float(data["confidence"]),
            evidence=evidence_obj,
        )


@dataclass
class ExplanationOutput:
    """
    Top-level payload capturing all detected events and natural language summary.
    Produced by Member 2/3 per docs/DATA_MODEL.md Section 5.
    """

    video_path: str
    processed_at: str
    explanation: str
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "processed_at": self.processed_at,
            "events": [event.to_dict() for event in self.events],
            "explanation": self.explanation,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExplanationOutput:
        events = [
            e if isinstance(e, Event) else Event.from_dict(e)
            for e in data.get("events", [])
        ]
        return cls(
            video_path=str(data["video_path"]),
            processed_at=str(data["processed_at"]),
            events=events,
            explanation=str(data.get("explanation", "")),
        )

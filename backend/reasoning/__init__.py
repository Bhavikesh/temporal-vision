"""
reasoning — Member 2 temporal reasoning and event engine for Temporal Vision.

Public surface and data contracts:
  - RelationType, EventType
  - SpatialRelation, Evidence, Event, ExplanationOutput
"""

from .engine import TemporalReasoningEngine
from .evidence import (
    EvidenceCollector,
    EvidenceValidationError,
)
from .explainer import EventExplainer
from .input_adapter import (
    ParsedFrame,
    ParsedObject,
    PerceptionInputAdapter,
    PerceptionValidationError,
)
from .relations import (
    RelationCalculator,
    RelationConfig,
)
from .rules import (
    EventRuleConfig,
    PairStage,
    TemporalEventEngine,
)
from .state import (
    ObjectObservation,
    ObjectStateManager,
    ObjectTemporalState,
)
from .types import (
    EventType,
    Evidence,
    Event,
    ExplanationOutput,
    RelationType,
    SpatialRelation,
)

__all__ = [
    "EventExplainer",
    "EventRuleConfig",
    "EventType",
    "Evidence",
    "EvidenceCollector",
    "EvidenceValidationError",
    "Event",
    "ExplanationOutput",
    "ObjectObservation",
    "ObjectStateManager",
    "ObjectTemporalState",
    "PairStage",
    "ParsedFrame",
    "ParsedObject",
    "PerceptionInputAdapter",
    "PerceptionValidationError",
    "RelationCalculator",
    "RelationConfig",
    "RelationType",
    "SpatialRelation",
    "TemporalEventEngine",
    "TemporalReasoningEngine",
]

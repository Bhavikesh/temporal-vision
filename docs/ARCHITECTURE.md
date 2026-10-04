# ARCHITECTURE — Temporal Vision

---

## A. Full Intended Architecture

The full system is a sequential pipeline from raw video to grounded natural language explanation. Each stage has a clean input/output contract.

```
VIDEO (MP4 / frame sequence)
        │
        ▼
┌───────────────────────────┐
│   GROUNDING DINO-B        │  Open-vocab detection
│   text prompt → boxes     │  "person . laptop . table ."
└───────────┬───────────────┘
            │  bbox + class + confidence
            ▼
┌───────────────────────────┐
│   SAM 2.1 Base+           │  Video segmentation + tracking
│   boxes → masks + IDs     │  Propagates across frames
└───────────┬───────────────┘
            │  mask + stable object_id
            ▼
┌───────────────────────────┐
│   DEPTH PRO               │  Metric monocular depth
│   frame → depth map       │  Object-level depth values
└───────────┬───────────────┘
            │  depth (nullable)
            ▼
┌───────────────────────────┐
│   PERSISTENT OBJECT STATE │  Per-frame structured state
│   id, bbox, centroid,     │  Stable IDs across all frames
│   mask_path, depth        │
└───────────┬───────────────┘
            │  List[PerceptionFrame]  (JSON)
            ▼
┌───────────────────────────┐
│   TEMPORAL SCENE GRAPH    │  Nodes = objects
│                           │  Edges = spatial relations
│   person → near → laptop  │  Evolves frame by frame
│   laptop → on → table     │
└───────────┬───────────────┘
            │  graph snapshot per frame
            ▼
┌───────────────────────────────────────┐
│  GNN / GRAPH TRANSFORMER + RULES      │  Full: learned
│  Temporal Event Reasoning             │  MVP: deterministic rules
│  Risk Reasoning                       │
│  Intent Inference                     │
└───────────┬───────────────────────────┘
            │  List[Event]  (JSON)
            ▼
┌───────────────────────────┐
│   EVIDENCE LAYER          │  Attach frame evidence to
│                           │  every event conclusion
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│   VLM / LLM EXPLANATION   │  Consumes structured evidence
│                           │  Produces grounded narrative
│   NOT the source of truth │  Must not hallucinate geometry
└───────────┬───────────────┘
            │
            ▼
        FRONTEND UI
```

### Key Design Principles (Full System)

- **Pretrained models** do all generic perception (no training on custom data)
- **The temporal scene graph** is the structured world model — not raw pixel space
- **The VLM/LLM** consumes structured evidence; it never overrides geometry or identity
- **Evidence** is mandatory for every event and risk conclusion
- **Object IDs** are stable from first detection to last frame

---

## B. Current MVP Architecture

The overnight MVP uses a simplified pipeline. Components marked `[PLANNED]` are not implemented.

```
VIDEO (MP4)
        │
        ▼
┌───────────────────────────┐
│   GROUNDING DINO-B        │  ✅ IMPLEMENTED
│   SwinB backbone          │  backend/perception/detector.py
│   prompt: "person .       │
│   laptop . table ."       │
└───────────┬───────────────┘
            │  bbox + class + confidence
            ▼
┌───────────────────────────┐
│   SAM 2.1 Base+           │  ✅ IMPLEMENTED
│   Video predictor         │  backend/perception/tracker.py
│   Box-prompted init       │
└───────────┬───────────────┘
            │  mask + stable object_id per frame
            ▼
┌───────────────────────────┐
│   DEPTH PRO               │  ✅ OPTIONAL (adapter present)
│   Graceful null if absent │  backend/perception/depth.py
└───────────┬───────────────┘
            │  depth (nullable float, metres)
            ▼
┌───────────────────────────┐
│   PERSISTENT OBJECT STATE │  ✅ IMPLEMENTED
│   PerceptionFrame JSON    │  backend/perception/types.py
│   output/perception_      │  backend/perception/pipeline.py
│   output.json             │
└───────────┬───────────────┘
            │  List[PerceptionFrame]  — team contract
            ▼
┌───────────────────────────┐
│   TEMPORAL RULE ENGINE    │  [Member 2 — feat/reasoning]
│   deterministic rules     │  NOT YET IMPLEMENTED
│   APPROACH / REACH /      │
│   PICK_UP / CARRY         │
└───────────┬───────────────┘
            │  List[Event]
            ▼
┌───────────────────────────┐
│   EVIDENCE                │  [Member 2 — feat/reasoning]
│   frame ref, distances,   │  NOT YET IMPLEMENTED
│   motion vectors          │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│   BACKEND / API           │  [Member 3 — feat/backend]
│   FastAPI                 │  NOT YET IMPLEMENTED
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│   FRONTEND UI             │  [Member 4 — feat/frontend]
│   Timeline, scene graph,  │  NOT YET IMPLEMENTED
│   evidence panel          │
└───────────────────────────┘
```

### Why This Reduced Architecture

The competition deadline requires a working end-to-end demo by tomorrow morning. The MVP trades scope for reliability:

- GNN reasoning → deterministic rules (faster, no training required)
- TAPNext++ tracking → SAM 2.1 (simpler, well-tested on demo video)
- VLM explanation → template-based (no model inference cost)
- Scene graph DB → in-memory JSON state (no infrastructure)

---

## Component Ownership

| Component | Owner | Branch | Status |
|-----------|-------|--------|--------|
| Grounding DINO | Member 1 | feat/perception | ✅ |
| SAM 2.1 | Member 1 | feat/perception | ✅ |
| Depth Pro | Member 1 | feat/perception | ✅ optional |
| Perception output JSON | Member 1 | feat/perception | ✅ |
| Debug visualisation | Member 1 | feat/perception | ✅ |
| Temporal state / relations | Member 2 | feat/reasoning | 🔜 |
| Event detection | Member 2 | feat/reasoning | 🔜 |
| Evidence layer | Member 2 | feat/reasoning | 🔜 |
| FastAPI backend | Member 3 | feat/backend | 🔜 |
| Frontend UI | Member 4 | feat/frontend | 🔜 |

---

## Interface Contract

Member 1 outputs `output/perception_output.json`:

```json
[
  {
    "frame_index": 120,
    "timestamp": 4.0,
    "objects": [
      {
        "id": "person_01",
        "class": "person",
        "confidence": 0.96,
        "bbox": [120, 80, 300, 650],
        "centroid": [210, 365],
        "mask_path": "masks/frame_00120/person_01.png",
        "depth": 2.4
      }
    ]
  }
]
```

Members 2, 3, and 4 consume this JSON. They must not import `backend/perception` directly.

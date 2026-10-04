# PRD — Temporal Vision

## Product

**Temporal Vision** — a temporal scene-understanding system that transforms video into structured, evidence-grounded explanations of what is happening over time.

---

## Problem

Traditional computer vision answers:
> "What objects are present in this frame?"

It does not robustly answer:
> "What is happening? How are objects related? What changed over time? What evidence supports that conclusion?"

Object detection without persistent identity, temporal context, or relational reasoning cannot detect events, infer intent, or explain its conclusions.

---

## Target Outcome

Move from isolated, frame-level object detection to:

- **Persistent object identity** across frames
- **Spatial relations** between objects
- **Temporal state changes** (object moves, is picked up, is carried)
- **Event detection** with named, structured events
- **Evidence** attached to every conclusion
- **Explanation** grounded in the visual evidence — not hallucinated

---

## Primary Users

- Video monitoring / safety / security operators
- Visual analytics researchers
- Hackathon demo judges evaluating temporal scene understanding

---

## MVP Demo Scenario (Tonight)

A controlled, single-camera indoor scenario:

```
Person enters frame
  → approaches a table with a laptop on it
  → reaches toward the laptop
  → picks up the laptop
  → carries the laptop away
```

Target object classes: **PERSON**, **LAPTOP**, **TABLE**

Target events: **APPROACH**, **REACH**, **PICK_UP**, **CARRY**

---

## Functional Requirements

| # | Requirement | MVP |
|---|-------------|-----|
| FR-01 | Accept a short video file as input | ✅ |
| FR-02 | Detect PERSON, LAPTOP, TABLE using text-prompted detection | ✅ |
| FR-03 | Maintain stable object IDs across frames | ✅ |
| FR-04 | Propagate object masks through video | ✅ |
| FR-05 | Capture object position (bbox, centroid) per frame | ✅ |
| FR-06 | Capture metric depth per object where available | Optional |
| FR-07 | Infer temporal spatial relations (near, on, moving-toward) | Member 2 |
| FR-08 | Detect APPROACH, REACH, PICK_UP, CARRY events | Member 2 |
| FR-09 | Attach structured evidence to each detected event | Member 2 |
| FR-10 | Generate a natural language explanation of events | Member 2/4 |
| FR-11 | Visualise video with overlaid boxes, masks, IDs | ✅ debug mode |
| FR-12 | Display event timeline and evidence panel in UI | Member 4 |

---

## Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR-01 | Modular — each component has a clean JSON interface |
| NFR-02 | Deterministic where possible (no random IDs, stable ordering) |
| NFR-03 | Reproducible on the same hardware without retraining |
| NFR-04 | No database required for the MVP |
| NFR-05 | No cloud service required — runs entirely local |
| NFR-06 | Output is human-readable JSON |
| NFR-07 | Failure of one optional component (e.g. Depth Pro) must not crash the pipeline |

---

## Out of Scope Tonight

The following are **explicitly excluded** from the overnight MVP:

- Point tracking (TAP / TAPIR / TAPNext++)
- GNN or Graph Transformer reasoning
- Learned temporal event model
- Learned risk classifier
- InternVideo3 / any video foundation model
- VLM / LLM reasoning (beyond a simple template explanation)
- Real-time inference
- Multi-camera support
- Authentication or user accounts
- Cloud deployment
- Persistent database
- Large event taxonomy (only the 4 demo events)

---

## Success Criteria (Demo)

The demo is a success if a judge can watch the annotated video and event timeline, and see:

1. Person, Laptop, and Table correctly identified with stable IDs
2. At least APPROACH and PICK_UP events correctly detected
3. Evidence panel shows the data supporting each event
4. Explanation text correctly describes the action

---

## Repo

`github.com/Bhavikesh/temporal-vision`  
Current branch: `feat/perception`

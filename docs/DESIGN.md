# DESIGN — Temporal Vision

## Demo Experience

The system processes a short video and presents a multi-panel interface showing what happened, why the system concluded it happened, and where in the video the evidence lives.

---

## Main Screen Layout

```
┌──────────────────────────────┬──────────────────────────┐
│                              │  EVENT TIMELINE          │
│      VIDEO PLAYER            │  ─────────────────────── │
│                              │  00:02  APPROACH         │
│  [Person #01]  [Laptop #01]  │  00:03  REACH            │
│  [Table #01]                 │  00:05  PICK_UP  ◄ now   │
│                              │  00:08  CARRY            │
│  (bboxes / masks overlaid)   │                          │
│                              │  SCENE GRAPH             │
│                              │  Person #01 → near →     │
│                              │    Laptop #01            │
│                              │  Laptop #01 → on →       │
│                              │    Table #01             │
├──────────────────────────────┴──────────────────────────┤
│  EVIDENCE PANEL — PICK_UP @ 00:05                       │
│  ─────────────────────────────────────────────────────  │
│  Timestamp        : 5.2s                                │
│  Subject          : Person #01                          │
│  Object           : Laptop #01                          │
│  Distance         : 0.42 m                              │
│  Position change  : true (Laptop moved with Person)     │
│  Sync motion      : true                                │
│  Confidence       : 0.94                                │
├─────────────────────────────────────────────────────────┤
│  EXPLANATION                                            │
│  "Person #01 picked up Laptop #01 at 00:05              │
│   and carried it away from Table #01."                  │
└─────────────────────────────────────────────────────────┘
```

---

## Panel Descriptions

### Video Player

- Plays the original video
- Overlaid:
  - Bounding boxes (colour-coded per class)
  - Segmentation masks (semi-transparent)
  - Object labels: `Person #01`, `Laptop #01`, `Table #01`
  - Confidence score
  - Depth value where available
- Scrubbing syncs Event Timeline and Evidence Panel

### Event Timeline

- Ordered list of detected events with timestamps
- Each event is clickable — jumps video to that frame
- Events: APPROACH, REACH, PICK_UP, CARRY
- Colour: green (normal) / amber (attention) / red (risk)

### Scene Graph

- Per-frame snapshot of spatial relations
- Nodes: detected objects with stable IDs
- Edges: spatial relations (near, on, moving-toward, holding)
- Updates as video scrubs

### Evidence Panel

- Populated when an event is selected
- Shows the structured data that triggered the event rule:
  - Frame reference
  - Subject / object IDs
  - Distance metric
  - Motion delta
  - Confidence

### Explanation

- One or two sentences describing the event
- Grounded in evidence, not hallucinated
- MVP: template-based (`"Person #01 {verb} Laptop #01 at {time}"`)
- Future: VLM narrative from structured evidence

---

## Object Label Convention

```
Person #01      (class, index)
Laptop #01
Table #01
```

Index is 1-based. Zero-padded to 2 digits (`01`, `02`).  
IDs are **stable** — the same object keeps the same ID for the entire video.

---

## Colour Palette

| Class | Colour |
|-------|--------|
| Person | Green `(0, 200, 50)` |
| Laptop | Blue `(50, 150, 255)` |
| Table | Cyan `(0, 200, 220)` |
| Unknown | Grey `(180, 180, 180)` |

---

## Event Definitions (MVP)

| Event | Trigger Condition |
|-------|-------------------|
| APPROACH | Person centroid moving toward Table; distance decreasing |
| REACH | Person centroid within reach threshold of Laptop |
| PICK_UP | Laptop centroid begins moving in sync with Person centroid; Laptop no longer resting on Table position |
| CARRY | Person and Laptop moving together across multiple frames |

These rules are implemented by Member 2 (feat/reasoning), consuming the perception JSON.

---

## Debug Mode

When `--debug` is passed to the pipeline, the system writes:

- `output/debug_frames/frame_XXXXX.jpg` — annotated frame images
- `output/debug_video.mp4` — assembled annotated video
- `output/masks/frame_XXXXX/<id>.png` — binary masks

This is intended for Member 4 to use as evidence visualisation assets.

# DATA_MODEL — Temporal Vision JSON Contracts

This document is the single source of truth for all cross-module data schemas.  
Every module must produce and consume exactly these formats.  
Do not add fields that are not defined here without updating this document first.

---

## 1. DetectedObject

Produced by Member 1 (perception). Consumed by Member 2 (reasoning) via `PerceptionFrame`.

```json
{
  "id":         "person_01",
  "class":      "person",
  "confidence": 0.96,
  "bbox":       [120, 80, 300, 650],
  "centroid":   [210, 365],
  "mask_path":  "masks/frame_00120/person_01.png",
  "depth":      2.40
}
```

### Field Definitions

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | string | No | Stable object identity across all frames. Format: `{class}_{NN}` e.g. `person_01`. Never changes within a video run. |
| `class` | string | No | Class label as returned by the detector. Matches the text prompt token. |
| `confidence` | float | No | Detector confidence. Range: `[0.0, 1.0]`. |
| `bbox` | [int, int, int, int] | No | Axis-aligned bounding box in **original image pixel coordinates**: `[x1, y1, x2, y2]`. Top-left origin. |
| `centroid` | [int, int] | No | Bounding-box centre in pixel coordinates: `[cx, cy]`. |
| `mask_path` | string | Yes | Relative path from the output directory root to the binary mask PNG. `null` if mask not available for this frame. |
| `depth` | float | Yes | Metric depth at the object centroid, in **metres**. `null` if Depth Pro is disabled or failed. |

### Constraints
- `bbox` values are integers in `[0, image_width-1]` and `[0, image_height-1]`
- `centroid` is derived as `[int((x1+x2)/2), int((y1+y2)/2)]`
- `depth` raw map is **never** embedded in JSON — only the compact scalar value
- If an object is absent from a frame, it is **omitted** entirely — not represented with null bbox

---

## 2. PerceptionFrame

One entry per sampled video frame. Produced by Member 1.  
This is the primary output file: `output/perception_output.json`

```json
{
  "frame_index": 120,
  "timestamp":   4.0,
  "objects": [
    {
      "id":         "person_01",
      "class":      "person",
      "confidence": 0.96,
      "bbox":       [120, 80, 300, 650],
      "centroid":   [210, 365],
      "mask_path":  "masks/frame_00120/person_01.png",
      "depth":      2.40
    },
    {
      "id":         "laptop_01",
      "class":      "laptop",
      "confidence": 0.93,
      "bbox":       [430, 300, 560, 390],
      "centroid":   [495, 345],
      "mask_path":  "masks/frame_00120/laptop_01.png",
      "depth":      1.80
    }
  ]
}
```

### Field Definitions

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `frame_index` | int | No | Original video frame number (0-indexed). |
| `timestamp` | float | No | Time in seconds from video start. `frame_index / fps`. Rounded to 4 decimal places. |
| `objects` | array | No | List of `DetectedObject` visible in this frame. Empty array `[]` if no objects detected. |

### Full output file

`output/perception_output.json` is a JSON array of `PerceptionFrame` objects, sorted ascending by `frame_index`.

---

## 3. SpatialRelation

Produced by Member 2 (reasoning). Represents a directed relation between two objects at a given time.

```json
{
  "frame_index":  120,
  "timestamp":    4.0,
  "subject_id":   "person_01",
  "relation":     "near",
  "object_id":    "laptop_01",
  "distance_m":   0.85
}
```

### Relation Vocabulary (MVP)

| Relation | Meaning |
|----------|---------|
| `near` | Subject centroid is within proximity threshold of object |
| `on` | Subject is resting on object (Laptop on Table) |
| `moving-toward` | Subject centroid is approaching object over time |
| `moving-away` | Subject centroid is receding from object over time |
| `holding` | Subject and object moving in sync (post PICK_UP) |

---

## 4. Event

Produced by Member 2. One event per detected action.

```json
{
  "event":      "PICK_UP",
  "timestamp":  5.2,
  "subject":    "person_01",
  "object":     "laptop_01",
  "confidence": 0.94,
  "evidence": {
    "frame_index":        156,
    "distance_m":         0.42,
    "position_change":    true,
    "synchronized_motion": true,
    "depth_delta_m":      null
  }
}
```

### Field Definitions

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `event` | string | No | Event type. MVP: `APPROACH`, `REACH`, `PICK_UP`, `CARRY`. |
| `timestamp` | float | No | Time in seconds when event was detected. |
| `subject` | string | No | ID of the acting object (always `person_01` in MVP). |
| `object` | string | Yes | ID of the target object. `null` for events with no target. |
| `confidence` | float | No | Rule confidence. `[0.0, 1.0]`. |
| `evidence` | object | No | Structured data that triggered the rule. Must be present. |

### Evidence Block

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `frame_index` | int | No | Frame where evidence was observed. |
| `distance_m` | float | Yes | Distance between subject and object centroids. Null if depth unavailable. |
| `position_change` | bool | Yes | Whether the object's position changed relative to previous frame. |
| `synchronized_motion` | bool | Yes | Whether subject and object moved together (PICK_UP / CARRY). |
| `depth_delta_m` | float | Yes | Change in depth, in metres. Null if depth unavailable. |

---

## 5. ExplanationOutput

Produced by Member 2 or 3.

```json
{
  "video_path":   "input/demo.mp4",
  "processed_at": "2026-10-04T14:30:00",
  "events":       [ /* List[Event] */ ],
  "explanation":  "Person #01 approached Table #01 at 00:02, reached for Laptop #01 at 00:03, picked it up at 00:05, and carried it away at 00:08."
}
```

---

## Artefact File Layout

```
output/
├── perception_output.json      ← Member 1 produces; Member 2 consumes
├── events.json                 ← Member 2 produces; Member 3/4 consume
├── explanation.json            ← Member 2/3 produces; Member 4 consumes
├── masks/
│   └── frame_XXXXX/
│       ├── person_01.png       ← Binary mask, uint8, 0 or 255
│       ├── laptop_01.png
│       └── table_01.png
└── debug_frames/               ← Optional; Member 4 may use for evidence assets
    └── frame_XXXXX.jpg
```

---

## Versioning

This schema is **v1** (MVP). Any breaking change must be documented here with a version bump and communicated to all members before merging.

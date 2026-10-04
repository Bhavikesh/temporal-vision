# TASKS — Team Responsibilities and Integration Plan

## Team of Four

---

### Member 1 — Perception

**Branch:** `feat/perception`  
**Status:** 🔄 In progress

**Responsibilities:**
- Grounding DINO-B detection adapter
- SAM 2.1 video tracking adapter
- Depth Pro optional adapter
- `PerceptionFrame` JSON output (team contract)
- Frame extraction and pipeline orchestration
- Binary mask saving
- Debug visualisation (annotated frames + video)
- Smoke test (`scripts/test_perception.py`)

**Deliverable:**  
`output/perception_output.json` — a valid JSON array of `PerceptionFrame` objects.

**Files owned:**
```
backend/perception/
├── __init__.py
├── config.py
├── device.py
├── types.py
├── detector.py
├── tracker.py
├── depth.py
├── debug.py
└── pipeline.py
scripts/test_perception.py
checkpoints/           (not committed)
```

**Done when:**
- [ ] GDino detects person/laptop/table on the demo video
- [ ] SAM 2.1 propagates masks across all sampled frames
- [ ] Object IDs are stable
- [ ] `perception_output.json` is valid and follows the schema in `DATA_MODEL.md`
- [ ] Debug video is human-readable

---

### Member 2 — Temporal Reasoning

**Branch:** `feat/reasoning`  
**Status:** 🔜 Not started

**Responsibilities:**
- Consume `output/perception_output.json`
- Maintain per-object state across frames
- Compute spatial relations between objects per frame
- Implement rule-based event detection (APPROACH, REACH, PICK_UP, CARRY)
- Attach structured evidence to each event
- Write `output/events.json`
- Write template-based explanation

**Input:**  
`output/perception_output.json` (from Member 1)

**Output:**  
`output/events.json` — list of `Event` objects per `DATA_MODEL.md`

**Key rules:**
- Must handle `null` depth gracefully
- Must handle frames with missing objects (object absent = no rule fires)
- Evidence block is mandatory for every event
- Do not add a GNN or ML model tonight

---

### Member 3 — Backend / Integration

**Branch:** `feat/backend`  
**Status:** 🔜 Not started

**Responsibilities:**
- FastAPI application
- Endpoint to accept a video file, trigger the perception pipeline, return results
- Serve `perception_output.json` and `events.json` to the frontend
- Mock mode: return pre-computed results for demo stability
- Schema validation middleware

**Key endpoints (suggested):**
```
POST /run          — accept video, run full pipeline
GET  /results      — return perception + events JSON
GET  /video/{frame} — serve annotated frame images
```

**Do not:**
- Re-implement perception or reasoning logic
- Add a database
- Import `backend.perception` internals directly

---

### Member 4 — Frontend

**Branch:** `feat/frontend`  
**Status:** 🔜 Not started

**Responsibilities:**
- Video player with overlay (boxes, masks, IDs)
- Event timeline panel
- Scene graph visualisation (per-frame)
- Evidence panel (populated on event click)
- Explanation text panel
- Consume REST endpoints from Member 3

**Assets available from Member 1:**
- `output/debug_frames/frame_XXXXX.jpg` — annotated frames
- `output/masks/frame_XXXXX/*.png` — binary masks
- `output/debug_video.mp4` — pre-annotated video

---

## Integration Order

Integration must happen in this order. Do not skip steps.

```
Step 1  ←  Member 1 completes perception output
           perception_output.json is validated

Step 2  ←  Members 2 and 3 agree on events.json schema
           (already defined in DATA_MODEL.md)

Step 3  ←  Member 2 implements reasoning
           events.json is produced and validated

Step 4  ←  Member 3 wires FastAPI to both JSON files
           /results endpoint tested with curl

Step 5  ←  Member 4 connects frontend to backend
           All four panels rendering with real data

Step 6  ←  End-to-end demo test on the demo video
           All events detected, explanation correct

Step 7  ←  Final validation
           Judges can run the demo without intervention
```

---

## Shared Constraints

All members must follow `docs/RULES.md`.

**No member may:**
- Modify another member's module files
- Change the schema in `DATA_MODEL.md` without team agreement
- Commit model weights, videos, or secrets
- Claim a feature is working without running a real test

---

## Demo Video

The canonical demo video is the person-laptop-table scenario.  
Location: provided by team lead (not committed to git).  
Members 2, 3, 4 should use the pre-computed `perception_output.json` from Member 1 until their own pipeline is integrated.

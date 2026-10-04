# AI_SPEC — Model and AI Component Specifications

This document defines the responsibilities, inputs, outputs, and constraints of every AI/ML component in Temporal Vision.

---

## 1. Grounding DINO-B (SwinB backbone)

**Role:** Open-vocabulary object detection. Locates objects described by a free-form text prompt in a single frame.

**Status:** ✅ Implemented — `backend/perception/detector.py`

**Checkpoint:** `checkpoints/groundingdino_swinb_cogcoor.pth`  
**Config:** `checkpoints/GroundingDINO_SwinB_cfg.py`

**Input:**
- RGB frame (numpy `uint8`, H×W×3)
- Text prompt string, e.g. `"person . laptop . table ."`
- `box_threshold` (float, default 0.35)
- `text_threshold` (float, default 0.25)

**Output per frame:**
```
List of DetectedObject:
  id          : str   — temporary local ID, e.g. "person_01"
  cls         : str   — class label from prompt
  confidence  : float — in [0, 1]
  bbox        : [x1, y1, x2, y2]  — pixel coordinates
  centroid    : [cx, cy]           — pixel coordinates
```

**Constraints:**
- Does NOT expose raw model tensors. Output is clean Python dataclasses.
- Runs on the best available device: CUDA → MPS → CPU
- On macOS (no CUDA), runs in CPU-only mode (custom C++ ops not compiled — this is expected and correct)
- One BERT download per first run (bert-base-uncased from HuggingFace)

**Text prompt format:**  
Classes separated by ` . ` with a trailing `.`:  
`"person . laptop . table ."`

---

## 2. SAM 2.1 Base+ (Hiera backbone)

**Role:** Video object segmentation and tracking. Takes bounding box prompts from the detector and propagates binary masks through the entire video.

**Status:** ✅ Implemented — `backend/perception/tracker.py`

**Checkpoint:** `checkpoints/sam2.1_hiera_base_plus.pt`  
**Config:** `configs/sam2.1/sam2.1_hiera_b+.yaml` (bundled in the sam2 package)

**Input:**
- Directory of JPEG frames named `%05d.jpg` (0-indexed original frame numbers)
- Seed frame index (where detector ran)
- List of `DetectedObject` from the detector (bounding boxes used as prompts)

**Output per frame:**
- Binary mask PNG per object: `masks/frame_XXXXX/<id>.png`
- Updated `bbox`, `centroid` derived from the mask

**ID assignment:**
- Deterministic: objects sorted by `(class, bbox_x1)` at the seed frame
- SAM integer IDs map to stable human IDs: `person_01`, `laptop_01`, `table_01`
- If an object is not visible in a frame, it is absent from that frame's output (never fabricated)

**MPS note:**  
Requires `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon. Set automatically by the pipeline.

**Constraints:**
- Requires frames as JPEG files (not raw video). The pipeline extracts frames to a temp directory.
- Does not hallucinate mask positions. Absent = absent.

---

## 3. Depth Pro (Apple ml-depth-pro)

**Role:** Metric monocular depth estimation. Provides per-pixel depth in metres from a single RGB frame.

**Status:** ✅ Optional adapter — `backend/perception/depth.py`  
If checkpoint is absent or package is not installed, depth fields are `null` and a single warning is logged. The pipeline continues.

**Checkpoint:** `checkpoints/depth_pro.pt`

**Input:**
- RGB frame (numpy `uint8`, H×W×3)

**Output:**
- `depth_map`: float32 numpy array (H×W), values in metres — NOT serialised to JSON
- `depth_at(cx, cy)`: float — depth at object centroid (this is what goes in the JSON)
- `depth_mean(depth_map, mask)`: float — mean depth inside the object mask

**Constraints:**
- Full depth map is never written to JSON (too large). Only compact per-object values.
- `depth` field in output JSON may be `null`.
- Member 2 (reasoning) must handle `null` depth gracefully.

---

## 4. Temporal Reasoning Engine

**Role:** Consume the perception JSON, maintain per-object state across frames, infer spatial relations, and detect named events.

**Status:** 🔜 NOT YET IMPLEMENTED — Member 2 (`feat/reasoning`)

### Current MVP approach (planned):
- Deterministic rule-based engine
- Rules operate on `PerceptionFrame` sequence
- Relations computed from centroid distances and depth deltas

### Full intended approach (future):
- **Temporal Scene Graph**: graph data structure, nodes = objects, edges = relations
- **GNN / Graph Transformer**: learned event classification over graph snapshots
- Relation types: `near`, `on`, `moving-toward`, `holding`, `moving-away`
- Event types: `APPROACH`, `REACH`, `PICK_UP`, `CARRY`, and future taxonomy

**Input (from Member 1):**
```json
List[PerceptionFrame]  — perception_output.json
```

**Output:**
```json
List[Event]  — see DATA_MODEL.md
```

---

## 5. VLM / LLM Explanation

**Role:** Generate a natural language explanation of detected events, grounded in structured evidence.

**Status:** 🔜 NOT YET IMPLEMENTED

### Current MVP approach (planned):
- Template-based string generation
- No external model call
- Example: `"Person #01 picked up Laptop #01 at 00:05 and carried it away from Table #01."`

### Full intended approach (future):
- VLM (e.g. GPT-4V, Gemini, InternVL) receives structured event + evidence JSON
- Produces grounded narrative

### Hard constraints (applies to both MVP and full):
- The VLM **must not** override geometry, identity, or relations inferred from vision
- The VLM **must** receive the evidence block explicitly
- The VLM **must not** be used to detect events — event detection is the reasoning engine's job
- If the VLM produces a claim not supported by evidence, it is discarded

---

## 6. InternVideo3 (NOT in current MVP)

**Role (intended):** Video-level semantic understanding, action recognition, temporal grounding.

**Status:** ❌ NOT in MVP — explicitly excluded (DEC-004 equivalent)

**Future:** May be evaluated after the MVP baseline is validated.

---

## Summary Table

| Component | MVP Status | Location | Notes |
|-----------|-----------|----------|-------|
| Grounding DINO-B | ✅ Working | `backend/perception/detector.py` | CPU mode on Mac |
| SAM 2.1 Base+ | ✅ Working | `backend/perception/tracker.py` | MPS fallback required |
| Depth Pro | ✅ Optional | `backend/perception/depth.py` | null if absent |
| Temporal Rule Engine | 🔜 Planned | `feat/reasoning` | Member 2 |
| GNN Reasoning | ❌ Future | — | Post-MVP |
| VLM Explanation | 🔜 Planned | `feat/reasoning` or `feat/backend` | Template for MVP |
| InternVideo3 | ❌ Not in MVP | — | Post-MVP |

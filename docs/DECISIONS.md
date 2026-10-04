# DECISIONS — Temporal Vision Decision Log

All significant technical and scope decisions are logged here.  
Format: ID · Decision · Reason · Status · Future notes

---

## DEC-001 — Use Pretrained Perception Models

**Decision:**  
Use pretrained Grounding DINO-B, SAM 2.1 Base+, and Depth Pro. Do not fine-tune or train custom perception models.

**Reason:**  
Pretrained models provide a strong baseline for the demo objects (person, laptop, table) without requiring any labelled dataset or training time. The competition deadline makes training infeasible.

**Status:** Accepted

---

## DEC-002 — Persistent Temporal Scene Graph as World Model

**Decision:**  
The intermediate representation between perception and reasoning is a persistent temporal scene graph where nodes are objects with stable IDs and edges are spatial/temporal relations.

**Reason:**  
Raw bounding boxes from a frame-level detector are insufficient for temporal reasoning. The scene graph provides a structured memory of how objects relate over time, enabling event detection without frame-by-frame heuristics.

**Status:** Accepted — design documented in `ARCHITECTURE.md`  
**MVP note:** The graph is implicit in the `PerceptionFrame` sequence for the overnight demo. An explicit graph data structure is planned for the reasoning layer.

---

## DEC-003 — Rule-Based Event Engine for MVP

**Decision:**  
Detect APPROACH, REACH, PICK_UP, and CARRY using deterministic geometric rules operating on centroid distances, bounding box motion, and depth deltas.

**Reason:**  
A learned event classifier (GNN, temporal transformer) requires training data and training time. Deterministic rules produce reliable results for the controlled demo scenario and can be implemented in hours.

**Status:** Accepted  
**Future:** Replace with a GNN / Graph Transformer classifier after the demo baseline is validated.

---

## DEC-004 — No TAP / TAPIR / TAPNext++ Tonight

**Decision:**  
Do not introduce point tracking (TAP, TAPIR, TAPNext++) in the overnight MVP.

**Reason:**  
SAM 2.1 provides object-level mask propagation which is sufficient for the demo. Adding a separate point tracker introduces installation complexity, dependency conflicts, and integration risk without a concrete benefit for the 3-object demo.

**Status:** Accepted  
**Future:** TAPNext++ may add value for fine-grained hand/finger tracking in a more complex scenario.

---

## DEC-005 — No GNN or Graph Transformer Tonight

**Decision:**  
Do not train or integrate a GNN or Graph Transformer for the overnight MVP.

**Reason:**  
A working end-to-end pipeline with deterministic reasoning is more valuable for the demo than a partially-trained GNN. Correctness and integration stability take priority.

**Status:** Accepted  
**Future:** Core part of the full architecture (see `ARCHITECTURE.md`, Section A).

---

## DEC-006 — Depth Pro is Optional

**Decision:**  
Depth Pro must not block the pipeline. If the checkpoint is absent or the package is not installed, all depth fields are `null` and the pipeline continues.

**Reason:**  
Depth adds valuable 3D context but is not required for event detection in the 2D case (centroid distances, bounding box overlap are sufficient). Installation of Depth Pro is non-trivial and should not risk breaking the demo.

**Status:** Accepted  
**Implementation:** `DepthProAdapter` in `backend/perception/depth.py` returns `(None, None)` if unavailable, logs one clear warning.

---

## DEC-007 — JSON File I/O as Module Interface

**Decision:**  
Modules communicate through JSON files (`perception_output.json`, `events.json`) rather than direct function calls or shared memory.

**Reason:**  
Each team member works on a separate branch and may run their component independently. File I/O enables independent testing and makes integration explicit. Schemas are versioned in `DATA_MODEL.md`.

**Status:** Accepted  
**Trade-off:** Not suitable for real-time. Acceptable for the overnight demo.

---

## DEC-008 — MPS (Apple Silicon) as Primary Device

**Decision:**  
The primary inference device is MPS (Metal Performance Shaders) via PyTorch on Apple Silicon. CUDA is supported but not the primary target. CPU is the final fallback.

**Reason:**  
The development machine is Apple Silicon (arm64 macOS). CUDA is not available. MPS provides significant speedup over CPU for both GroundingDINO and SAM 2.1.

**Status:** Accepted  
**Implementation:** `backend/perception/device.py` selects CUDA → MPS → CPU automatically.  
**Note:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is required and set automatically by the pipeline.

---

## DEC-009 — Stable Deterministic Object IDs

**Decision:**  
Object IDs (`person_01`, `laptop_01`, `table_01`) are assigned deterministically by sorting detections by `(class, bbox_x1)` at the seed frame, independent of detector output order.

**Reason:**  
If IDs changed between runs or frames, the downstream reasoning layer could not build consistent state. Deterministic assignment makes the output reproducible.

**Status:** Accepted  
**Implementation:** `SAM2Tracker._assign_stable_ids()` in `backend/perception/tracker.py`.

---

## DEC-010 — VLM Must Not Override Vision

**Decision:**  
If a VLM or LLM is used for explanation, it must receive structured evidence as input and must not override geometry, object identity, or relations produced by the vision pipeline.

**Reason:**  
VLMs can hallucinate spatial relationships that contradict the visual evidence. The explanation must be grounded in what the vision system actually observed. Trust the vision, use the language model only for narration.

**Status:** Accepted  
**MVP:** Template-based explanation only (no LLM call). LLM integration is post-MVP.

# MEMORY — Temporal Vision Project Memory

Compact reference. Keep it current. Do not put secrets here.

---

## Project

**Name:** Temporal Vision  
**Tagline:** Temporal Visual Reasoning Engine  
**Repo:** `github.com/Bhavikesh/temporal-vision`  
**Deadline:** Tomorrow morning (competition demo)

---

## Core Idea

Move from frame-level object detection to temporal scene understanding:

```
WHO + WHAT + WHERE + WHEN + HOW → WHAT HAPPENED
```

Primary representation: **Persistent Temporal Scene Graph**  
- Nodes = objects with stable IDs  
- Edges = spatial and temporal relations  
- Evolves frame by frame  

---

## Demo

**Scenario:** Indoor, single camera, 3 objects

```
Person enters
  → approaches table
  → reaches toward laptop
  → picks up laptop
  → carries laptop away
```

**Objects:** PERSON · LAPTOP · TABLE  
**Events:** APPROACH · REACH · PICK_UP · CARRY

---

## Current Implementation Status

| Component | Status |
|-----------|--------|
| Grounding DINO-B detection | ✅ implemented |
| SAM 2.1 video tracking | ✅ implemented |
| Depth Pro adapter | ✅ optional |
| PerceptionFrame JSON output | ✅ implemented |
| Debug visualisation | ✅ implemented |
| Temporal state / relations | 🔜 Member 2 |
| Event detection (rule-based) | 🔜 Member 2 |
| Evidence layer | 🔜 Member 2 |
| FastAPI backend | 🔜 Member 3 |
| Frontend UI | 🔜 Member 4 |

---

## Perception Stack

| Model | Checkpoint | Size |
|-------|-----------|------|
| Grounding DINO-B (SwinB) | `groundingdino_swinb_cogcoor.pth` | ~909 MB |
| SAM 2.1 Base+ | `sam2.1_hiera_base_plus.pt` | ~309 MB |
| Depth Pro | `depth_pro.pt` | TBD (optional) |

---

## Environment

```
Conda env   : yolo_env
Python      : 3.10.20
PyTorch     : 2.11.0
Device      : MPS (Apple Silicon)
OS          : macOS ARM64
```

**Critical version pins:**
- `transformers==4.38.2` — must NOT be upgraded (GroundingDINO BERT compat)
- `scipy` — must be conda-installed (pip binary broken on ARM)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — required for SAM 2.1 on MPS

---

## Key File Locations

```
backend/perception/         — Member 1 code
docs/                       — all documentation
checkpoints/                — model weights (not committed)
output/                     — runtime artefacts (not committed)
scripts/test_perception.py  — smoke test
requirements-perception.txt — install instructions
```

---

## Run Commands

```bash
# Run perception pipeline
conda run -n yolo_env python -c "
import os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import sys; sys.path.insert(0, '.')
from backend.perception.config import PerceptionConfig
from backend.perception.pipeline import PerceptionPipeline
cfg = PerceptionConfig(input_video='demo.mp4', output_dir='output', debug=True)
PerceptionPipeline(cfg).run()
"

# Run smoke test
conda run -n yolo_env python scripts/test_perception.py \
  --video demo.mp4 --output output_smoke --debug
```

---

## Differentiators

- Persistent object **identity** (not just detection per frame)
- **Depth** as a 3D spatial cue
- **Relations** between objects over time
- **Evidence-grounded** event conclusions
- **Explanation** grounded in observed data

---

## What We Deliberately Excluded

- TAP / TAPIR / TAPNext++ — unnecessary for controlled demo
- GNN / Graph Transformer — correctness > learning tonight
- InternVideo3 — no time for integration
- VLM as reasoning engine — must not hallucinate
- Database — file I/O is sufficient
- Real-time inference — correctness first

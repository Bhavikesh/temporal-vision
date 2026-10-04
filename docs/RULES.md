# RULES — Temporal Vision Engineering Rules

These rules apply to all members of the team. They are not suggestions.

---

## Code

**RULE-01 — Keep code minimal.**  
Write the smallest correct implementation. Every line must justify its existence. No speculative abstractions.

**RULE-02 — Prefer deterministic logic over unnecessary ML.**  
If a rule-based condition correctly detects an event, use it. Do not add a model to replace logic that already works.

**RULE-03 — No model is added without a concrete reason.**  
Identify exactly what signal the model provides, what data it requires, and what it replaces. If the answer is unclear, do not add the model.

**RULE-04 — Do not train foundation models.**  
Use pretrained checkpoints only. No fine-tuning of perception models in this project.

**RULE-05 — Do not claim unimplemented functionality.**  
If a component is planned but not yet written and tested, it is **PLANNED**, not **IMPLEMENTED**. This applies to documentation, demos, and presentations.

**RULE-06 — Keep modules independent.**  
Each module communicates through JSON contracts defined in `docs/DATA_MODEL.md`. No module imports internals of another module's code.

---

## Data

**RULE-07 — Use explicit JSON contracts.**  
Every cross-module interface is a JSON document conforming to a schema defined in `DATA_MODEL.md`. Schema changes require updating that document first.

**RULE-08 — Object IDs must be stable.**  
An object assigned ID `person_01` in frame 0 keeps that ID for the entire video. IDs are assigned deterministically (sorted by class then x-position at first detection).

**RULE-09 — Evidence must be retained.**  
Every detected event must have a corresponding evidence block containing the frame index, subject/object IDs, and the quantitative values that triggered the rule.

---

## Repository

**RULE-10 — Do not commit model weights.**  
No `.pt`, `.pth`, `.ckpt`, or `.safetensors` file may be committed to git. They belong in `checkpoints/` which is in `.gitignore`.

**RULE-11 — Do not commit videos or large generated artefacts.**  
Input videos, output masks, and debug frames are excluded from git. They are runtime artefacts.

**RULE-12 — Do not commit secrets.**  
No API keys, tokens, or credentials in any file. Use environment variables.

**RULE-13 — Do not introduce unnecessary infrastructure.**  
No database is required for the MVP. No cloud service. No message queue. The pipeline runs locally with file I/O.

---

## Quality

**RULE-14 — Validate real output before declaring success.**  
Running the pipeline without errors is not success. The output JSON must be inspected and validated against the schema. The detection result must be checked against real video frames.

**RULE-15 — Preserve current working code.**  
If a component is working, do not refactor it during the competition window unless there is a concrete defect. Stability beats elegance tonight.

**RULE-16 — Favour correctness over premature optimisation.**  
Achieve correct results first. Frame sampling, resolution reduction, and batching are second-order concerns.

---

## Reasoning

**RULE-17 — VLMs and LLMs must not be the source of truth for geometry or identity.**  
A VLM cannot override an object's bounding box, depth, or ID. It may only produce natural language over structured evidence it is explicitly provided.

**RULE-18 — The perception layer must not perform reasoning.**  
Grounding DINO, SAM 2.1, and Depth Pro provide observations. They do not decide what events occurred. That is the responsibility of the temporal reasoning layer.

**RULE-19 — Depth failure must not break the pipeline.**  
If Depth Pro is unavailable or fails, all `depth` fields are `null`. The pipeline continues. The reasoning layer must handle nullable depth.

---

## Branching

**RULE-20 — Work only on your assigned branch.**  
Member 1 → `feat/perception`  
Member 2 → `feat/reasoning`  
Member 3 → `feat/backend`  
Member 4 → `feat/frontend`  

Do not push directly to `main`. Merge only after the component has passed its own smoke test.

# Multimodal LLM for Industrial Task Planning

**COS40005 Computing Technology Project B — Capstone**  
Swinburne University of Technology × ARENA2036 × University of Stuttgart

---

## Overview

An intelligent simulation-based robotic system that interprets natural language instructions and executes corresponding tasks in a simulated industrial environment. The system integrates Large Language Models (LLMs), computer vision, and rule-based task planning into a modular five-stage pipeline.

```
User instruction
    → [1] LLM Parse        extract action, object, destination, spatial relation
    → [2] Vision Lookup    identify objects and positions in the workspace
    → [3] Task Planning    generate step-by-step robot action sequence
    → [4] Execution        send commands to robot (MockRobot / real simulation)
    → [5] Feedback         validate completion, log result, retry on failure
```

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Minh Hoang Duong | 104487115 | Team Leader |
| Lakshit Bansal | 105028858 | Vision Module |
| Ved Jay Makhijani | 104762184 | LLM Module / Pipeline |
| Dinith Thejana | 105231766 | Simulation Backend |
| Kaveesha Dharmadasa | 105271678 | Documentation / Scene Representation |

**Supervisors:** Prof. Prem Prakash Jayaraman · Prof. Boris Eisenbart · Muhammad Saeed  
**Industry Partner:** ARENA2036 / University of Stuttgart

---

## Project Structure

```
P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning/
│
├── main.py                              ← Pipeline entry point
├── conftest.py                          ← Pytest configuration
├── pytest.ini                           ← Test markers
├── requirements.txt                     ← Dependencies
├── README.md
├── .env                                 ← API keys (never committed)
├── .env.example                         ← Template — copy to .env
│
├── llm_backend/                         ← LLM instruction parser (Sprint 1)
│   ├── __init__.py
│   ├── custom_LLM_parser.py             ← parse_instruction() — main entry point
│   ├── schema.py                        ← ParsedInstruction Pydantic model
│   ├── prompts.py                       ← System prompt + 6 few-shot examples
│   ├── edge_cases.py                    ← Empty/vague/synonym handling
│   ├── tracker.py                       ← Cross-domain pipeline task tracker
│   ├── demo.py                          ← LLM module demo
│   ├── hello_world.py                   ← API connection test
│   └── backends/                        ← Per-model API implementations
│       ├── openai_backend.py            ← GPT-4o via OpenAI API
│       ├── gemini_backend.py            ← Gemini via Google API
│       ├── deepseek_backend.py          ← DeepSeek via OpenAI-compatible API
│       └── huggingface_backend.py       ← Local HuggingFace models
│
├── llm_backend/LLM_eval/                ← Multi-model evaluation (Sprint 2/3)
│   ├── comparison_report.py             ← Full evaluation report runner
│   ├── evaluator.py                     ← Runs models against test cases
│   ├── metrics.py                       ← 10 metrics per model per category
│   ├── test_cases.py                    ← 25 labelled test cases, 6 categories
│   ├── model_registry.py                ← Model loader for evaluation
│   ├── baseline_parser.py               ← Rule-based parser (no LLM) for comparison
│   ├── eval_report.py                   ← End-to-end + baseline evaluation runner
│   ├── evaluation_metrics.csv           ← Generated — metrics output
│   └── evaluation_results.json         ← Generated — raw per-case results
│
├── task_planner/                        ← Task planning module (Sprint 2/3)
│   ├── __init__.py
│   └── planner.py                       ← Rule-based planner with spatial relations
│
├── simulation_backend/                  ← Execution module (Sprint 2)
│   ├── __init__.py
│   ├── action_schema.py                 ← RobotCommand, ActionPlan Pydantic schemas
│   ├── mock_robot.py                    ← MockRobot simulator (no PyBullet required)
│   ├── executor.py                      ← Runs ActionPlan step by step
│   └── robots/                          ← Real robot implementations
│       ├── kuka_robot.py
│       ├── ur5_robot.py
│       └── franka_pand_robot.py
│
├── vision_backend/                      ← Vision module (Sprint 1/2)
│   ├── __init__.py
│   ├── vision_output.py                 ← YOLOv8 object detection
│   ├── scene_representation.py          ← Object label + position mapping
│   ├── spatial_relationships.py         ← Left/right/near spatial reasoning
│   ├── invalid_actions.py              ← Action validation
│   └── safety_checks.py                ← Pre-execution safety validation
│
├── tests/                               ← Test suite
│   ├── test_llm_module.py               ← 24 unit tests (Sprint 1)
│   ├── test_sprint2.py                  ← 38 unit tests (Sprint 2)
│   ├── integration_tests.py             ← 67 integration tests (Sprint 3)
│   └── test_2.py
│
└── documentation/                       ← Reports and docs
    ├── vision_framework_comparison.md
    └── tool_recommendations.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MinhWorkingAI/P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning.git
cd P54-Embodied-Multimodal-LLM-for-Industrial-Task-Planning
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:
```
# Controls which LLM the pipeline uses
LLM_BACKEND=openai      # or: gemini, deepseek, huggingface

# OpenAI (GPT-4o)
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.0

# Google Gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-1.5-pro
GEMINI_TEMPERATURE=0.0

# DeepSeek
DEEPSEEK_API_KEY=your-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.0
```

Each team member uses their own `.env` with their own keys. The `.env` file is in `.gitignore` and is never committed.

---

## Running the Pipeline

### Single instruction
```bash
python main.py "pick up the red block and place it in the left tray"
```

### Interactive mode
```bash
python main.py --interactive
```
Type any instruction at the prompt. Type `status` to see the tracker summary. Type `quit` to exit.

### Quiet mode (minimal output)
```bash
python main.py --quiet "locate the yellow block"
```

### Switch model without changing code
Set `LLM_BACKEND` in your `.env`:
```
LLM_BACKEND=gemini
```
Then run normally — no code change needed.

### Test all 6 instruction categories
```bash
# Simple
python main.py "pick up the red block and place it in the left tray"

# Spatial
python main.py "place the red block to the left of the blue block"

# Synonym
python main.py "grab the yellow block and drop it in the right tray"

# Multi-step / spatial
python main.py "move the blue block near the workstation"

# Ambiguous — exits gracefully at Stage 1
python main.py "put that thing over there"

# Edge case — all caps normalised
python main.py "PICK UP THE RED BLOCK AND PLACE IT IN THE LEFT TRAY"
```

---

## Running Tests

### All unit tests (no API key or PyBullet required)
```bash
pytest tests/ -v -m "not integration"
```
Expected: **129 passed**

### Integration tests only (no API required)
```bash
pytest tests/integration_tests.py -v -m "not integration"
```

### Full test suite including LLM calls (requires API key)
```bash
pytest tests/ -v
```

### Single test class
```bash
pytest tests/integration_tests.py::TestSpatialRelationPlanning -v
pytest tests/integration_tests.py::TestFullPipelineIntegration -v
pytest tests/test_sprint2.py::TestMockRobot -v
```

---

## Running the Evaluation

### Baseline only (no API key needed — instant)
```bash
cd llm_backend/LLM_eval
python eval_report.py --baseline-only
```

### Full evaluation across all models (requires API keys)
```bash
cd llm_backend/LLM_eval
python eval_report.py
```

### Specific models only
```bash
python eval_report.py --models openai gemini
```

### Export results to CSV and JSON
```bash
python eval_report.py --export
```
Outputs: `evaluation_metrics.csv` and `evaluation_results.json`

### Sprint 2 multi-model comparison report
```bash
cd llm_backend/LLM_eval
python comparison_report.py
```

---

## Evaluation Categories

| Category | Cases | Description |
|---|---|---|
| simple | 5 | Basic single-action instructions |
| spatial | 5 | Positional relationships (left of, near, on top of) |
| synonym | 5 | Non-standard action words (grab, drop, find) |
| multi_step | 3 | Instructions implying two sequential actions |
| ambiguous | 3 | Vague or underspecified instructions |
| edge_case | 4 | Unknown objects, formatting variations, boundaries |

---

## Pipeline Stages

### Stage 1 — LLM Parse (`llm_backend/custom_LLM_parser.py`)
Sends instruction to GPT-4o / Gemini / DeepSeek with a structured system prompt and 6 few-shot examples. Returns `ParsedInstruction` with action, object, destination, spatial relation, and confidence. Handles empty, vague, and synonym edge cases before calling the API.

### Stage 2 — Vision Lookup (`vision_backend/`)
Detects objects in the simulation using YOLOv8 and OpenCV. Returns a scene map of object labels and (x, y) positions. Currently uses a stub in `main.py` — swap `get_scene()` for `get_current_scene()` from `vision_backend.scene_representation` to connect the real module.

**To connect real vision module:**
```python
# In main.py, replace get_scene() with:
from vision_backend.scene_representation import get_current_scene
def get_scene() -> dict:
    return get_current_scene()
```

### Stage 3 — Task Planning (`task_planner/planner.py`)
Rule-based planner combining `ParsedInstruction` and scene map into an ordered `ActionPlan`. Generates `locate → move → pick → move → place` sequences. Sprint 3 added spatial offset handling — "left of", "right of", "near", "on top of" etc. compute offset positions relative to reference objects.

### Stage 4 — Execution (`simulation_backend/`)
`Executor` runs each `RobotCommand` sequentially on `MockRobot`. Stops on first failure and returns `ExecutionResult`. Swap `MockRobot` for a real robot implementation from `simulation_backend/robots/` to connect PyBullet.

### Stage 5 — Feedback (`llm_backend/tracker.py`)
Validates task completion, logs all 5 stages to `task_log.json` with a unique `task_id`. Triggers retry flag on failure or low confidence.

---

## Example Output

```
════════════════════════════════════════════════════════════
  PIPELINE START
  Instruction : pick up the red block and place it in the left tray
  Model       : openai
  Task ID     : d3c4f72a
════════════════════════════════════════════════════════════

  [1/5] LLM Parse (openai)
       Action      : pick
       Object      : red block
       Destination : left tray
       Spatial     : in
       Confidence  : high
       Latency     : 2926ms

  [2/5] Vision Lookup  [STUB]
       Objects in scene: ['red block', 'blue block', ..., 'left tray', 'right tray']

  [3/5] Task Planning
       Steps generated : 5
       Step 1: LOCATE 'red block'
       Step 2: MOVE 'red block' → (2.5, 1.0)
       Step 3: PICK 'red block'
       Step 4: MOVE 'left tray' → (6.0, 1.0)
       Step 5: PLACE 'left tray'

  [4/5] Execution  [MockRobot]
       ✓ Plan completed successfully in 0ms

  [5/5] Feedback & Validation
       ✓ Task completed — 5/5 steps

════════════════════════════════════════════════════════════
  PIPELINE COMPLETE  ✓  Task ID: d3c4f72a
════════════════════════════════════════════════════════════
```

---

## Spatial Relation Handling

The task planner supports positional instructions using offset-based spatial reasoning:

| Relation | Offset (dx, dy) | Example |
|---|---|---|
| left of | (−1.5, 0.0) | "place the red block to the left of the blue block" |
| right of | (+1.5, 0.0) | "move the green block to the right of the workstation" |
| near | (+0.8, +0.8) | "put the yellow block near the workstation" |
| next to | (+1.2, 0.0) | "place it next to the blue block" |
| on top of | (0.0, 0.0) | "stack the red block on top of the blue block" |
| in front of | (0.0, −1.5) | "move it in front of the workstation" |
| in | (0.0, 0.0) | "place the block in the left tray" |

---

## Test Coverage

| File | Tests | API needed | PyBullet needed |
|---|---|---|---|
| `tests/test_llm_module.py` | 24 | No | No |
| `tests/test_sprint2.py` | 38 | No | No |
| `tests/integration_tests.py` | 67 | No | No |
| LLM integration (`-m integration`) | 11 | Yes | No |
| **Total (no API)** | **129** | — | — |

---

## Key Design Decisions

**Rule-based task planner** — Deterministic, zero API cost, fully testable without external dependencies, and sufficient for the constrained pick-and-place simulation environment. An LLM-based planner can be substituted in future iterations.

**MockRobot** — Eliminates cross-platform PyBullet dependency during development. The `Executor` interface is identical for mock and real robots — swap one import.

**Pydantic schemas** — `ParsedInstruction`, `RobotCommand`, `ActionPlan` enforce strict interface contracts between modules. Validation errors surface at module boundaries rather than deep in pipeline logic.

**LLM_BACKEND in .env** — Model selection is a deployment-time decision. Each team member sets their own key and model. The codebase is model-agnostic.

**Baseline parser** — Rule-based keyword matching with no LLM provides a research comparison point. Demonstrates that LLMs provide 30–40pp accuracy gains on spatial, synonym, and ambiguous instruction categories.

---

## Sprint Progress

| Sprint | Weeks | Deliverables |
|---|---|---|
| Sprint 1 | 4–6 | LLM parser, schema, prompts, edge cases, multi-model evaluation framework, 24 unit tests |
| Sprint 2 | 7–9 | Task planner, mock robot, executor, action schema, 5-stage pipeline, tracker, 38 unit tests |
| Sprint 3 | 10–12 | Spatial relation handling, multi-step planning, baseline parser, full evaluation suite, 67 integration tests |

---

## Literature

- Ahn et al. — *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (SayCan)
- Driess et al. — *PaLM-E: An Embodied Multimodal Language Model*
- Radford et al. — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Wei et al. — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- Yao et al. — *ReAct: Synergizing Reasoning and Acting in Language Models*
- Bode et al. — *A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics* (arXiv 2410.22997)

---

## Contact

| Supervisor | Email |
|---|---|
| Muhammad Saeed | msaeed@swin.edu.au |
| Prof. Boris Eisenbart | beisenbart@swin.edu.au |
| Prof. Prem Prakash Jayaraman | pjayaraman@swin.edu.au |

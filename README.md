# SVHN Digit Recognition — ML Inference & Evaluation Platform

An end-to-end machine learning engineering project built around Street View
House Numbers (SVHN) digit recognition.

The project began as an ANN-vs-CNN computer vision study and evolved into a
production-oriented ML system with reproducible evaluation, persisted metrics,
automated technical reporting, typed inference contracts, FastAPI serving,
Model Context Protocol (MCP) interoperability, policy-driven inference
orchestration, durable runtime state, and human-in-the-loop controlled actions.

The goal is no longer only to train an accurate classifier. The repository
demonstrates how a trained ML model can be evaluated, packaged, served,
observed, exposed to independent AI agents, and operated through maintainable
and controlled software architecture.

---

## Highlights

- **95.14% accuracy / ~95.14% macro F1** on an independent stratified
  holdout containing 24,000 samples.
- Improved from **67.36% ANN baseline → 74.84% improved ANN → 95.14% CNN**.
- FastAPI/Uvicorn inference and evaluation API.
- Model Context Protocol (MCP) server exposing reusable ML-system capabilities.
- Independent MCP interoperability verified with **Claude Code** and
  **VS Code/Copilot**.
- Policy-driven inference orchestration with `ACCEPTED`, `UNCERTAIN`,
  and `FAILED` outcomes.
- Application-owned action authorization with `READ_ONLY`, `CONTROLLED`,
  and `PROHIBITED` risk classes.
- Human-in-the-loop approval workflow for controlled runtime changes using
  LangGraph interrupt/resume semantics.
- Durable runtime configuration persistence using SQLite.
- Durable LangGraph approval/checkpoint persistence across MCP process restarts.
- Typed Python domain models using dataclasses and Pydantic API contracts.
- SQLite-backed persisted evaluation and per-prediction reliability observations.
- Confidence, threshold, selective-risk, and calibration analysis.
- Deterministic failure and fallback reliability evaluation.
- Automated Typst technical-report generation.
- Model architecture inspection and automated evaluation intelligence.
- Dockerized API runtime.
- **222 automated tests** covering preprocessing, inference, orchestration,
  reliability, MCP interoperability, controlled actions, HITL approval,
  persistence, restart recovery, API behavior, and real-model integration.

---

## System Architecture

```text
                         External Consumers
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
             FastAPI Clients              MCP Consumers
                  │                             │
                  ▼                    ┌────────┴────────┐
             FastAPI API               │                 │
                  │                    ▼                 ▼
                  │              Claude Code      VS Code/Copilot
                  │                    │                 │
                  │                    └────────┬────────┘
                  │                             │
                  │                             ▼
                  │                      ┌─────────────┐
                  │                      │ MCP Server  │
                  │                      └──────┬──────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
      Inference Service   Evaluation Service   Controlled Actions
             │                   │                   │
             │                   │                   ▼
             │                   │              ActionPolicy
             │                   │                   │
             │                   │                   ▼
             │                   │        ControlledActionExecutor
             │                   │                   │
             │                   │                   ▼
             │                   │            LangGraph HITL
             │                   │           interrupt / resume
             │                   │                   │
             ▼                   ▼                   ▼
      Runtime Config       Evaluation SQLite   Approval Checkpoints
             │                                      SQLite
             ▼
      runtime_state.db
             │
             ▼
      Inference Policy
             │
             ▼
       SVHN Predictor
             │
             ▼
     Image Preprocessing
             │
             ▼
       TensorFlow CNN
```

The architecture deliberately separates:
- model execution;
- inference and fallback policy;
- runtime configuration;
- evaluation and reliability persistence;
- external API and MCP contracts;
- action authorization;
- human approval workflow state;
- durable operational state.

FastAPI and MCP are external adapters over application services rather than
owners of model, inference, or authorization policy.

LangGraph owns pause/resume workflow mechanics for controlled actions, while
application code remains authoritative for risk classification, approval
validation, and controlled execution.

---

## Model Development

The original modeling work compares fully connected Artificial Neural
Networks (ANNs) with Convolutional Neural Networks (CNNs) for grayscale
SVHN digit classification.

| Model | Accuracy |
|---|---:|
| ANN Model 1 | 67.36% |
| ANN Model 2 | 74.84% |
| CNN Model 1 | 93.78% |
| CNN Model 2 / Selected CNN | **95.14%** |

The CNN substantially outperformed the ANN models because convolutional
layers preserve and learn spatial image structure rather than operating
only on flattened pixels.

---

## Authoritative Model Evaluation

The selected CNN was reconstructed and evaluated using the project's
historical stratified holdout protocol.

### Historical Stratified Holdout

| Metric | Result |
|---|---:|
| Accuracy | **95.14%** |
| Macro F1 | **~95.14%** |
| Evaluation samples | **24,000** |
| Correct predictions | **22,833** |
| Incorrect predictions | **1,167** |
| Independent of training | **Yes** |

This is the authoritative performance result used by the project.

The repository also records a **96.84%** result on the original HDF5 test
split. That result is retained as a diagnostic only because those samples
were included in the historical combined training/evaluation procedure.
It is therefore **not presented as an independent test result**.

This distinction is intentional: an evaluation metric is incomplete
without the evaluation protocol that produced it.

---

## Inference Orchestration

The serving architecture separates three concepts that are often combined
in simple ML APIs:

```text
PredictionResult
      │
      │ What did the model produce?
      ▼
InferenceAttempt
      │
      │ Did model execution succeed?
      ▼
InferenceDecision
        What should the system do with the result?
```

The orchestration policy currently supports three final states:

- `ACCEPTED` — successful prediction satisfying the configured confidence
  policy.
- `UNCERTAIN` — usable prediction that does not satisfy the confidence
  threshold and requires review.
- `FAILED` — model execution did not produce a usable prediction.

The baseline evaluated confidence threshold is **0.90**. Reliability analysis
on the authoritative 24,000-sample Historical Stratified Holdout shows that
this operating point provides **91.38% coverage**, **98.44% accepted accuracy**,
**1.56% selective risk**, and an **8.62% uncertainty rate**.

The `0.90` value is the project's validated bootstrap/default operating point,
not a permanently fixed runtime value. The active serving threshold may be
changed through the controlled human-in-the-loop workflow and is persisted
across MCP process restarts.

Threshold selection remains an operational policy decision rather than a
claim of a globally optimal value. Final selection depends on reliability,
risk, coverage, and review-cost requirements of the serving environment.

---

## Failure Boundaries

The system distinguishes invalid caller input from actual model failures.

```text
Unsupported media type
        │
        └──► HTTP 415

Malformed image input
        │
        └──► HTTP 400

Valid input + model/runtime failure
        │
        └──► FAILED InferenceDecision

Inference service unavailable
        │
        └──► HTTP 503
```

This prevents bad client input from being incorrectly recorded as an ML
model failure.

---

## Runtime Configuration

Inference behavior is represented by a typed, immutable `InferenceConfig`
domain object.

Bootstrap defaults:

```text
model_name            = cnn
confidence_threshold  = 0.90
```

Configuration is validated when constructed so invalid operating values fail
early rather than leaking into runtime inference behavior.

Runtime configuration is managed through `InferenceConfigService`. The service
loads persisted configuration when available and falls back to
`DEFAULT_INFERENCE_CONFIG` only when no durable runtime state exists.

```text
Process startup
      │
      ▼
InferenceConfigStore
      │
      ├── persisted config exists ───► restore persisted value
      │
      └── no persisted config ───────► use default 0.90
```

Approved configuration changes are persisted to:

```text
generated/runtime_state.db
```

The write ordering is intentionally:

```text
validate new configuration
        ↓
persist new configuration
        ↓
replace active in-memory configuration
```

This ensures that a persistence failure does not silently replace the active
runtime state.

Process-restart verification confirmed that an approved threshold change
survives termination of the MCP Python process and is restored by the newly
created runtime.

---

## Human-in-the-Loop Controlled Actions

Runtime mutation is governed by deterministic application-owned authorization
rather than by the LLM, MCP client, or LangGraph itself.

Actions are classified into three risk levels:

| Risk | Meaning |
|---|---|
| `READ_ONLY` | May execute without human approval |
| `CONTROLLED` | Requires explicit approval before execution |
| `PROHIBITED` | Must not execute |

Current controlled action:

```text
set_confidence_threshold
```

The workflow is:

```text
MCP controlled-action request
        │
        ▼
ActionPolicy
        │
        ▼
ApprovalRequest
        │
        ▼
LangGraph interrupt()
        │
        ▼
Human approval required
        │
        ├── rejected ─────────────► no configuration change
        │
        └── approved
                │
                ▼
        ControlledActionExecutor
                │
                ▼
        exact request/decision validation
                │
                ▼
        InferenceConfigService
                │
                ▼
        persistent runtime update
```

Approval is bound to the exact action and exact arguments originally requested.
A decision for one request cannot authorize a different controlled action or
different argument set.

Approval workflow state is persisted through LangGraph's SQLite checkpointer:

```text
generated/approval_checkpoints.db
```

This allows a pending approval to survive MCP process termination.

The restart-recovery path was verified end to end:

```text
request threshold change
        ↓
approval_required
        ↓
process terminates before approval
        ↓
new MCP runtime starts
        ↓
original thread_id resumes
        ↓
approval executes
        ↓
new configuration persists
```

LangGraph is responsible for durable pause/resume workflow state. Application
code remains responsible for authorization rules and controlled execution.

---

## FastAPI Service

Start the API locally with:

```bash
uvicorn api.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

for the interactive Swagger/OpenAPI interface.

### Prediction

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/predict` | Raw CNN prediction with probabilities |
| `POST` | `/predict/orchestrated` | Prediction plus orchestration decision |

### Evaluation

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/evaluation/summary` | Authoritative evaluation metrics |
| `GET` | `/evaluation/insights` | Derived evaluation intelligence |
| `GET` | `/evaluation/classes` | Per-class metrics |
| `GET` | `/evaluation/classes/{digit}` | Metrics for one digit |
| `GET` | `/evaluation/confusion-matrix` | Persisted confusion matrix |

### Model & Runtime

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/model/summary` | CNN architecture and capacity |
| `GET` | `/inference/config` | Active serving configuration |
| `GET` | `/health` | Service health |

The API uses Pydantic response models so public contracts remain separate
from internal domain and persistence representations.

---

## Automated Evaluation & Reporting

Evaluation is implemented as a reusable subsystem rather than as notebook-only
code.

The pipeline:

```text
Saved CNN
   │
   ▼
Evaluation Protocol
   │
   ▼
Evaluation Domain Objects
   │
   ▼
SQLite Persistence
   │
   ├────────────► FastAPI
   │
   └────────────► Report Data
                        │
                        ▼
                      Typst
                        │
                        ▼
                 Technical PDF Report
```

Persisted results include:

- evaluation protocol and provenance
- aggregate accuracy and macro F1
- per-class precision, recall, F1, and sample counts
- confusion matrix
- correct/incorrect prediction counts
- strongest and weakest classes
- most common misclassification

Persisting evaluation results prevents expensive CNN evaluation from being
repeated every time the report is generated.

---

## Reliability Evaluation

Reliability evaluation extends model accuracy into confidence behavior,
selective prediction, calibration, persistence, and controlled orchestration
failure analysis.

### Confidence & Threshold Analysis

The authoritative 24,000-sample Historical Stratified Holdout produced:

| Metric | Result |
|---|---:|
| Accuracy | **95.14%** |
| Mean confidence | **96.75%** |
| Mean confidence — correct predictions | **98.10%** |
| Mean confidence — incorrect predictions | **70.45%** |

High confidence does not guarantee correctness. The evaluation identified
**148 incorrect predictions with confidence >= 0.99**.

At the baseline evaluated **0.90** orchestration threshold:

| Metric | Result |
|---|---:|
| Coverage | **91.38%** |
| Accepted accuracy | **98.44%** |
| Selective risk | **1.56%** |
| Uncertainty rate | **8.62%** |
| Incorrect accepted predictions | **342** |

Increasing the threshold generally increases accepted-prediction accuracy and
reduces selective risk at the cost of lower coverage and a higher review rate.

### Calibration

Confidence calibration was measured using 10 equal-width bins:

| Metric | Result |
|---|---:|
| Expected Calibration Error (ECE) | **1.65%** |
| Maximum Calibration Error (MCE) | **31.91%** |

The low aggregate ECE indicates relatively good overall calibration. The raw
MCE is driven by a sparsely populated 0.10–0.20 confidence bin containing only
four predictions.

A more substantially populated 0.70–0.80 confidence region contained 417
predictions and showed a **9.14 percentage-point overconfidence gap**.

### Reliability Persistence

Per-prediction reliability observations are persisted in SQLite and associated
with the evaluation protocol that produced them.

The persistence layer stores source facts:

- observation index
- true label
- predicted label
- confidence

Derived properties such as correctness, threshold acceptance, uncertainty,
selective risk, and calibration membership are computed from those observations
rather than redundantly stored.

All **24,000 Historical Stratified Holdout observations** are persisted for
repeatable downstream reliability analysis.

### Failure & Fallback Reliability

The orchestration layer is evaluated with deterministic failure injection
against the real orchestration service.

The controlled scenario suite covers:

1. primary prediction accepted without fallback
2. primary model failure recovered by fallback
3. primary and fallback model failure
4. low-confidence primary prediction recovered by fallback
5. low-confidence primary and fallback predictions requiring review

It verifies both fallback triggers — `primary_model_failed` and
`primary_low_confidence` — as well as explicit `FAILED` and `UNCERTAIN`
outcomes requiring review.

Controlled scenario-suite results:

| Metric | Result |
|---|---:|
| Scenarios | **5** |
| Failure rate | **20.00%** |
| Fallback usage rate | **80.00%** |
| Review-required rate | **40.00%** |

These percentages describe an intentionally constructed reliability scenario
suite. They are **not production incident-rate estimates**.

Run the controlled analysis with:

```bash
python -m tools.reliability.run_failure_analysis
```

---

## Automated Testing

Run the complete suite with:

```bash
python -m pytest -v
```

Current milestone:

```text
222 passed
```

Coverage includes:

- image preprocessing;
- typed prediction results;
- generic model execution;
- inference adapters;
- confidence policy behavior;
- threshold boundaries;
- primary/fallback orchestration behavior;
- invalid-input propagation;
- runtime configuration validation;
- FastAPI response contracts;
- HTTP failure semantics;
- serving metadata;
- real saved-CNN integration;
- confidence-versus-correctness analysis;
- confidence-threshold and selective-risk evaluation;
- calibration analysis;
- SQLite reliability-observation persistence;
- deterministic failure injection;
- fallback recovery and review escalation;
- MCP tool discovery and invocation;
- independent MCP consumer interoperability;
- action risk classification;
- immutable approval request/decision contracts;
- controlled action execution;
- approval mismatch and rejection behavior;
- LangGraph interrupt/resume behavior;
- persistent runtime configuration;
- durable LangGraph checkpoint recovery;
- graph reconstruction from persisted approval state;
- process-restart configuration recovery;
- process-restart pending-approval recovery.

A real-model integration test exercises the path:

```text
image bytes
    ↓
preprocessing
    ↓
TensorFlow CNN
    ↓
adapter
    ↓
execution
    ↓
policy
    ↓
orchestration
    ↓
InferenceDecision
```
The persistence test suite additionally verifies:

```text
approved runtime update
    ↓
SQLite persistence
    ↓
Python process replacement
    ↓
runtime reconstruction
    ↓
same active configuration
```

and 

```text
pending HITL approval
    ↓
SQLite checkpoint
    ↓
Python process replacement
    ↓
same thread_id
    ↓
workflow resume
    ↓
controlled execution
```
---

## Independent MCP Consumer Verification

The SVHN platform exposes inference, evaluation, runtime configuration, and
controlled-action capabilities through a Model Context Protocol (MCP) server.

The interface was independently verified with:

- **Claude Code**
- **VS Code / Copilot Agent**

Both consumers connected to the same local MCP server over `stdio` without
depending on the project's internal LangGraph agent implementation.

### MCP tools

The server currently exposes nine tools:

| Tool | Purpose |
|---|---|
| `predict_digit(image_data)` | Classify an SVHN digit image |
| `get_model_summary()` | Inspect production CNN architecture |
| `get_inference_config()` | Read active inference configuration |
| `get_evaluation_summary()` | Retrieve aggregate evaluation metrics |
| `get_evaluation_insights()` | Retrieve derived evaluation insights |
| `get_digit_metrics(digit)` | Retrieve per-digit metrics |
| `set_confidence_threshold(value)` | Request a controlled threshold change |
| `resolve_confidence_threshold_approval(...)` | Resume a pending approval workflow |
| `get_runtime_debug()` | Inspect runtime identity/state for diagnostics |

### Read-only interoperability

External consumers independently verified:

- MCP connection over `stdio`;
- dynamic capability discovery;
- zero-argument tool invocation;
- parameterized tool invocation;
- structured result handling;
- image inference;
- historical per-class evaluation retrieval;
- multi-tool reasoning and natural-language synthesis.

Example end-to-end workflow:

```text
sample image
    ↓
predict_digit
    ↓
predicted class
    ↓
get_digit_metrics
    ↓
historical performance context
    ↓
external-agent explanation
```

### Controlled-action interoperability

VS Code/Copilot additionally verified the controlled-action workflow:

```text
set_confidence_threshold
        ↓
approval_required
        ↓
human approval
        ↓
resolve_confidence_threshold_approval
        ↓
configuration updated
```

The MCP client may present the approval requirement through its own user
interface, but the controlled action itself remains governed by application
authorization and LangGraph workflow state.

### Process-restart recovery

Durable workflow behavior was independently verified by terminating the MCP
Python process while a controlled action was awaiting approval.

After a new MCP runtime was created:

- persisted inference configuration was restored;
- the original approval `thread_id` remained resumable;
- the pre-crash approval request was recovered from SQLite;
- the controlled action executed successfully after approval;
- the updated threshold was persisted for subsequent runtimes.

### Architectural result

```text
External MCP Consumer
        │
        │ MCP / stdio
        ▼
SVHN MCP Server
        │
        ├── Read-only capabilities
        │
        └── Controlled capabilities
                    │
                    ▼
             Application Policy
                    │
                    ▼
             LangGraph HITL
                    │
                    ▼
            Durable Checkpoint
```

External agents own conversational reasoning and capability selection.

The application remains authoritative for:

- model inference;
- configuration validation;
- confidence policy;
- fallback behavior;
- evaluation facts;
- risk classification;
- authorization;
- controlled action execution.

### Verification status

| Capability | Claude Code | VS Code / Copilot |
|---|---:|---:|
| MCP connection | Passed | Passed |
| Dynamic tool discovery | Passed | Passed |
| Zero-argument invocation | Passed | Passed |
| Parameterized invocation | Passed | Passed |
| Image inference | Passed | Passed |
| Evaluation-context retrieval | Passed | Passed |
| Multi-tool synthesis | Passed | Passed |
| Controlled threshold workflow | — | Passed |
| Human approval / resume | — | Passed |
| Process-restart recovery | — | Passed |
| Dependency on internal agent | None | None |

---

## Docker

Build the service:

```bash
docker build -t svhn-digit-api .
```

Run it:

```bash
docker run -p 8000:8000 svhn-digit-api
```

The container launches the FastAPI application through Uvicorn on port
`8000`.

---

## Repository Structure

```text
.
├── agent/
│   ├── approval_graph.py             # HITL controlled-action workflow
│   ├── graph.py                      # Agent workflow construction
│   ├── nodes.py                      # Agent nodes
│   └── state.py                      # Agent state contracts
│
├── api/
│   └── app.py                        # FastAPI application
│
├── config/
│   ├── inference_config.py           # Typed inference configuration
│   ├── inference_config_service.py   # Active runtime configuration service
│   ├── inference_config_store.py     # Persistent runtime config store
│   ├── approval_checkpoint.py        # Durable LangGraph checkpointer
│   └── project_paths.py              # Canonical project paths
│
├── mcp_server/
│   ├── server.py                     # MCP capability interface
│   └── schemas.py                    # MCP result contracts
│
├── src/
│   ├── inference.py                  # SVHN CNN predictor
│   ├── preprocessing.py              # Image preprocessing
│   ├── orchestration/
│   │   ├── action_policy.py          # Action risk classification
│   │   ├── approval.py               # Approval domain contracts
│   │   ├── controlled_executor.py    # Authorized controlled execution
│   │   ├── inference_service.py      # Runtime inference service
│   │   └── ...
│   └── schemas/                      # Typed application/API contracts
│
├── tools/
│   ├── approval/                     # HITL verification utilities
│   ├── evaluation/                   # Evaluation persistence & services
│   ├── reliability/                  # Confidence, calibration & failure analysis
│   ├── figures/                      # Automated figure generation
│   └── model/                        # Model inspection
│
├── docs/
│   └── mcp-verification/
│       ├── claude/                   # Claude Code interoperability evidence
│       └── vscode/                   # VS Code/Copilot interoperability evidence
│
├── tests/                            # Automated regression suite
├── reports/                          # Typst report source/output
├── figures/                          # Generated report figures
├── artifacts/                        # Evaluation/model artifacts
├── models/                           # Saved CNN
├── sample_images/                    # Integration-test/sample inputs
├── notebooks/                        # Original modeling work
├── generated/                        # Generated runtime/report artifacts
│
├── Dockerfile
├── requirements.txt
├── CHANGELOG.md
└── README.md
```
Persistent generated databases include:

```text
generated/ml_framework.db
generated/runtime_state.db
generated/approval_checkpoints.db
```

Runtime-state databases are generated artifacts and are not committed as
source-controlled application state.

---

## Engineering Design Principles

Several design rules guide the evolution of the project:

- **Python owns facts; Typst owns presentation.**
- **Callers ask for meaning, never position.**
- Shared resources have one canonical definition.
- Evaluation metrics retain the protocol that produced them.
- Domain objects protect their own invariants.
- Configuration errors fail fast at construction time.
- Invalid caller input is distinct from model/inference failure.
- Prediction results, inference attempts, and orchestration decisions are
  separate concepts.
- API contracts remain separate from internal domain and persistence models.
- Orchestration remains deterministic, typed, testable, and independent of
  agent frameworks.
- Reliability observations store source facts; derived reliability meaning is
  computed from them.
- Model inference produces reliability observations once; downstream analyses
  can consume them many times.
- Reliability evaluation observes the orchestration contract rather than
  changing that contract for test convenience.

---

## Technology Stack

**Machine Learning**

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- h5py

**Serving & Interoperability**

- FastAPI
- Uvicorn
- Pydantic
- Model Context Protocol (MCP)
- stdio MCP transport
- Pillow

**Agent & Human-in-the-Loop**

- LangGraph
- interrupt / resume workflows
- durable LangGraph SQLite checkpoints
- application-owned action authorization

**ML Engineering**

- Python dataclasses
- SQLite
- pytest
- Docker
- persistent runtime configuration
- reliability and fallback evaluation

**Reporting**

- Typst
- Matplotlib
- automated Python report-data generation

---

## Roadmap

### Completed — Model & Reliability Engineering

- ANN/CNN model development and comparison
- reproducible model evaluation
- evaluation protocol/provenance modeling
- SQLite evaluation persistence
- automated technical reporting
- evaluation intelligence service
- typed FastAPI contracts
- inference orchestration foundation
- production-facing orchestration service
- typed runtime configuration and serving metadata
- confidence-versus-correctness analysis
- confidence-threshold and selective-risk evaluation
- calibration analysis
- reliability-observation persistence
- deterministic failure and fallback reliability evaluation

### Completed — MCP & Agent Interoperability

- MCP server exposing inference, model, configuration, and evaluation capabilities
- typed MCP-facing operations
- MCP integration testing
- independent external-consumer verification with Claude Code
- independent external-consumer verification with VS Code/Copilot
- dynamic MCP tool discovery over `stdio`
- parameterized MCP tool invocation
- end-to-end image inference with evaluation-context retrieval
- multi-tool external-agent reasoning

### Completed — Controlled Human-in-the-Loop Actions

- application-owned action risk classification
- `READ_ONLY`, `CONTROLLED`, and `PROHIBITED` action policy
- immutable approval request and decision contracts
- exact action/argument approval matching
- centralized controlled-action executor
- controlled confidence-threshold mutation
- LangGraph interrupt/resume approval workflow
- MCP-controlled action integration
- approval and rejection verification
- persistent inference runtime configuration
- persistent LangGraph approval checkpoints
- recovery of runtime configuration after Python-process restart
- recovery and resume of pending approval workflows after process restart

### Next — Trusted Human Approval Enforcement

- strengthen the trust boundary around approval resolution
- prevent autonomous MCP consumers from self-authorizing controlled actions
- distinguish trusted human approval events from agent-generated approval claims
- preserve application-owned authorization independent of client UI behavior

### Future

- durable approval/audit event history
- inference decision/audit records
- latency and fallback metrics
- orchestration observability
- additional controlled runtime actions
- stronger multi-process/runtime coordination
- production-grade authentication and authorization boundaries

Future capabilities are intentionally listed as roadmap items rather than
represented as currently implemented functionality.

---

## Project Evolution

This repository demonstrates the progression from:

```text
Model Training
    ↓
Model Evaluation
    ↓
Reproducible Evaluation Protocols
    ↓
Evaluation Persistence
    ↓
Automated Reporting
    ↓
Typed Service Layer
    ↓
FastAPI Serving
    ↓
Inference Orchestration
    ↓
Reliability Engineering
    ↓
MCP Interoperability
    ↓
Independent Agent Consumption
    ↓
Application-Owned Action Policy
    ↓
Human-in-the-Loop Control
    ↓
Durable Runtime Configuration
    ↓
Persistent Workflow Checkpoints
    ↓
Process-Restart Recovery
```
The project therefore extends beyond model training into serving,
interoperability, reliability, controlled automation, and durable
human-in-the-loop ML-system operation.

---

## Author

**Jerome Jabson**

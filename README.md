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
- Independent MCP interoperability verified with **Claude Code**,
  **VS Code/Copilot**, and **n8n**.
- MCP consumers verified over both **stdio** and **Streamable HTTP** transports.
- n8n verified as both a direct no-code MCP consumer and an AI-agent-driven
  MCP consumer.
- Policy-driven inference orchestration with `ACCEPTED`, `UNCERTAIN`,
  and `FAILED` outcomes.
- Human-in-the-loop approval workflow for controlled runtime changes using
  LangGraph interrupt/resume semantics.
- Typed Python domain models using dataclasses and Pydantic API contracts.
- SQLite-backed persisted evaluation and per-prediction reliability observations.
- Confidence, threshold, selective-risk, and calibration analysis.
- Deterministic failure and fallback reliability evaluation.
- Automated Typst technical-report generation.
- Model architecture inspection and automated evaluation intelligence.
- Dockerized API runtime.
- **273 automated tests** covering preprocessing, inference, orchestration,
  reliability, MCP interoperability, controlled actions, HITL approval,
  persistence, restart recovery, API behavior, and real-model integration.
- Trusted human-in-the-loop authorization for controlled runtime changes,
  separated from untrusted MCP/agent requests.
- Server-owned, single-use approval artifacts persisted in SQLite and bound
  to the exact controlled action and arguments being authorized.
- LangGraph provides durable interrupt/resume workflow mechanics while
  application services remain authoritative for approval verification and
  controlled execution.
- Durable runtime configuration persistence and cross-process synchronization
  using SQLite.
- Durable LangGraph approval checkpoints and trusted approval artifacts survive
  MCP process restarts and workflow reconstruction.
- Independent security verification confirmed that external MCP consumers can
  request and inspect controlled actions but cannot approve, reject, resume,
  or self-authorize them.

---

## System Architecture

```text
                              External Consumers
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
                FastAPI Clients               MCP Consumers
                     │                             │
                     ▼                             ▼
                FastAPI API                  SVHN MCP Server
                                                   │
                          ┌────────────────────────┼───────────────────────┐
                          │                        │                       │
                          ▼                        ▼                       ▼
                    Read-only Tools       Controlled Requests       Status Inspection
                                                   │
                                                   ▼
                                            ActionPolicy
                                                   │
                                                   ▼
                                           ApprovalRequest
                                                   │
                                                   ▼
                                          LangGraph interrupt
                                                   │
                                                   │
                               MCP cannot authorize this boundary
                                                   │
                                                   ▼
                                      Trusted Local Human Channel
                                                   │
                                                   ▼
                                          ApprovalArtifact
                                                   │
                                                   ▼
                                   approval_artifacts.db
                                                   │
                                                   ▼
                                      TrustedApprovalVerifier
                                                   │
                                                   ▼
                                    ControlledActionExecutor
                                      execute_trusted()
                                                   │
                                                   ▼
                                    single-use artifact consume
                                                   │
                                                   ▼
                                      InferenceConfigService
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         │                                                  │
                         ▼                                                  ▼
                runtime_state.db                                  Active Runtime Config
                         │                                                  │
                         └─────────────────────────┬────────────────────────┘
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
- action risk classification;
- controlled-action requests;
- trusted human authorization;
- approval evidence persistence;
- workflow pause/resume state;
- controlled execution;
- durable operational state.

FastAPI and MCP are external adapters over application services rather than
owners of inference policy or authorization.

MCP consumers are intentionally treated as untrusted authorization sources.
They may request controlled actions and inspect their status, but they cannot
create trusted approval evidence or directly authorize execution.

LangGraph owns durable pause/resume workflow mechanics. Trusted approval is
established separately through application-owned approval artifacts and
verified immediately before controlled execution.

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

Runtime mutation is governed by deterministic, application-owned authorization
rather than by an LLM, MCP client, agent framework, or LangGraph resume payload.

Actions are classified into three risk levels:

| Risk | Meaning |
|---|---|
| `READ_ONLY` | May execute without human approval |
| `CONTROLLED` | Requires trusted human approval before execution |
| `PROHIBITED` | Must not execute |

The current controlled action is:

```text
set_confidence_threshold
```

### Trust boundary

External MCP consumers are treated as untrusted authorization sources.

They may:

- request a controlled action;
- receive the generated `request_id` and `thread_id`;
- inspect the pending workflow;
- inspect the final workflow status.

They may not:

- approve or reject the request;
- create trusted approval evidence;
- mark themselves as trusted;
- resume a workflow with authorization authority;
- execute the controlled action directly.

The former MCP approval resolver was removed. In particular, an MCP-supplied
value such as:

```text
approved = true
```

does not constitute trusted authorization.

### Trusted approval flow

The controlled-action workflow is:

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
approval_required
        │
        ├──────────────► MCP may inspect status
        │
        │
        ▼
Trusted local human channel
        │
        ▼
ApprovalArtifact
        │
        ▼
SQLite approval-artifact store
        │
        ▼
LangGraph resume
        │
        ▼
TrustedApprovalVerifier
        │
        ▼
ControlledActionExecutor.execute_trusted()
        │
        ▼
single-use artifact consumption
        │
        ▼
InferenceConfigService
        │
        ▼
persistent runtime update
```

The trusted local human channel is currently provided by:

```bash
python -m tools.approval.run_threshold_approval <THREAD_ID>
```

The CLI retrieves the original persisted approval request, presents the exact
requested action and arguments to the human operator, records the resulting
trusted decision as an `ApprovalArtifact`, and resumes the existing durable
workflow.

### Approval artifacts

Trusted human approval evidence is represented independently from the
LangGraph workflow state.

Each approval artifact is bound to:

- a unique artifact ID;
- the original approval request ID;
- the exact controlled action;
- canonicalized action arguments;
- the approval or rejection decision;
- a trusted issuer;
- an issuance timestamp;
- an expiration timestamp;
- a unique nonce;
- an optional consumption timestamp.

Artifacts are persisted in:

```text
generated/approval_artifacts.db
```

Approval artifacts are immutable and insert-only. Only one artifact may exist
for a request.

Approved artifacts are single-use. The executor atomically marks an artifact
as consumed before executing the controlled mutation, preventing replay and
providing at-most-once authorization semantics.

### Durable workflow state

LangGraph approval workflow state is persisted separately through its SQLite
checkpointer:

```text
generated/approval_checkpoints.db
```

This allows a pending controlled-action request to survive MCP process
termination and later resume using the same `thread_id`.

Runtime configuration is persisted independently in:

```text
generated/runtime_state.db
```

This separation keeps three distinct responsibilities explicit:

```text
approval_checkpoints.db
    → workflow progress

approval_artifacts.db
    → trusted human authorization evidence

runtime_state.db
    → active serving configuration
```

### Cross-process runtime synchronization

Runtime configuration may be changed by the trusted approval CLI while the MCP
server is already running.

`InferenceConfigService` refreshes persisted configuration before returning the
active state, allowing long-running processes to observe changes made by other
processes without requiring an MCP restart.

The verified behavior is:

```text
trusted CLI process
        │
        ▼
writes new runtime configuration
        │
        ▼
runtime_state.db
        │
        ▼
already-running MCP process
        │
        ▼
refreshes persisted configuration
        │
        ▼
observes new threshold
```

### Security invariant

Controlled execution requires all of the following:

```text
valid pending ApprovalRequest
        +
exact action match
        +
exact argument match
        +
trusted ApprovalArtifact
        +
trusted issuer
        +
artifact not expired
        +
artifact not previously consumed
```

Only after those conditions are satisfied may the controlled mutation execute.

This establishes the core invariant:

```text
MCP request ≠ authorization

MCP approved=true ≠ authorization

Trusted human approval evidence
is required for controlled execution.
```

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

## Development Environment

The project uses a repository-local Python virtual environment:

```text
svhn-digit-recognition-ann-vs-cnn/.venv
```

Create and install the environment with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The editable installation includes the project's application packages:

```text
src
config
mcp_server
agent
tools
api
```

PyCharm is configured to use:

```text
<project-root>/.venv/bin/python
```

as the project interpreter so PyCharm execution, terminal sessions, pytest,
FastAPI, MCP, LangGraph, and command-line tooling use the same Python
environment.

---

## Automated Testing

Run the complete suite with:

```bash
python -m pytest -v
```

Current milestone:

```text
273 passed
```

Both supported pytest invocation styles are verified:

```bash
python -m pytest
pytest
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
- trusted approval artifact persistence and verification;
- single-use approval consumption and replay prevention;
- MCP self-authorization prevention;
- cross-process runtime configuration synchronization;
- n8n direct no-code MCP interoperability;
- n8n AI-agent MCP interoperability;
- Streamable HTTP MCP transport.

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
- **n8n**

Claude Code and VS Code/Copilot connect through the local `stdio` MCP
transport.

n8n connects through the MCP server's **Streamable HTTP** transport:

```text
http://host.docker.internal:8001/mcp
```

when n8n is running inside Docker.

These consumers use the same MCP capabilities without depending on the
project's internal LangGraph conversational-agent implementation.

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
| `get_confidence_threshold_approval_status(thread_id)` | Inspect a controlled-action workflow without authorization authority |
| `get_runtime_debug()` | Inspect runtime identity/state for diagnostics |

The MCP interface intentionally exposes no tool capable of approving,
rejecting, resuming with authorization authority, or otherwise self-authorizing
a controlled action.

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

### n8n no-code and AI-agent interoperability

n8n was independently verified as both a direct MCP consumer and an
AI-agent-driven MCP consumer.

The direct no-code workflow verified:

```text
Manual Trigger
    ↓
MCP Client
    ↓
get_inference_config()
    ↓
structured runtime configuration
```

and:

```text
Manual Trigger
    ↓
MCP Client
    ↓
get_digit_metrics(digit=7)
    ↓
structured historical evaluation metrics
```

The AI-agent workflow verified:

```text
Natural-language request
    ↓
n8n AI Agent
    ↓
LLM
    ↓
MCP Client Tool
    ↓
SVHN MCP Server
    ↓
tool selection and invocation
    ↓
structured result
    ↓
natural-language explanation
```

For digit `7`, the agent independently retrieved:

```text
precision = 97.70%
recall    = 95.46%
F1 score  = 96.56%
support   = 2400
```

No SVHN-specific business logic was implemented inside n8n. The workflow
consumes the same reusable MCP interface used by the other external clients.

### n8n trusted-approval boundary verification

The n8n AI Agent was also given access to the controlled
`set_confidence_threshold` operation.

The verified sequence was:

```text
active threshold = 0.95
        ↓
n8n requests 0.93
        ↓
approval_required
        ↓
n8n cannot self-approve
        ↓
active threshold remains 0.95
        ↓
trusted local human CLI approval
        ↓
controlled action executes
        ↓
active threshold = 0.93
        ↓
n8n observes executed state
```

The request remained bound to the same durable identifiers throughout the
workflow:

```text
thread_id
request_id
```

This independently verified that an AI-agent consumer may request and inspect
a controlled action but cannot authorize it.

A separate observation from this verification was that the LLM-generated
natural-language summary incorrectly described one already-executed action as
still awaiting external completion, even though the authoritative MCP response
reported `status = executed` and the runtime configuration independently
confirmed the new threshold.

This reinforces the architectural rule that structured application/tool state
is authoritative and LLM-generated interpretation is not.


### Trusted controlled-action interoperability

VS Code/Copilot independently verified the trusted controlled-action workflow.

The external MCP consumer requested a confidence-threshold change:

```text
0.90 → 0.95
```

The MCP server returned:

```text
approval_required
```

including the generated `thread_id`, `request_id`, action name, exact
arguments, and current threshold.

Before human approval:

- `get_confidence_threshold_approval_status` reported
  `approval_required`;
- `get_inference_config` continued to report `0.90`;
- MCP tool discovery exposed no approve, reject, resume, or authorization tool.

The trusted human then approved the request through the local application-owned
CLI:

```bash
python -m tools.approval.run_threshold_approval <THREAD_ID>
```

The trusted channel recorded the approval artifact and resumed the durable
workflow.

The resulting execution reported:

```text
Trusted decision recorded: approved
Action status: executed
Active confidence threshold: 0.95
```

Without restarting the MCP server, the same external MCP consumer then queried
the workflow and active configuration again.

The MCP server reported:

```text
approval status: executed
confidence threshold: 0.95
```

This verifies the complete separation between external agent capability and
trusted authorization:

```text
External MCP Consumer
        │
        ├── request
        ├── inspect
        │
        └── cannot authorize
                 │
                 ▼
        Trusted Human Channel
                 │
                 ▼
        ApprovalArtifact
                 │
                 ▼
       Trusted verification
                 │
                 ▼
        Controlled execution
```

A separate rejection scenario was also verified. A requested change to `0.92`
was rejected through the trusted CLI, the workflow returned `rejected`, and no
unauthorized configuration mutation occurred.

### Process-restart recovery

Durable workflow behavior has been verified across process reconstruction.

A controlled threshold request may be interrupted while awaiting human
authorization and later recovered from SQLite using the original
`thread_id`.

The restart-recovery path is:

```text
request threshold change
        │
        ▼
approval_required
        │
        ▼
process terminates before approval
        │
        ▼
new application runtime
        │
        ├── restores runtime configuration
        ├── restores LangGraph checkpoint
        └── restores original ApprovalRequest
                 │
                 ▼
        trusted human decision
                 │
                 ▼
        persisted ApprovalArtifact
                 │
                 ▼
        workflow resume
                 │
                 ▼
        trusted verification
                 │
                 ▼
        controlled execution
```

Automated tests verify that:

- the pending LangGraph checkpoint survives graph reconstruction;
- the original approval request ID is preserved;
- a trusted approval artifact may be created after reconstruction;
- the recovered workflow resumes successfully;
- the artifact is consumed after authorized execution;
- the resulting runtime configuration is persisted.

Runtime configuration persistence was also independently verified across MCP
process restarts.

### Architectural result

```text
External MCP Consumer
        │
        │ MCP / stdio
        ▼
SVHN MCP Server
        │
        ├── inference capabilities
        ├── evaluation capabilities
        ├── configuration inspection
        ├── controlled-action request
        └── approval-status inspection
                 │
                 ▼
           Application Policy
                 │
                 ▼
          ApprovalRequest
                 │
                 ▼
          LangGraph HITL
          interrupt / resume
                 │
                 │
       authorization boundary
                 │
                 ▼
       Trusted Human Channel
                 │
                 ▼
         ApprovalArtifact
                 │
                 ▼
    TrustedApprovalVerifier
                 │
                 ▼
 ControlledActionExecutor
                 │
                 ▼
      Runtime Configuration
```

External agents own conversational reasoning, tool selection, and presentation.

The application remains authoritative for:

- model inference;
- configuration validation;
- confidence policy;
- fallback behavior;
- evaluation facts;
- risk classification;
- approval-request identity;
- trusted approval verification;
- replay prevention;
- controlled action execution;
- persistent runtime state.

This separation allows independent MCP consumers to use the same ML system
without giving those consumers authorization authority over controlled runtime
changes.

### Verification status

| Capability | Claude Code | VS Code / Copilot | n8n |
|---|---:|---:|---:|
| MCP connection | Passed | Passed | Passed |
| `stdio` transport | Passed | Passed | — |
| Streamable HTTP transport | — | — | Passed |
| Dynamic tool discovery | Passed | Passed | Passed |
| Zero-argument invocation | Passed | Passed | Passed |
| Parameterized invocation | Passed | Passed | Passed |
| Image inference | Passed | Passed | — |
| Evaluation-context retrieval | Passed | Passed | Passed |
| Multi-tool synthesis | Passed | Passed | Passed |
| AI-agent tool selection | — | Passed | Passed |
| Controlled-action request | — | Passed | Passed |
| Pending approval inspection | — | Passed | Passed |
| MCP self-authorization blocked | — | Passed | Passed |
| Trusted human CLI approval | — | Passed | Passed |
| Trusted human rejection | — | Passed | — |
| Trusted artifact verification | — | Passed | Passed |
| Controlled execution after trusted approval | — | Passed | Passed |
| Process-restart workflow recovery | — | Passed | — |
| Cross-process runtime synchronization | — | Passed | Passed |
| Dependency on internal conversational agent | None | None | None |

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
├── agent/
│   ├── approval_graph.py             # Durable trusted-approval workflow
│   ├── graph.py                      # Agent workflow construction
│   ├── nodes.py                      # Agent nodes
│   └── state.py                      # Agent state contracts
│
├── config/
│   ├── approval_artifact_store.py    # Persistent trusted approval artifacts
│   ├── approval_checkpoint.py        # Durable LangGraph checkpointer
│   ├── inference_config.py           # Typed inference configuration
│   ├── inference_config_service.py   # Cross-process runtime configuration service
│   ├── inference_config_store.py     # Persistent runtime configuration
│   └── project_paths.py              # Canonical project paths
│
├── src/
│   └── orchestration/
│       ├── action_policy.py          # Action risk classification
│       ├── approval.py               # Approval request/decision contracts
│       ├── approval_artifact.py      # Trusted approval evidence domain
│       ├── controlled_executor.py    # Trusted controlled-action execution
│       ├── trusted_approval.py       # Trusted artifact verification
│       ├── trusted_approval_service.py # Trusted human decision recording
│       └── ...
│
├── tools/
│   └── approval/
│       └── run_threshold_approval.py # Trusted local human approval channel
│
├── generated/
│   ├── runtime_state.db              # Active runtime configuration
│   ├── approval_checkpoints.db       # Durable LangGraph workflow state
│   └── approval_artifacts.db         # Trusted human approval evidence
│
├── docs/
│   └── mcp-verification/
│       ├── claude/
│       ├── vscode/
│       └── n8n/
│           ├── README.md
│           └── n8n_full_workflow_verification.txt
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
- MCP `stdio` transport
- MCP Streamable HTTP transport
- n8n
- Pillow


**Agent & Human-in-the-Loop**

- LangGraph
- interrupt / resume workflows
- durable LangGraph SQLite checkpoints
- trusted server-owned approval artifacts
- application-owned action authorization
- n8n AI Agent / MCP Client Tool interoperability

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
- independent no-code consumer verification with n8n
- n8n AI Agent + MCP Client Tool verification
- dynamic MCP tool discovery over `stdio`
- Streamable HTTP MCP transport
- parameterized MCP tool invocation
- end-to-end image inference with evaluation-context retrieval
- multi-tool external-agent reasoning

### Completed — Trusted Human-in-the-Loop Actions

- application-owned action risk classification
- `READ_ONLY`, `CONTROLLED`, and `PROHIBITED` action policy
- immutable approval request contracts
- exact action/argument approval matching
- centralized controlled-action executor
- controlled confidence-threshold mutation
- LangGraph interrupt/resume workflow
- persistent inference runtime configuration
- persistent LangGraph approval checkpoints
- server-owned trusted approval artifacts
- exact request/action/argument binding
- approval expiration handling
- single-use authorization artifacts
- replay prevention
- MCP self-authorization prevention
- trusted local human approval channel
- approval and rejection verification
- process-restart workflow recovery
- cross-process runtime configuration synchronization
- n8n trusted-approval boundary verification

### Completed — Development Environment Normalization

- repository-local `.venv`
- pinned direct runtime dependencies
- pinned development dependencies
- explicit SQLite LangGraph checkpoint dependency
- editable application package installation
- `api` and `tools` included in setuptools package discovery
- PyCharm project interpreter aligned with repository `.venv`
- both `pytest` and `python -m pytest` verified
- **273 passing automated tests**

### Next — Production Hardening

- harden LangGraph checkpoint deserialization
- strengthen production authentication and authorization boundaries
- expand audit/event history
- add inference decision and operational audit records
- add latency, fallback, and orchestration observability


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

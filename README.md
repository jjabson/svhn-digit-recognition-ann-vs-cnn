# SVHN Digit Recognition — ML Inference & Evaluation Platform

An end-to-end machine learning engineering project built around Street View
House Numbers (SVHN) digit recognition.

The project began as an ANN-vs-CNN computer vision study and evolved into a
production-oriented ML system with reproducible evaluation, persisted metrics,
automated technical reporting, typed inference contracts, FastAPI serving,
Docker containerization, and policy-driven inference orchestration.

The goal is no longer only to train an accurate classifier. The repository
demonstrates how a trained ML model can be evaluated, packaged, served,
observed, and integrated into a maintainable software architecture.

---

## Highlights

- **95.14% accuracy / ~95.14% macro F1** on an independent stratified
  holdout containing 24,000 samples.
- Improved from **67.36% ANN baseline → 74.84% improved ANN → 95.14% CNN**.
- FastAPI/Uvicorn inference and evaluation API.
- Policy-driven inference orchestration with `ACCEPTED`, `UNCERTAIN`,
  and `FAILED` outcomes.
- Typed Python domain models using dataclasses and Pydantic API contracts.
- SQLite-backed persisted evaluation and per-prediction reliability observations.
- Confidence, threshold, selective-risk, and calibration analysis.
- Deterministic failure and fallback reliability evaluation.
- Automated Typst technical-report generation.
- Model architecture inspection and automated evaluation intelligence.
- Dockerized API runtime.
- **84 automated tests** covering preprocessing, inference, orchestration,
  reliability analysis, persistence, configuration, API behavior, and
  real-model integration.

---

## System Architecture

```text
                         Clients
                            │
                            ▼
                     ┌─────────────┐
                     │   FastAPI   │
                     │   /docs     │
                     └──────┬──────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     Inference Orchestration       Evaluation API
              │                           │
      ┌───────┴────────┐                  ▼
      │                │          Evaluation Service
      ▼                ▼                  │
Inference Policy   Runtime Config         ▼
      │                            SQLite Evaluation
      ▼                                 Store
 SVHN Predictor                           │
      │                                   │
      ▼                                   ▼
Image Preprocessing              Metrics / Insights
      │                                   │
      ▼                                   ▼
TensorFlow CNN                    Report Generation
                                          │
                                          ▼
                                      Typst / PDF
```

The architecture deliberately separates model execution, orchestration
decisions, persistence, API contracts, evaluation logic, and presentation.

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

The current confidence threshold is **0.90**. Reliability analysis on the
authoritative 24,000-sample Historical Stratified Holdout shows that this
operating point provides **91.38% coverage**, **98.44% accepted accuracy**,
**1.56% selective risk**, and an **8.62% uncertainty rate**.

The 0.90 threshold remains an evaluated operating point rather than a claimed
globally optimal production threshold. Final threshold selection depends on
the reliability, risk, and review-cost requirements of the serving environment.

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

Inference behavior is represented by a typed, immutable configuration
object.

Current defaults:

```text
model_name            = cnn
confidence_threshold  = 0.90
```

Configuration is validated when constructed so invalid operating values
fail early rather than leaking into runtime inference behavior.

The service factory also accepts injected configuration, allowing different
runtime policies without modifying orchestration code or global state.

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

At the current **0.90** orchestration threshold:

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

---

## Automated Testing

Run the complete suite with:

```bash
python -m pytest -v
```

Current milestone:

```text
84 passed, 1 warning
```

Coverage includes:

- image preprocessing
- typed prediction results
- generic model execution
- inference adapters
- confidence policy behavior
- threshold boundaries
- primary/fallback orchestration behavior
- invalid-input propagation
- runtime configuration validation
- service-factory configuration injection
- FastAPI response contracts
- HTTP failure semantics
- serving metadata
- real saved-CNN integration
- confidence-versus-correctness analysis
- confidence-threshold and selective-risk evaluation
- calibration analysis
- SQLite reliability-observation persistence
- deterministic failure injection
- fallback recovery and review escalation

The remaining warning is a known FastAPI/Starlette `TestClient` dependency
deprecation warning and does not affect current test correctness.

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
├── api/
│   └── app.py                 # FastAPI application
│
├── config/
│   ├── inference_config.py    # Runtime inference behavior
│   └── project_paths.py       # Canonical project paths
│
├── src/
│   ├── inference.py           # SVHN CNN predictor
│   ├── preprocessing.py       # Image preprocessing
│   ├── orchestration/         # Execution, adapters, policy, services
│   └── schemas/               # Typed domain/API contracts
│
├── tools/
│   ├── evaluation/            # Evaluation persistence & services
│   ├── reliability/           # Confidence, calibration & failure analysis
│   ├── figures/               # Automated figure generation
│   └── model/                 # Model inspection
│
├── tests/                     # Automated test suite
├── reports/                   # Typst report source/output
├── figures/                   # Generated report figures
├── artifacts/                 # Evaluation/model artifacts
├── models/                    # Saved CNN
├── sample_images/             # Integration-test/sample inputs
├── notebooks/                 # Original modeling work
├── generated/                 # Generated runtime/report artifacts
│
├── Dockerfile
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

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

**Serving & Application**

- FastAPI
- Uvicorn
- Pydantic
- Pillow

**ML Engineering**

- Python dataclasses
- SQLite
- pytest
- Docker

**Reporting**

- Typst
- Matplotlib
- automated Python report-data generation

---

## Roadmap

### Completed

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

### Next — MCP Interoperability

The next phase will expose selected inference and evaluation capabilities
through an MCP interface while keeping orchestration and reliability logic
independent of agent frameworks.

Planned work includes:

- MCP interface for tool interoperability
- typed MCP-facing operations
- MCP integration testing
- test agent consuming the MCP interface

### Future

After MCP interoperability:

- demonstration agent
- inference decision/audit records
- latency and fallback metrics
- orchestration observability

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
Persistence
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
```

The emphasis is on treating machine learning as a software system rather
than only as a trained model.

---

## Author

**Jerome Jabson**

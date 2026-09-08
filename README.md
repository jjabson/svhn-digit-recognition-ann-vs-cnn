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
- SQLite-backed persisted evaluation results.
- Automated Typst technical-report generation.
- Model architecture inspection and automated evaluation intelligence.
- Dockerized API runtime.
- **39 automated tests** covering preprocessing, inference, orchestration,
  configuration, API behavior, and real-model integration.

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

The current confidence threshold is **0.90**. It is deliberately treated
as a provisional orchestration setting rather than a calibrated production
threshold.

Threshold calibration and coverage-versus-risk analysis are planned as
part of the reliability evaluation phase.

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

## Automated Testing

Run the complete suite with:

```bash
python -m pytest -v
```

Current milestone:

```text
39 passed
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

### Next — Reliability Evaluation

Planned work includes:

- confidence-threshold analysis
- confidence vs. correctness measurement
- coverage-versus-risk analysis
- uncertainty behavior
- failure injection
- fallback-effectiveness evaluation

### Future

After the orchestration system has been reliability-tested:

- MCP interface for agent/tool interoperability
- MCP test agent
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

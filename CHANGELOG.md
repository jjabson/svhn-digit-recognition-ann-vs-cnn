# Changelog

## v0.3.0

### Added

- Automated Typst report generation
- Dataset metadata generation
- Automated EDA figures
- Automated preprocessing figures
- CNN model inspection
- FastAPI inference service
- Docker support
- Report build pipeline
- Unit tests

## Added

- CNN Architecture chapter
- High-level CNN architecture visualization
- Detailed layer-by-layer CNN visualization
- Model inspection framework
- Automatic architecture grouping
- Integrated model figures into report generation

## Phase 3.1 – Intelligent Domain Objects

### Added
- Introduced semantic ArchitectureInfo properties:
  - first_conv
  - flatten_layer
  - hidden_dense_layer
  - dropout_layer
  - output_layer
  - final_feature_group

- Added computed properties to GroupInfo:
  - number_of_layers
  - parameters
  - output_shape

- Added ModelSummary dataclass.

### Changed
- Replaced dictionary-based architecture inspection with typed domain objects.
- Eliminated GroupSummary by moving derived values into GroupInfo.
- Removed hardcoded layer indexes from report generation.
- Report variables are now generated from the inspected model.

### Result
- Python is now the single source of truth for architecture metadata.
- Typst consumes generated model variables rather than duplicated values.

### Added
- Added a reusable evaluation domain model with `EvaluationProtocol`,
  `ClassMetrics`, and `EvaluationInfo`.
- Added model-agnostic prediction evaluation for accuracy, precision,
  recall, F1 score, support, and confusion matrices.
- Added correct and incorrect prediction metrics.
- Added historical SVHN evaluation data reconstruction using the original
  stratified 80/20 notebook protocol.
- Added batch image preprocessing for model evaluation.
- Added shared project path configuration in `config/project_paths.py`.

### Changed
- Centralized dataset, model, figure, generated-data, and report paths.
- Updated report, figure, model inspection, and evaluation tooling to use
  shared project paths.
- Reproduced the historical CNN evaluation directly from the saved model
  and evaluation data instead of relying on previously generated CSV
  artifacts.

### Validation
- Historical evaluation reproduced 95.14% test accuracy and approximately
  95.14% macro F1 across 24,000 evaluation samples.
- Historical evaluation produced 22,833 correct predictions and 1,167
  incorrect predictions.
- Full Typst report build completed successfully after the shared-path
  refactor.

## Phase 4.3 – Evaluation Protocol Comparison

### Added
- Added `EvaluationComparison` for semantic access to multiple evaluation results.
- Added support for the original HDF5 test split as a second evaluation protocol.
- Added provenance metadata to `EvaluationProtocol`.
- Added semantic properties for evaluation independence and accuracy percentage.
- Added comparison support for historical holdout and original test diagnostics.

### Changed
- Refactored `evaluate_trained_model()` to accept a reusable evaluation data loader.
- Replaced hardcoded historical evaluation loading with protocol-driven evaluation.
- Preserved evaluation meaning alongside computed metrics.

### Validation
- Historical Stratified Holdout:
  - Accuracy: 95.14%
  - Macro F1: 95.14%
  - Support: 24,000
  - Correct: 22,833
  - Incorrect: 1,167

- Original HDF5 Test Diagnostic:
  - Accuracy: 96.84%
  - Macro F1: 96.84%
  - Support: 18,000
  - Correct: 17,432
  - Incorrect: 568
  - Not independent of historical training

## Phase 4.4 — Persisted Model Evaluation Reporting

### Added
- Added SQLite persistence for model evaluation results.
- Added evaluation storage and retrieval through `evaluation_store.py`.
- Added serialization and reconstruction of evaluation protocols, aggregate metrics, per-class metrics, and confusion matrices.
- Persisted both the Historical Stratified Holdout and Original HDF5 Test Diagnostic evaluation results.
- Added semantic evaluation variables for strongest class, weakest class, and most common misclassification.
- Added an evaluation protocol comparison section to the generated report.
- Added `scikit-learn` as an explicit project dependency.

### Changed
- Updated report generation to consume persisted evaluation results instead of rerunning CNN inference.
- Updated evaluation figure generation to consume persisted evaluation data.
- Established the Historical Stratified Holdout evaluation as the authoritative independent evaluation for the saved CNN.
- Classified the Original HDF5 Test evaluation as diagnostic because its samples were included in the historical combined training/evaluation dataset.
- Improved confusion-matrix presentation and reporting.
- Reduced redundant model evaluation during report builds.

## Phase 4.5 — API Readiness and Evaluation Intelligence

### Added
- Added an evaluation service layer to separate application logic from SQLite persistence.
- Added semantic evaluation helpers for identifying the best-performing class, weakest-performing class, and most common misclassification.
- Added `EvaluationInsights` as a domain-level container for derived evaluation intelligence.
- Added Pydantic response schemas to define stable public API contracts for evaluation data.
- Added `GET /evaluation/insights` endpoint backed by persisted evaluation results.
- Added Swagger/OpenAPI response examples for evaluation insights.
- Added FastAPI endpoint groups for System, Prediction, and Evaluation.

### Changed
- Improved FastAPI application metadata, endpoint summaries, descriptions, and Swagger organization.
- Updated report-data generation to consume evaluation intelligence through the service layer.
- Kept API response formatting separate from internal evaluation domain models and persistence structures.

---

## Phase 4.5 — API Readiness and Evaluation Intelligence

### Added
- Added an evaluation service layer separating application logic from SQLite persistence.
- Added semantic evaluation helpers for strongest class, weakest class, and most common misclassification.
- Added the `EvaluationInsights` domain container.
- Added Pydantic response schemas to define stable public API contracts.
- Added `GET /evaluation/insights` for model evaluation insights.
- Added `GET /evaluation/summary` for headline metrics from the primary independent evaluation.
- Added `GET /evaluation/classes` for per-class precision, recall, F1 score, and sample counts.
- Added `GET /evaluation/classes/{digit}` for retrieving evaluation metrics for an individual digit class.
- Added a Pydantic response contract for `POST /predict`.
- Added Swagger/OpenAPI response examples for prediction and evaluation resources.
- Added explicit OpenAPI documentation for expected `404` responses when a requested digit class does not exist.
- Added FastAPI endpoint groups for System, Prediction, and Evaluation resources.

### Changed
- Improved FastAPI application metadata, endpoint summaries, descriptions, and Swagger organization.
- Updated report-data generation to consume evaluation intelligence through the service layer.
- Kept API response formatting separate from internal domain and persistence models.
- Renamed public API `support` terminology to `sample_count` for clearer, non-specialist-facing responses.
- Renamed public API `total_support` terminology to `total_sample_count` for consistency.
- Updated prediction responses to use an explicit Pydantic API contract instead of an untyped dictionary response.
- Documented application-level `404` behavior as part of the public OpenAPI contract.

## Phase 4.5 — API Readiness and Evaluation Intelligence

### Added
- Added an evaluation service layer separating application logic from SQLite persistence.
- Added semantic evaluation helpers for strongest class, weakest class, and most common misclassification.
- Added the `EvaluationInsights` domain container.
- Added Pydantic response schemas to define stable public API contracts.
- Added `GET /evaluation/insights` for high-level model evaluation intelligence.
- Added `GET /evaluation/summary` for headline metrics from the primary independent evaluation.
- Added `GET /evaluation/classes` for per-class precision, recall, F1 score, and sample counts.
- Added `GET /evaluation/classes/{digit}` for retrieving evaluation metrics for an individual digit class.
- Added `GET /evaluation/confusion-matrix` for exposing the persisted confusion matrix as structured JSON.
- Added `GET /model/summary` for trained CNN architecture and capacity information.
- Added a model service layer between model inspection logic and the API.
- Added semantic layer properties for dense-layer units and dropout rate.
- Added a Pydantic response contract for `POST /predict`.
- Added Pydantic response contracts for evaluation and model-summary resources.
- Added Swagger/OpenAPI response examples for prediction, evaluation, and model resources.
- Added explicit OpenAPI documentation for expected `404` responses when a requested digit class does not exist.
- Added FastAPI endpoint groups for System, Prediction, Evaluation, and Model resources.

### Changed
- Improved FastAPI application metadata, endpoint summaries, descriptions, and Swagger organization.
- Updated report-data generation to consume evaluation intelligence through the service layer.
- Kept API response formatting separate from internal domain and persistence models.
- Renamed public API `support` terminology to `sample_count` for clearer non-specialist-facing responses.
- Renamed public API `total_support` terminology to `total_sample_count` for consistency.
- Updated prediction responses to use an explicit Pydantic API contract instead of an untyped dictionary response.
- Converted NumPy confusion-matrix data into a JSON-safe labels-and-matrix API representation.
- Reused the already-loaded inference model for model-summary inspection instead of loading the Keras model again per request.
- Exposed numeric model facts such as parameter counts as numeric API values while leaving display formatting to presentation layers.
- Documented application-level `404` behavior as part of the public OpenAPI contract.
- Expanded Swagger documentation with an explicit Model resource category.
- Bumped the API version from `1.1.0` to `1.2.0`.

### Fixed
- Fixed `/evaluation/insights` route registration so the insights endpoint is associated with its correct handler and response contract.
- Corrected inconsistent public API terminology between class-level and evaluation-summary sample counts.
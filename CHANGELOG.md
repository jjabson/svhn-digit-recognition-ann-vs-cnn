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
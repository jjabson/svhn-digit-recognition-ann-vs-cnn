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
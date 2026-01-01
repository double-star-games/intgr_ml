# Changelog

All notable changes to IntgrML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2025-12-31

### Added

- **Python Multiclass Classification**: `IntgrBoostClassifier` now supports multiclass (K > 2 classes) directly
  - Auto-detects number of classes from training labels
  - Uses One-vs-Rest (OvR) internally for K > 2 classes
  - `predict()` returns class labels (0 to K-1)
  - `predict_proba()` returns (n_samples, K) array with softmax-normalized probabilities
  - `classes_` and `n_classes_` attributes available after fitting
  - Models saved as `.ovr` format for multiclass, `.sbf` for binary

- **Sidecar Metadata Persistence**: Models now save quantization parameters to companion `.meta` files
  - Saves `clip_min`, `clip_max`, `n_bins`, and `n_classes`
  - Backward compatible: old models without `.meta` files load with defaults
  - JSON format for human readability and debugging
  - Example: `model.sbf` creates `model.sbf.meta`

### Documentation

- Updated [Python API Reference](docs/PYTHON_API.md) with multiclass usage and sidecar metadata

---

## [1.2.7] - 2025-12-30

### Added

- **Parameter Validation**: All sklearn API classes now validate parameters at `fit()` time with clear error messages
  - Type checking (TypeError for wrong types)
  - Range checking (ValueError for out-of-bounds values)
  - Covers: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_leaf`, `n_bins`, `clip_min`, `clip_max`, `subsample`, `temperature`, `random_state`

- **Temperature Parameter**: New `temperature` parameter for `IntgrBoostClassifier` and `IntgrForestClassifier`
  - Controls `predict_proba()` probability calibration
  - Default: 100.0
  - Higher values produce more uniform probabilities
  - Lower values produce more confident predictions

- **Feature Limit Validation**: Clear error message when exceeding 65,535 features

### Documentation

- Updated [Python API Reference](docs/PYTHON_API.md) with validation rules and temperature parameter

---

## [1.2.6] - 2025-12-30

### Fixed

- Repository moved to `double-star-games/intgr_ml` organization

---

## [1.2.5] - 2025-12-30

### Fixed

- **CLI Prediction Bug**: Fixed issue where CLI-trained models could predict incorrectly on certain datasets
  - Corrected prediction threshold semantics
  - Improved quantization for datasets with clipped values

---

## [1.2.0] - 2025-11-06

### Added

- **One-vs-Rest (OvR) Multiclass Classification**
  - Train K-class classifiers using `intgrmlc boost-ovr train`
  - Parallel training across CPU cores
  - New `.ovr` model format

- **MCU Export**: Export models to compact `.mcu` format for embedded deployment
  - `intgrmlc boost export --model model.sbf --out model.mcu`

- **Determinism Verification**: Verify bit-exact reproducibility
  - `intgrmlc boost determinism --data train.csv --runs 3`

- **Parity Testing**: Verify desktop/MCU inference match
  - `intgrmlc boost parity --model model.sbf --data test.csv`

### Performance

- Covertype 7-class: 78.6% accuracy with OvR
- Parallel training: ~6× speedup on multi-core systems

---

## [1.1.0] - 2025-10-XX

### Fixed

- Improved gradient accumulator precision (Q16.16 format)
- Fixed binary leaf encoding in random forests

### Added

- Q16.16 accumulator as default for IntgrBoost
- Model metadata tracking (version, format info)

### Performance

- IntgrBoost: Up to 68% accuracy on HIGGS dataset
- IntgrForest: Up to 65% accuracy on HIGGS dataset

---

## [1.0.0] - 2025-XX-XX

### Initial Release

**Algorithms:**
- IntgrBoost: Gradient boosted trees
- IntgrForest: Random forests
- IntgrLinear: Linear/logistic regression
- IntgrBayes: Naive Bayes classifier
- IntgrReduce: PCA dimensionality reduction

**Features:**
- Pure integer arithmetic (no floating-point at runtime)
- Bit-exact deterministic training and inference
- Edge/embedded deployment ready
- Python package with scikit-learn compatible API
- CLI tool (`intgrmlc`) for training and inference

**Platforms:**
- Linux x86_64 (Tier 1)
- Windows x64 (Tier 1)
- Linux ARM64 (Tier 1)

---

## Legend

- **Added**: New features
- **Fixed**: Bug fixes
- **Changed**: Changes in existing functionality
- **Performance**: Performance improvements
- **Documentation**: Documentation updates

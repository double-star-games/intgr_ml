# Changelog

All notable changes to IntgrML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.1] - 2026-01-01

### 🐛 Fixed

- **Q8.8 Inference Truncation Fix**: Fixed asymmetric rounding in Q8.8 to Q0 conversion
  - Old: `acc >> 8` (truncates toward negative infinity)
  - New: `(acc + 128) >> 8` (proper rounding)
  - Impact: Slight positive bias in predictions near zero is now balanced
  - Affected: `Boost::predict_one()` and `MCURunner::predict()`

- **Multiclass Bias Calculation Fix**: Fixed integer truncation in OvR bias initialization
  - Old: `(sum_labels / num_samples) * 32` (loses precision)
  - New: `(sum_labels * 32) / num_samples` (scale before division)
  - Impact: More accurate initial bias for multiclass heads

### 🧪 Testing

- 425 C++ tests passing
- 85 Python tests passing
- HIGGS benchmark: 67.51% accuracy with balanced logits (-52 to +52)

---

## [1.3.0] - 2025-12-31

### ✨ Added

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

### 🧪 Testing

- Added comprehensive multiclass and persistence test suite (`python/tests/test_multiclass.py`)

### 📚 Documentation

- Updated PYTHON_API.md with multiclass usage and sidecar metadata
- Updated RUNBOOK.md with v1.3.0 features

---

## [1.2.7] - 2025-12-30

### ✨ Added

- **Parameter Validation**: All sklearn API classes now validate parameters at `fit()` time with clear error messages
  - Type checking (TypeError for wrong types)
  - Range checking (ValueError for out-of-bounds)
  - Covers: n_estimators, max_depth, learning_rate, min_samples_leaf, n_bins, clip_min/max, subsample, temperature, random_state

- **Temperature Parameter**: New `temperature` parameter for `IntgrBoostClassifier` and `IntgrForestClassifier`
  - Controls `predict_proba()` logit-to-probability calibration
  - Default: 100.0 (matches Q8.8 fixed-point scale)
  - Higher values → more uniform probabilities
  - Lower values → more confident predictions

- **Feature Limit Validation**: Clear error when exceeding 65,535 features with actionable message

### 🧪 Testing

- Added 38 new validation tests (68 total Python tests)
- CLI<-->Python consistency tests (Issue #32)

### 📚 Documentation

- Updated PYTHON_API.md with validation rules and temperature parameter
- Updated RUNBOOK.md with DX improvements

---

## [1.2.6] - 2025-12-30

### 🐛 Fixed

- Updated public repo references from `pmeade/intgr_ml` to `double-star-games/intgr_ml`
- Release workflow now publishes to organization repo

---

## [1.2.5] - 2025-12-30

### 🐛 Fixed

- **CLI Quantization Bug**: Fixed CLI-trained models always predicting Class 0
  - Root cause 1: Prediction threshold used `> 0` instead of `>= 0`
  - Root cause 2: Percentile quantization collapsed nearly-constant values
  - Added canonical `Quantizer::quantize_linear()` method
  - Added `CsvLoadStrategy::LINEAR` as default

---

## [1.0.0-dev] - 2025-12-14

### 🔄 Development Version

**Current development version targeting 1.0.0 release.**

### ✨ Changes in 1.0.0-dev

- **Rebranding**: Renamed from SnapML to IntgrML
  - C++ namespace: `snap` → `intgr`
  - Binary format magic: `SNAPMCU1` → `INTGMCU1`
  - All documentation and code updated
- **Version unification**: All version strings now 1.0.0-dev
- **License clarification**: Commercial & Community License v1.1

### ✨ Features (from internal development)

- Complete gradient boosting implementation (IntgrBoost)
- Random forest support (IntgrForest)
- One-vs-Rest multiclass classification (OvR)
- Mixed-precision quantization (int8/int16)
- MCU export format for embedded systems
- Python bindings with scikit-learn compatible API
- CLI tool (intgrmlc) for training and inference
- Q16.16 fixed-point arithmetic for numerical stability
- SIMD optimizations (AVX2/NEON/WASM)
- Comprehensive test coverage (287 C++ tests, 36 Python tests)

---

## Previous Internal Versions (Pre-Release)

## [1.2.1] - 2025-11-06 (Internal)

### ⚠️ BREAKING CHANGES

**Models trained with v1.2.0 MUST be retrained.**

Desktop inference had a critical bug where the learning rate was not applied during prediction. This caused incorrect predictions and parity mismatches with MCU inference. All v1.2.0 models must be retrained with v1.2.1.

### 🐛 Fixed

- **CRITICAL**: Fixed learning rate application bug in `Boost::predict_one` (src/intgrboost/Boost.cpp:39-49)
  - Desktop inference now correctly applies learning rate: `acc += (tree_pred * lr_fp) >> 8`
  - Previously: `acc += tree_pred` (learning rate ignored)
  - Impact: All v1.2.0 predictions were incorrect
  - Migration: Retrain all models with v1.2.1

- Fixed MCU format missing `bias_fp` field
  - Header size: 36 → 40 bytes
  - Added `i32 bias_fp` at offset 32
  - Required for correct MCU inference initialization

- Fixed MCU inference not applying learning rate
  - MCU runner now matches desktop behavior
  - Achieved 100% bit-exact parity (100/100 samples)

### ✨ Added

#### MCU Export & Embedded Deployment

- **MCU binary format** (`.mcu`) for embedded systems
  - Optimized for microcontrollers (STM32, ESP32, ARM Cortex-M)
  - 40-byte header with CRC-32 checksum
  - Sequential read-only parsing (zero dynamic allocation)
  - 10-30% smaller than `.sbf` format
  - Specification: `docs/formats/sbf_mcu.md`

- **New command**: `intgrmlc boost export`
  - Export `.sbf` models to `.mcu` format
  - Usage: `intgrmlc boost export --model model.sbf --out model.mcu`

- **New command**: `intgrmlc boost parity`
  - Verify bit-exact parity between desktop and MCU inference
  - Tests N samples, reports match percentage
  - Usage: `intgrmlc boost parity --model model.sbf --data test.csv --samples 100`
  - Exit code 0: 100% parity, 5: mismatch detected

#### Determinism & Quality Assurance

- **New command**: `intgrmlc boost determinism`
  - Verify bit-exact deterministic training across multiple runs
  - Computes SHA-256 hash of serialized models
  - Usage: `intgrmlc boost determinism --data train.csv --out model.sbf --runs 3`
  - Exit code 0: all hashes match, 5: mismatch detected

- **MCURunner** inference engine (`src/intgrruntime/MCURunner.cpp`)
  - Loads and runs `.mcu` models
  - Zero dynamic allocation
  - Bit-exact parity with desktop

- **Hasher** module (`src/intgrcommon/Hasher.cpp`)
  - SHA-256 cryptographic hashing for model verification
  - Used by determinism command

### 🧪 Changed

- **Updated 264 unit tests** for correct learning rate behavior
  - `test_boost_inference.cpp`: Updated expected values
  - `test_binary_sanity.cpp`: Fixed LR overflow (256 → 255)
  - `test_boosttrainer.cpp`: Increased LR to avoid Q8.8 rounding (64 → 128)
  - `test_q16_precision.cpp`: Updated float baseline, relaxed tolerances
  - All tests passing (264/264, 3411 assertions)

- **Learning rate range**: `0.0-1.0` (was `0.0-infinity`)
  - Stored as `u8` in Q8.8 format
  - Max value: 255 (~0.996)
  - Prevents overflow in fixed-point arithmetic

### 📚 Documentation

- Added `CRITICAL_BUG_FIX.md` - Complete analysis of learning rate bug and migration guide
- Added `docs/formats/sbf_mcu.md` - Comprehensive MCU format specification
- Updated `README.md` - Added v1.2.1 section with parity demo
- Updated `CLI_REFERENCE.md` - Added determinism, export, parity commands
- Updated version to 1.2.1 in all documentation

### 🔧 Technical Details

**Bug Analysis**:
```cpp
// OLD (buggy):
i32 Boost::predict_one(const BoostModel& model, const i8* features) {
    i64 acc = model.bias_fp;
    for (const auto& tree : model.trees) {
        i32 tree_pred = predict_tree(tree, features);
        acc += tree_pred;  // ❌ Learning rate not applied
    }
    return static_cast<i32>(acc);
}

// NEW (correct):
i32 Boost::predict_one(const BoostModel& model, const i8* features) {
    i64 acc = model.bias_fp;
    for (const auto& tree : model.trees) {
        i32 tree_pred = predict_tree(tree, features);
        i64 scaled_pred = static_cast<i64>(tree_pred) * model.learning_rate_fp;
        acc += (scaled_pred >> 8);  // ✅ Apply learning rate (Q8.8 → Q0)
    }
    return static_cast<i32>(acc);
}
```

**Parity Verification**:
- Before fix: Desktop=-46, MCU=-10 (mismatch)
- After fix: Desktop=-46, MCU=-46 (100% parity)

---

## [1.2.0] - 2025-11-06

### ✨ Added

#### One-vs-Rest (OvR) Multiclass Support

- **Native multiclass training** via K binary IntgrBoost heads
  - New module: `boost-ovr` (CLI), `OvR` (C++ API)
  - K independent binary classifiers (one vs. rest)
  - Parallel head training with 5.95× speedup
  - File format: `.ovr` (96-byte header + K embedded `.sbf` models)

- **New commands**: `intgrmlc boost-ovr train|predict`
  - Usage: `intgrmlc boost-ovr train --data iris.csv --out model.ovr --trees 200`
  - Usage: `intgrmlc boost-ovr predict --model model.ovr --data test.csv --out preds.csv`
  - Optional `--logits` flag to output all K class scores

- **Multiclass inference**
  - Prediction: `argmax(K logits)`
  - Logit output: Q16.16 fixed-point scores per class
  - Overhead: 1.035× per head (17k pred/sec for K=7)

### 📈 Performance

**Covertype 7-class Benchmark**:
- Test Accuracy: **78.6%** (baseline: 48.8%, +29.9pp improvement)
- Training Time: 42.5s for 46k samples (7 heads × 150 trees)
- Inference: 17k predictions/sec
- Determinism: ✅ 100% hash match across 3 runs
- Parallel Speedup: 5.95× (7-core CPU)

### 📚 Documentation

- Added `OVR_DESIGN.md` - Complete OvR architecture and usage guide
- Updated `README.md` - Added "What's New in v1.2" section
- Updated `CLI_REFERENCE.md` - Added `boost-ovr` module specification

---

## [1.1.0] - 2025-10-XX

### 🐛 Fixed

**Major Quality & Accuracy Improvements**:

- **IntgrBoost**: Fixed Q16.16 accumulator precision loss
  - Accuracy: 58.6% → **68.0%** (+9.4pp improvement)
  - Issue: Accumulator was truncating fractional bits
  - Impact: HIGGS 100k binary classification

- **IntgrForest**: Fixed binary leaf encoding bug
  - Accuracy: 52.8% → **65.0%** (+12.2pp improvement)
  - Issue: Leaves encoded as 0/32 instead of -32/+32
  - Impact: Binary classification (all datasets)

### ✨ Added

- **Q16.16 accumulator** as default for IntgrBoost
  - 16-bit integer + 16-bit fractional precision
  - No precision loss during gradient accumulation
  - Configurable via `--accum q16_16` flag

- **Model metadata tracking**
  - Version, git commit, accumulator format
  - Embedded in `.sbf` file header
  - Used for compatibility checks

- **Golden baselines** for regression testing
  - 8 validation datasets (HIGGS, Covertype, etc.)
  - Automated accuracy threshold checks
  - Prevents future regressions

### 🧪 Testing

- **260 comprehensive tests** (252 unit + 8 determinism)
  - 100% pass rate
  - Coverage: training, inference, determinism, edge cases
  - Test suites: boost, forest, linear, bayes, reduce

- **Bit-exact determinism verification**
  - 3-run hash tests for all modules
  - SHA-256 model hashing
  - Validates reproducibility

### 📈 Performance

**HIGGS 100k Binary Classification**:
- IntgrBoost: **68.0%** accuracy (200 trees, depth 10)
- IntgrForest: **65.0%** accuracy (100 trees, depth 8)
- Training: ~27s (20× faster than float baseline)
- Inference: ~15ms per 1k samples

**Covertype 7-class Multiclass** (baseline, pre-OvR):
- IntgrBoost: **48.8%** accuracy
- IntgrForest: **51.0%** accuracy

### 📚 Documentation

- Added `FINDINGS_v1.1.md` - Complete technical analysis and competitive positioning
- Added `PHASE1_COMPLETE.md` - Q16.16 accumulator implementation details
- Updated `README.md` - Added "What's New in v1.1" section

---

## [1.0.0] - 2025-XX-XX

### ✨ Initial Release

**Core Modules**:
- **IntgrBoost**: Gradient boosted trees (binary classification, regression)
- **IntgrForest**: Random forests
- **IntgrLinear**: Linear/logistic regression
- **IntgrBayes**: Naive Bayes classifier
- **IntgrReduce**: PCA dimensionality reduction
- **IntgrEdge**: Universal inference engine

**Features**:
- Pure integer arithmetic (zero floating-point ops)
- Int8 quantization with percentile-based binning
- Bit-exact deterministic training and inference
- Edge-ready models (no FPU required)
- 20× faster training vs float baseline
- 2× faster inference
- 4× smaller memory footprint

**CLI**:
- Unified `intgrmlc` command-line interface
- 6 modules: `boost`, `forest`, `linear`, `bayes`, `reduce`, `edge`
- CSV input/output
- Model serialization: `.sbf`, `.sff`, `.slf`, `.sbn`, `.srf`

**Documentation**:
- `README.md` - Getting started guide
- `CLI_REFERENCE.md` - Complete command specification
- `docs/benchmarks.md` - Performance baselines

---

## Release Notes Format

Each release includes:
- **Version number**: [Major.Minor.Patch]
- **Date**: YYYY-MM-DD
- **Categories**:
  - ⚠️ BREAKING CHANGES - Non-backward compatible changes
  - 🐛 Fixed - Bug fixes
  - ✨ Added - New features
  - 🔧 Changed - Changes in existing functionality
  - 🗑️ Deprecated - Soon-to-be removed features
  - ❌ Removed - Removed features
  - 🔒 Security - Security fixes
  - 📈 Performance - Performance improvements
  - 🧪 Testing - Test coverage changes
  - 📚 Documentation - Documentation updates

---

**Legend**:
- ⚠️ Breaking Change
- 🐛 Bug Fix
- ✨ New Feature
- 🔧 Change
- 📈 Performance
- 🧪 Testing
- 📚 Documentation
- 🔒 Security

---

For detailed technical analysis and migration guides, see:
- [CRITICAL_BUG_FIX.md](CRITICAL_BUG_FIX.md) - v1.2.1 learning rate bug details
- [OVR_DESIGN.md](OVR_DESIGN.md) - v1.2.0 multiclass architecture
- [FINDINGS_v1.1.md](FINDINGS_v1.1.md) - v1.1.0 quality improvements

# Changelog

All notable changes to IntgrML will be documented in this file.

This project uses [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2024-12-28

### Added
- Initial public release of IntgrML
- Integer-only machine learning engine with pure fixed-point arithmetic
- C++ API for model loading and inference
- Command-line tool (`intgrmlc`) with support for:
  - IntgrBoost: Gradient boosted trees (binary and regression)
  - IntgrBoost-OvR: One-vs-Rest multiclass classification
  - IntgrForest: Random forests
  - IntgrLinear: Linear and logistic regression
  - IntgrBayes: Naive Bayes classifier
  - IntgrReduce: PCA dimensionality reduction
- Python package (`intgrml`) with scikit-learn compatible API
- Determinism verification (`intgrmlc boost determinism`)
- MCU export for embedded deployment (`intgrmlc boost export`)
- Desktop/MCU parity testing (`intgrmlc boost parity`)

### Technical Details
- Q8.8 fixed-point values with Q16.16 accumulator
- Bit-exact reproducibility across platforms
- 65,535 feature limit enforced across all trainers and runtimes
- Model formats: `.sbf` (Boost), `.sff` (Forest), `.slf` (Linear), `.sbn` (Bayes), `.srf` (PCA), `.ovr` (OvR)

### Platforms
- Linux x86_64: Tier 1 (validated, supported)
- Windows x64: Tier 1 (committed, validation in progress)

### License
- Commercial & Community License with free tier for small businesses and researchers
- See COMMERCIAL_LICENSE.md for full terms

---

## [Unreleased]

### Planned
- Binary wheels for pip installation
- Windows full validation
- Additional platform support (macOS, ARM)

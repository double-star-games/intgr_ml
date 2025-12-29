# IntgrML Python Quickstart

**Version**: 1.2.1
**Status**: Beta (Python bindings)

---

## Try It in 2 Minutes

Want to see IntgrML working immediately? Run one of our hello world examples:

```bash
# Quick verification (no files needed)
python -m intgrml.hello_world

# Or run a specific example
python python/examples/hello_world/binary_classification.py
```

For complete, minimal examples, see `python/examples/hello_world/`:
- `binary_classification.py` - Basic 2-class classification
- `multiclass_classification.py` - 3-class OvR classification
- `determinism_demo.py` - Bit-exact reproducibility demo

---

## Installation

### From Source (Current)

```bash
# Clone repository
git clone https://github.com/pmeade/intgr_ml.git
cd intgr_ml

# Install dependencies
pip install numpy pybind11 scikit-learn

# Build and install
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install intgrml
```

---

## Quick Example

### Simple API

```python
import numpy as np
from intgrml import Boost

# Create dataset
X_train = np.random.randn(1000, 10)
y_train = (X_train[:, 0] > 0).astype(np.int32)

# Train model
model = Boost(trees=100, depth=6)
model.fit(X_train, y_train)

# Predict
X_test = np.random.randn(100, 10)
predictions = model.predict(X_test)

print(f"Predictions: {predictions[:10]}")
```

---

## scikit-learn Compatible API

### Drop-in Replacement for XGBoost/LightGBM

```python
from intgrml import IntgrBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create dataset
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train (identical API to XGBoost)
clf = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.25,
    random_state=42
)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Get probabilities
y_proba = clf.predict_proba(X_test)
print(f"Probability shape: {y_proba.shape}")

# Feature importances
importances = clf.feature_importances_
print(f"Most important feature: {importances.argmax()}")
```

---

## Pandas Integration

```python
import pandas as pd
from intgrml import IntgrBoostClassifier

# Load data as DataFrame
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train (automatically handles DataFrames)
clf = IntgrBoostClassifier(n_estimators=100)
clf.fit(X, y)

# Predict
predictions = clf.predict(X)

# Feature names preserved
print(f"Features used: {clf.feature_names_in_}")
```

---

## Model Persistence

```python
from intgrml import Boost

# Train model
model = Boost(trees=100, depth=6)
model.fit(X_train, y_train)

# Save model (.sbf format)
model.save("model.sbf")

# Load model
model_loaded = Boost()
model_loaded.load("model.sbf")

# Predictions are identical
pred1 = model.predict(X_test)
pred2 = model_loaded.predict(X_test)
assert (pred1 == pred2).all()
```

---

## Hyperparameter Tuning with GridSearchCV

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.25, 0.5],
}

# Grid search
clf = IntgrBoostClassifier(random_state=42)
grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Best model
print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.2%}")

# Use best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
```

---

## Deterministic Training

```python
from intgrml import IntgrBoostClassifier

# Train twice with same seed
clf1 = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf1.fit(X_train, y_train)

clf2 = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf2.fit(X_train, y_train)

# Predictions are bit-exact identical
pred1 = clf1.predict(X_test)
pred2 = clf2.predict(X_test)
assert (pred1 == pred2).all()  # ✓ Always passes
```

---

## Edge Deployment Workflow

```python
from intgrml import IntgrBoostClassifier

# 1. Train on desktop
clf = IntgrBoostClassifier(n_estimators=200, max_depth=6)
clf.fit(X_train, y_train)

# 2. Save model
clf.save("model.sbf")

# 3. Export to MCU format (from CLI)
# $ intgrmlc boost export --model model.sbf --out model.mcu

# 4. Verify parity
# $ intgrmlc boost parity --model model.sbf --data test.csv --samples 100
# Output: Parity: 100/100 samples match (100.00%)

# 5. Deploy model.mcu to embedded device (STM32, ESP32, etc.)
```

---

## API Compatibility: scikit-learn Standard

IntgrML implements the standard scikit-learn API, making it compatible with XGBoost, LightGBM, and the entire sklearn ecosystem:

```python
# Standard sklearn-compatible API (works with any sklearn-compatible library)
from intgrml import IntgrBoostClassifier

clf = IntgrBoostClassifier(n_estimators=100, max_depth=6, learning_rate=0.25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

**Why IntgrML is Different:**
- IntgrML uses **integer-only arithmetic** (Q8.8 and Q16.16 fixed-point)
- IntgrML is **bit-exact deterministic** (same input → same output, every time)
- IntgrML requires **no FPU** (runs on microcontrollers)
- IntgrML provides **desktop/device parity** (train on laptop, deploy to MCU bit-identically)

**Trade-off:** ~5-7pp accuracy loss vs float-based GBDT for determinism and edge deployment

---

## What IntgrML Excels At

**Unique Advantages:**

| Feature | Float GBDT Libraries | IntgrML |
|---------|---------------------|--------|
| **Deterministic** | ❌ Non-deterministic | ✅ Bit-exact |
| **Edge Deployment** | ⚠️ FPU required | ✅ No FPU needed |
| **Model Size** | Float32 (large) | **4× smaller** (int8) |
| **Reproducibility** | ❌ Different runs vary | ✅ Perfect reproducibility |
| **Desktop/MCU Parity** | ❌ Different results | ✅ Bit-identical |
| **sklearn API** | ✅ Yes | ✅ Yes |

**Performance (Benchmark: HIGGS 100k, Intel i7-12700K):**
- Training: ~20× faster than unoptimized float baseline (27s vs 540s)
- Inference: ~2× faster than baseline
- Accuracy: ~5pp lower than float GBDT (68% vs 72-75%)

*Note: Performance comparisons are configuration-specific. Float GBDT libraries like XGBoost and LightGBM are excellent tools with different design goals. IntgrML prioritizes determinism and edge deployment over maximum accuracy.*

---

## Current Limitations (v1.2.1)

- **Binary classification only** (multiclass via OvR supported in CLI, Python coming in v1.3)
- **No GPU support** (CPU and embedded only)
- **No sparse features** (dense arrays only)
- **No categorical features** (must be label-encoded first)
- **No sample weights** (coming in v1.3)

---

## Roadmap

### v1.3.0 (Q1 2026)
- ✅ Multiclass classification in Python API
- ✅ Sample weights support
- ✅ Categorical feature handling
- ✅ Sparse matrix support

### v1.4.0 (Q2 2026)
- ✅ ONNX export
- ✅ TensorFlow Lite export
- ✅ GPU acceleration (optional)

---

## Getting Help

- **Documentation**: https://github.com/pmeade/intgr_ml
- **Issues**: https://github.com/pmeade/intgr_ml/issues
- **Discussions**: https://github.com/pmeade/intgr_ml/discussions
- **Examples**: `python/examples/` directory

---

## License

IntgrML is free for non-commercial research and evaluation.
Commercial use requires a license from **Pluperfect Biomimetics / Double Star**.

**Contact**: patrick@doublestar.net

---

## Acknowledgments

IntgrML's Python API is designed to be compatible with the scikit-learn ecosystem. We're grateful to:

- **scikit-learn team** for establishing excellent API standards and providing `BaseEstimator` and `ClassifierMixin` base classes (BSD license)
- **XGBoost and LightGBM teams** for demonstrating how to build sklearn-compatible gradient boosting libraries
- **pybind11 team** for excellent C++/Python binding tools

IntgrML implements the standard sklearn API to enable ecosystem compatibility while providing unique advantages for edge deployment and deterministic systems.

---

**Next Steps:**
1. Try the [Quickstart Example](python/examples/01_quickstart.ipynb)
2. Read the [Full Documentation](docs/)
3. Join the [Discussion Forum](https://github.com/pmeade/intgr_ml/discussions)

# IntgrML Python API Reference

**Version**: 1.2.1
**Status**: Production Ready
**License**: Commercial & Community License (free for personal, research, and early-stage startup use; commercial tiers available). See `COMMERCIAL_LICENSE.md` for details.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core API](#core-api)
   - [Boost](#boost)
   - [Forest](#forest)
4. [scikit-learn API](#scikit-learn-api)
   - [IntgrBoostClassifier](#intgrboostclassifier)
   - [IntgrForestClassifier](#intgrforestclassifier)
   - [IntgrBoostRegressor](#intgrboostregressor)
5. [Common Patterns](#common-patterns)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/pmeade/intgr_ml.git
cd intgr_ml

# Install dependencies
pip install numpy pybind11 scikit-learn pandas

# Build and install
pip install -e .
```

### Requirements

**Required:**
- Python 3.10+
- NumPy >= 1.19.0
- pybind11 >= 2.10.0
- CMake >= 3.18
- C++20 compiler (GCC 11+, Clang 15+, MSVC 2022+)

**Optional:**
- scikit-learn >= 0.24.0 (for GridSearchCV, validation utilities)
- pandas >= 1.0.0 (for DataFrame support)

---

## Quick Start

### Simple Example

```python
import numpy as np
from intgrml import IntgrBoostClassifier

# Create dataset
X = np.random.randn(1000, 20)
y = (X[:, 0] > 0).astype(np.int32)

# Train model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### With scikit-learn

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
}
grid = GridSearchCV(IntgrBoostClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.best_estimator_.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
```

---

## Core API

The core API provides direct access to IntgrML's C++ implementation with minimal overhead.

### Boost

Low-level gradient boosting interface.

```python
from intgrml import Boost
```

#### Constructor

```python
Boost(trees=100, depth=6, bins=256, learning_rate=0.25,
      min_samples_leaf=8, random_state=42)
```

**Parameters:**

- **trees** : `int`, default=100
  Number of boosting rounds (trees to build)

- **depth** : `int`, default=6
  Maximum tree depth. Range: 1-15
  Deeper trees model more complex patterns but may overfit

- **bins** : `int`, default=256
  Number of histogram bins for feature quantization
  Higher values preserve more information but use more memory
  Must be power of 2: 64, 128, 256 (recommended), 512

- **learning_rate** : `float`, default=0.25
  Boosting learning rate (shrinkage). Range: 0.0-1.0
  Lower values require more trees but may generalize better

- **min_samples_leaf** : `int`, default=8
  Minimum number of samples required in a leaf node
  Higher values prevent overfitting on small data

- **random_state** : `int`, default=42
  Random seed for reproducibility
  IntgrML guarantees bit-exact reproducibility with same seed

**Example:**

```python
model = Boost(trees=200, depth=8, learning_rate=0.1, random_state=42)
```

#### Methods

##### `fit(X, y, clip_min=-10.0, clip_max=10.0)`

Train the boosting model.

**Parameters:**

- **X** : `ndarray` of shape `(n_samples, n_features)`
  Training data (float64)

- **y** : `ndarray` of shape `(n_samples,)`
  Target values (int32). Binary: 0 or 1

- **clip_min** : `float`, default=-10.0
  Minimum value for feature clipping during quantization

- **clip_max** : `float`, default=10.0
  Maximum value for feature clipping during quantization

**Returns:** `None`

**Example:**

```python
import numpy as np

X_train = np.random.randn(1000, 10)
y_train = (X_train[:, 0] > 0).astype(np.int32)

model = Boost(trees=100, depth=6)
model.fit(X_train, y_train, clip_min=-5.0, clip_max=5.0)
```

**Notes:**
- Features outside [clip_min, clip_max] are clipped before quantization
- Quantization uses Q8.8 fixed-point format (8-bit values)
- Same clip_min/max must be used for prediction (saved with model)

##### `predict(X)`

Predict raw logits for samples.

**Parameters:**

- **X** : `ndarray` of shape `(n_samples, n_features)`
  Samples to predict (float64)

**Returns:**

- **logits** : `ndarray` of shape `(n_samples,)` (int32)
  Raw prediction logits
  Positive values → class 1
  Negative values → class 0

**Example:**

```python
X_test = np.random.randn(100, 10)
logits = model.predict(X_test)
# logits are in Q8.8 format (divide by 256 for actual value)
```

##### `predict_class(X)`

Predict class labels for samples.

**Parameters:**

- **X** : `ndarray` of shape `(n_samples, n_features)`
  Samples to predict (float64)

**Returns:**

- **labels** : `ndarray` of shape `(n_samples,)` (int32)
  Predicted class labels (0 or 1)

**Example:**

```python
predictions = model.predict_class(X_test)
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
```

##### `save(path)`

Save model to file in .sbf (IntgrBoost Format) format.

**Parameters:**

- **path** : `str`
  File path to save model (e.g., "model.sbf")

**Returns:** `None`

**Example:**

```python
model.save("trained_model.sbf")
```

**File Format:**
- Binary format with model structure and weights
- Cross-platform compatible
- Typical size: 10-100 KB for 100 trees

##### `load(path)`

Load model from .sbf file.

**Parameters:**

- **path** : `str`
  File path to load model from

**Returns:** `None`

**Example:**

```python
model = Boost()
model.load("trained_model.sbf")
predictions = model.predict_class(X_test)
```

**Note:** Quantization parameters (clip_min, clip_max) reset to defaults (-10.0, 10.0)

#### Properties

##### `feature_importances_`

Get feature importances (read-only).

**Returns:**

- **importances** : `ndarray` of shape `(n_features,)` (float64)
  Normalized feature importances (sum to 1.0)
  Based on split count across all trees

**Example:**

```python
importances = model.feature_importances_
top_features = np.argsort(importances)[-5:][::-1]
print(f"Top 5 features: {top_features}")
```

##### `n_trees`

Number of trees in the model (read-only).

**Returns:** `int`

##### `n_features`

Number of features the model was trained on (read-only).

**Returns:** `int`

##### `is_fitted`

Whether the model has been fitted (read-only).

**Returns:** `bool`

---

### Forest

Low-level random forest interface.

```python
from intgrml import Forest
```

#### Constructor

```python
Forest(trees=100, depth=8, bins=256, min_samples_leaf=10,
       subsample=0.8, random_state=42)
```

**Parameters:**

- **trees** : `int`, default=100
  Number of trees in the forest

- **depth** : `int`, default=8
  Maximum tree depth

- **bins** : `int`, default=256
  Number of histogram bins

- **min_samples_leaf** : `int`, default=10
  Minimum samples per leaf

- **subsample** : `float`, default=0.8
  Fraction of samples to use for each tree (bagging)
  Range: 0.0-1.0

- **random_state** : `int`, default=42
  Random seed

#### Methods

Methods are similar to `Boost` class:
- `fit(X, y, clip_min, clip_max)`
- `predict(X)` - Returns logits
- `save(path)`
- `load(path)`

Properties:
- `n_trees`
- `n_features`
- `is_fitted`

---

## scikit-learn API

The sklearn-compatible API provides full integration with scikit-learn's ecosystem.

### IntgrBoostClassifier

Gradient boosted decision trees for binary classification, compatible with scikit-learn.

```python
from intgrml import IntgrBoostClassifier
```

#### Constructor

```python
IntgrBoostClassifier(n_estimators=100, max_depth=6, learning_rate=0.25,
                    min_samples_leaf=8, n_bins=256, clip_min=-10.0,
                    clip_max=10.0, random_state=None)
```

**Parameters:**

- **n_estimators** : `int`, default=100
  Number of boosting rounds (trees)
  Equivalent to `trees` in core API

- **max_depth** : `int`, default=6
  Maximum tree depth
  Equivalent to `depth` in core API

- **learning_rate** : `float`, default=0.25
  Boosting learning rate (shrinkage). Range: 0.0-1.0

- **min_samples_leaf** : `int`, default=8
  Minimum samples required in a leaf node

- **n_bins** : `int`, default=256
  Number of histogram bins for quantization

- **clip_min** : `float`, default=-10.0
  Minimum value for feature clipping

- **clip_max** : `float`, default=10.0
  Maximum value for feature clipping

- **random_state** : `int` or `None`, default=None
  Random seed for reproducibility
  If None, uses 42 as default

**Example:**

```python
clf = IntgrBoostClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
```

#### Methods

##### `fit(X, y, sample_weight=None)`

Fit the gradient boosting model.

**Parameters:**

- **X** : `array-like` of shape `(n_samples, n_features)`
  Training data
  Accepts: NumPy arrays, pandas DataFrames

- **y** : `array-like` of shape `(n_samples,)`
  Target values (binary: 0 or 1)
  Accepts: NumPy arrays, pandas Series

- **sample_weight** : `array-like` of shape `(n_samples,)` or `None`
  Sample weights (not yet supported in v1.2.1)
  Raises NotImplementedError if provided

**Returns:**

- **self** : `IntgrBoostClassifier`
  Fitted estimator

**Sets Attributes:**
- `n_features_in_` : Number of features
- `feature_names_in_` : Feature names (if DataFrame)
- `classes_` : Class labels [0, 1]
- `model_` : Internal C++ model

**Example:**

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
```

**With Pandas:**

```python
import pandas as pd

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_series = pd.Series(y, name="target")

clf.fit(df, y_series)
print(f"Features used: {clf.feature_names_in_}")
```

##### `predict(X)`

Predict class labels for samples.

**Parameters:**

- **X** : `array-like` of shape `(n_samples, n_features)`
  Samples to predict

**Returns:**

- **y_pred** : `ndarray` of shape `(n_samples,)` (int32)
  Predicted class labels (0 or 1)

**Example:**

```python
y_pred = clf.predict(X_test)
```

##### `predict_proba(X)`

Predict class probabilities for samples.

**Parameters:**

- **X** : `array-like` of shape `(n_samples, n_features)`
  Samples to predict

**Returns:**

- **proba** : `ndarray` of shape `(n_samples, 2)` (float64)
  Class probabilities
  Column 0: P(y=0 | X)
  Column 1: P(y=1 | X)
  Rows sum to 1.0

**Example:**

```python
proba = clf.predict_proba(X_test)
# Get probability of positive class
prob_positive = proba[:, 1]
```

**Note:** Applies sigmoid transformation to integer logits:
```python
p(y=1) = 1 / (1 + exp(-logit / 100))
```

##### `decision_function(X)`

Compute decision function of X.

**Parameters:**

- **X** : `array-like` of shape `(n_samples, n_features)`
  Samples

**Returns:**

- **decision** : `ndarray` of shape `(n_samples,)` (float64)
  Decision function values (raw logits as floats)
  Positive → class 1, Negative → class 0

**Example:**

```python
decision = clf.decision_function(X_test)
# Use custom threshold
y_pred_custom = (decision > 50).astype(int)
```

##### `score(X, y, sample_weight=None)`

Return the mean accuracy on the given test data and labels.

**Parameters:**

- **X** : `array-like` of shape `(n_samples, n_features)`
  Test samples

- **y** : `array-like` of shape `(n_samples,)`
  True labels

- **sample_weight** : `array-like` of shape `(n_samples,)` or `None`
  Sample weights (not supported)

**Returns:**

- **score** : `float`
  Mean accuracy

**Example:**

```python
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

##### `get_params(deep=True)`

Get parameters for this estimator (sklearn compatibility).

**Parameters:**

- **deep** : `bool`, default=True
  If True, return parameters for sub-objects

**Returns:**

- **params** : `dict`
  Parameter names mapped to their values

##### `set_params(**params)`

Set parameters for this estimator (sklearn compatibility).

**Parameters:**

- **params** : `dict`
  Estimator parameters

**Returns:**

- **self** : `IntgrBoostClassifier`

**Example:**

```python
clf.set_params(n_estimators=200, max_depth=8)
```

##### `save(path)`

Save model to file (.sbf format).

**Parameters:**

- **path** : `str`
  File path

**Returns:** `None`

**Example:**

```python
clf.save("model.sbf")
```

##### `load(path)`

Load model from file (.sbf format).

**Parameters:**

- **path** : `str`
  File path

**Returns:** `None`

**Example:**

```python
clf = IntgrBoostClassifier()
clf.load("model.sbf")
```

#### Attributes

##### `n_features_in_`

Number of features seen during fit.

**Type:** `int`

##### `feature_names_in_`

Names of features seen during fit (if DataFrame).

**Type:** `list` of `str`

Only available if fit with pandas DataFrame.

##### `classes_`

Class labels.

**Type:** `ndarray` of shape `(2,)`

Always `[0, 1]` for binary classification.

##### `feature_importances_`

Feature importances (normalized).

**Type:** `ndarray` of shape `(n_features_in_,)` (float64)

Normalized to sum to 1.0.

**Example:**

```python
import matplotlib.pyplot as plt

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.bar(range(len(importances)), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()
```

---

### IntgrForestClassifier

Random forest classifier for binary classification, compatible with scikit-learn.

```python
from intgrml import IntgrForestClassifier
```

#### Constructor

```python
IntgrForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10,
                     n_bins=256, subsample=0.8, clip_min=-10.0,
                     clip_max=10.0, random_state=None)
```

**Parameters:**

Similar to IntgrBoostClassifier, with addition of:

- **subsample** : `float`, default=0.8
  Fraction of samples to use for each tree (bagging)
  Range: 0.0-1.0

#### Methods

Same methods as IntgrBoostClassifier:
- `fit(X, y, sample_weight=None)`
- `predict(X)`
- `predict_proba(X)` — Returns class probabilities
- `decision_function(X)` — Returns raw logits
- `score(X, y)` — Returns mean accuracy
- `save(path)`
- `load(path)`
- `get_params(deep=True)`
- `set_params(**params)`

#### Attributes

- `n_features_in_`
- `feature_names_in_`
- `classes_`

---

### IntgrBoostRegressor

Gradient boosted trees for regression (not yet implemented).

```python
from intgrml import IntgrBoostRegressor
```

**Status:** Coming in v1.3.0

All methods raise `NotImplementedError`.

---

## Common Patterns

### Pattern 1: Train/Test Split

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Pattern 2: Cross-Validation

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import cross_val_score

clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
```

### Pattern 3: Hyperparameter Tuning

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.25, 0.5],
}

clf = IntgrBoostClassifier(random_state=42)
grid = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# Use best model
best_clf = grid.best_estimator_
y_pred = best_clf.predict(X_test)
```

### Pattern 4: Model Persistence

```python
import pickle
from intgrml import IntgrBoostClassifier

# Train model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Option 1: IntgrML native format (.sbf)
clf.save("model.sbf")

clf_loaded = IntgrBoostClassifier()
clf_loaded.load("model.sbf")

# Option 2: Pickle (saves sklearn wrapper too)
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("model.pkl", "rb") as f:
    clf_loaded = pickle.load(f)
```

### Pattern 5: Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from intgrml import IntgrBoostClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', IntgrBoostClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Note:** Scaling is optional for IntgrML as it quantizes features anyway.

### Pattern 6: Early Stopping (Manual)

```python
from intgrml import IntgrBoostClassifier
from sklearn.metrics import accuracy_score

best_score = 0
best_n_trees = 0

for n in [10, 25, 50, 100, 200, 500]:
    clf = IntgrBoostClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    score = accuracy_score(y_val, clf.predict(X_val))

    if score > best_score:
        best_score = score
        best_n_trees = n

    print(f"n_estimators={n}: {score:.3f}")

print(f"\nBest: {best_n_trees} trees with {best_score:.3f} accuracy")
```

### Pattern 7: Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
from intgrml import IntgrBoostClassifier

# Train base model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)

# Calibrate probabilities
calibrated = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
calibrated.fit(X_train, y_train)

# Get calibrated probabilities
proba = calibrated.predict_proba(X_test)
```

---

## Performance Tips

### 1. Quantization Parameters

Choose `clip_min` and `clip_max` based on your data:

```python
# Analyze feature distribution
import numpy as np

# Use percentiles to avoid outliers
clip_min = np.percentile(X_train, 1)
clip_max = np.percentile(X_train, 99)

clf = IntgrBoostClassifier(clip_min=clip_min, clip_max=clip_max)
clf.fit(X_train, y_train)
```

### 2. Memory vs Accuracy Trade-off

```python
# High memory, high accuracy
clf = IntgrBoostClassifier(n_bins=512, n_estimators=200, max_depth=8)

# Low memory, lower accuracy
clf = IntgrBoostClassifier(n_bins=64, n_estimators=50, max_depth=4)
```

### 3. Training Speed

```python
# Fast training
clf = IntgrBoostClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_leaf=20
)

# Slower but better accuracy
clf = IntgrBoostClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5
)
```

### 4. Overfitting Prevention

```python
# Prevent overfitting
clf = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=4,              # Shallow trees
    learning_rate=0.1,        # Low learning rate
    min_samples_leaf=20       # More samples per leaf
)
```

### 5. Parallel Processing with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Use all CPU cores
grid = GridSearchCV(
    IntgrBoostClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,              # Use all cores
    verbose=2
)
grid.fit(X_train, y_train)
```

---

## Troubleshooting

### Issue: Model gives different predictions after save/load

**Cause:** Quantization parameters (clip_min, clip_max) not stored in .sbf format.

**Solution:** Use same values for training and loading:

```python
# Training
clf = IntgrBoostClassifier(clip_min=-5.0, clip_max=5.0)
clf.fit(X_train, y_train)
clf.save("model.sbf")

# Loading - must recreate with same params
clf2 = IntgrBoostClassifier(clip_min=-5.0, clip_max=5.0)
clf2.load("model.sbf")
```

### Issue: Low accuracy compared to XGBoost

**Causes:**
1. IntgrML uses integer arithmetic (5-7pp accuracy loss expected)
2. Quantization parameters not optimal
3. Not enough trees/depth

**Solutions:**

```python
# 1. Increase model capacity
clf = IntgrBoostClassifier(n_estimators=300, max_depth=8)

# 2. Optimize quantization
clip_min = np.percentile(X_train, 1)
clip_max = np.percentile(X_train, 99)
clf = IntgrBoostClassifier(clip_min=clip_min, clip_max=clip_max)

# 3. Lower learning rate, more trees
clf = IntgrBoostClassifier(n_estimators=500, learning_rate=0.05)
```

### Issue: Training is slow

**Solutions:**

```python
# 1. Reduce number of trees
clf = IntgrBoostClassifier(n_estimators=50)

# 2. Reduce tree depth
clf = IntgrBoostClassifier(max_depth=4)

# 3. Increase min_samples_leaf
clf = IntgrBoostClassifier(min_samples_leaf=50)

# 4. Reduce bins
clf = IntgrBoostClassifier(n_bins=128)
```

### Issue: Memory usage too high

**Solutions:**

```python
# Use fewer bins
clf = IntgrBoostClassifier(n_bins=64)

# Use fewer trees
clf = IntgrBoostClassifier(n_estimators=50)

# Use shallower trees
clf = IntgrBoostClassifier(max_depth=4)
```

### Issue: NotImplementedError for sample_weight

**Workaround:** Resample data with replacement according to weights:

```python
import numpy as np

# Create weighted sample
n_samples = len(X_train)
probs = sample_weights / sample_weights.sum()
indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probs)

X_resampled = X_train[indices]
y_resampled = y_train[indices]

clf.fit(X_resampled, y_resampled)
```

### Issue: ValueError with "intgr_ml v1 supports up to 65,535 features"

**Cause:** IntgrML v1 enforces a maximum of 65,535 features (u16 limit).

**Solutions:**

```python
# 1. Use dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=1000)
X_reduced = pca.fit_transform(X_train)

clf = IntgrBoostClassifier()
clf.fit(X_reduced, y_train)

# 2. Use IntgrReduce (native integer PCA)
# (Available via CLI: intgrmlc reduce train ...)

# 3. Use feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10000)
X_selected = selector.fit_transform(X_train, y_train)

clf = IntgrBoostClassifier()
clf.fit(X_selected, y_train)
```

### Issue: Multiclass classification not supported

**Workaround:** Use One-vs-Rest manually:

```python
from sklearn.multiclass import OneVsRestClassifier
from intgrml import IntgrBoostClassifier

ovr = OneVsRestClassifier(IntgrBoostClassifier(n_estimators=100))
ovr.fit(X_train, y_multiclass)
y_pred = ovr.predict(X_test)
```

**Note:** Native multiclass support coming in v1.3.0.

---

## Ecosystem Compatibility

### sklearn API Standard

IntgrML implements the standard scikit-learn API, enabling compatibility with the entire sklearn ecosystem:

```python
# Standard sklearn-compatible API
from intgrml import IntgrBoostClassifier

clf = IntgrBoostClassifier(n_estimators=100, max_depth=6, learning_rate=0.25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Works with sklearn utilities
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid, cv=3)
```

**API Compatibility:** IntgrML, XGBoost, and LightGBM all implement the sklearn `BaseEstimator` interface, making them interchangeable in most workflows.

### IntgrML vs Float GBDT Libraries

| Feature | Float GBDT (e.g., XGBoost) | IntgrML |
|---------|---------------------------|--------|
| **Arithmetic** | Float32 | Int8/Int32 (Q8.8, Q16.16) |
| **Deterministic** | ❌ Non-deterministic | ✅ Bit-exact |
| **Edge Deployment** | ⚠️ FPU required | ✅ No FPU needed |
| **Model Size** | Float32 (larger) | **4× smaller** (int8) |
| **Desktop/MCU Parity** | ❌ Different results | ✅ Bit-identical |
| **GPU Support** | ✅ Yes | ❌ No |
| **Accuracy** | Higher (float precision) | Lower (~5pp trade-off) |

**Benchmark (HIGGS 100k, Intel i7-12700K):**
- Training: ~20× faster than unoptimized float baseline
- Inference: ~2× faster than baseline
- Accuracy: ~5pp lower (68% vs 72-75%)

*Disclaimer: Performance varies with datasets, hardware, and library versions. XGBoost and LightGBM are excellent libraries with different design goals. IntgrML prioritizes determinism and edge deployment over maximum accuracy.*

### When to Use IntgrML

✅ **Choose IntgrML for:**
- **Edge/MCU deployment** (STM32, ESP32, ARM Cortex-M)
- **Deterministic systems** (regulatory, safety-critical)
- **Reproducible research** (bit-exact results)
- **No FPU environments** (embedded, WASM)
- **Desktop/device parity** (train on laptop, deploy to MCU identically)

❌ **Choose float GBDT libraries for:**
- **Maximum accuracy** (IntgrML trades ~5pp for other benefits)
- **GPU acceleration** (IntgrML is CPU/embedded only)
- **Cloud-scale training** (distributed XGBoost/LightGBM)
- **Sparse features** (coming in IntgrML v1.3.0)

---

## Version History

### v1.2.1 (Current)
- Initial Python bindings release
- Binary classification support
- sklearn API compatibility
- Model serialization
- Pandas integration
- 90% test coverage

### v1.3.0 (Planned Q1 2026)
- Multiclass classification
- Sample weights support
- Sparse matrix support
- Store quantization params in .sbf
- Categorical features

### v1.4.0 (Planned Q2 2026)
- ONNX export
- TensorFlow Lite export
- Optional GPU acceleration

---

## Support

- **Documentation**: https://github.com/pmeade/intgr_ml
- **Issues**: https://github.com/pmeade/intgr_ml/issues
- **Discussions**: https://github.com/pmeade/intgr_ml/discussions
- **Email**: patrick@doublestar.net

---

## Acknowledgments

IntgrML's Python bindings are designed for compatibility with the scikit-learn ecosystem. We're grateful to:

- **scikit-learn team** - For establishing excellent API standards and providing the `BaseEstimator` and `ClassifierMixin` base classes (BSD license), which enable ecosystem compatibility
- **XGBoost and LightGBM teams** - For demonstrating how to build high-quality sklearn-compatible gradient boosting libraries
- **pybind11 team** - For providing excellent tools for C++/Python interoperability

IntgrML implements the standard sklearn API to enable seamless integration with the Python ML ecosystem while providing unique advantages for deterministic and edge deployment scenarios.

---

## License

IntgrML is licensed under the **Commercial & Community License**.

- Free for personal, research, and early-stage startup use (Revenue ≤ $200k, Funding ≤ $2M)
- Commercial tiers and OEM options available for larger organizations

See `COMMERCIAL_LICENSE.md` and `LICENSING_FAQ.md` for full terms.

**Contact**: sales@doublestar.net

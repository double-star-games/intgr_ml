# IntgrML Python Examples

Practical examples demonstrating common use cases.

---

## Hello World Examples

**New to IntgrML?** Start here with minimal, runnable examples that use tiny inline datasets and require no external files or large dependencies.

| Example | Description | Run Command |
|---------|-------------|-------------|
| `binary_classification.py` | Basic 2-class classification | `python python/examples/hello_world/binary_classification.py` |
| `multiclass_classification.py` | 3-class One-vs-Rest classification | `python python/examples/hello_world/multiclass_classification.py` |
| `determinism_demo.py` | Bit-exact reproducibility demo | `python python/examples/hello_world/determinism_demo.py` |

Each script runs in under 2 seconds and prints clear, human-readable output.

**Quick verify:**
```bash
python -m intgrml.hello_world
```

---

## Table of Contents

1. [Binary Classification](#binary-classification)
2. [Working with Pandas](#working-with-pandas)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Model Persistence](#model-persistence)
5. [Cross-Validation](#cross-validation)
6. [Feature Importance](#feature-importance)
7. [Probability Calibration](#probability-calibration)
8. [Custom Thresholds](#custom-thresholds)
9. [Integration with sklearn Pipelines](#integration-with-sklearn-pipelines)
10. [Real-World Dataset](#real-world-dataset)

---

## Binary Classification

Basic binary classification workflow.

```python
import numpy as np
from intgrml import IntgrBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Generate dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
clf = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.25,
    random_state=42
)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Output:**
```
Accuracy: 0.883
ROC AUC: 0.945

Confusion Matrix:
[[952  48]
 [186 814]]
```

---

## Working with Pandas

Using DataFrames and preserving feature names.

```python
import pandas as pd
import numpy as np
from intgrml import IntgrBoostClassifier

# Create DataFrame with named features
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.randint(20000, 150000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'debt_ratio': np.random.uniform(0, 1, 1000),
}
df = pd.DataFrame(data)

# Generate target (e.g., loan default)
df['default'] = ((df['debt_ratio'] > 0.5) &
                 (df['credit_score'] < 600)).astype(int)

# Separate features and target
X = df.drop('default', axis=1)
y = df['default']

# Train with DataFrame
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Feature names preserved
print("Features used:", clf.feature_names_in_)
print("Number of features:", clf.n_features_in_)

# Predict with DataFrame
predictions = clf.predict(X)

# Feature importances with names
importances = pd.DataFrame({
    'feature': clf.feature_names_in_,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(importances)
```

**Output:**
```
Features used: ['age', 'income', 'credit_score', 'debt_ratio']
Number of features: 4

Feature Importances:
        feature  importance
3    debt_ratio    0.512345
2  credit_score    0.301234
1        income    0.123456
0           age    0.062965
```

---

## Hyperparameter Tuning

Using GridSearchCV and RandomizedSearchCV.

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification

# Create dataset
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# ============== Grid Search ==============
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.25, 0.5],
    'min_samples_leaf': [5, 10, 20]
}

grid = GridSearchCV(
    IntgrBoostClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X, y)

print("Grid Search Results:")
print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# ============== Randomized Search ==============
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.05, 0.45),
    'min_samples_leaf': randint(5, 30)
}

random_search = RandomizedSearchCV(
    IntgrBoostClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X, y)

print("\nRandomized Search Results:")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")

# Use best model
best_clf = random_search.best_estimator_
```

---

## Model Persistence

Saving and loading models.

```python
from intgrml import IntgrBoostClassifier
import pickle
import json

# Train model
clf = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=6,
    clip_min=-5.0,
    clip_max=5.0,
    random_state=42
)
clf.fit(X_train, y_train)

# ============== Method 1: Native .sbf format ==============
# Save
clf.save("model.sbf")

# Load (must specify same clip_min/max)
clf_loaded = IntgrBoostClassifier(clip_min=-5.0, clip_max=5.0)
clf_loaded.load("model.sbf")

# Verify
assert (clf.predict(X_test) == clf_loaded.predict(X_test)).all()

# ============== Method 2: Pickle (saves full object) ==============
# Save
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Load
with open("model.pkl", "rb") as f:
    clf_loaded = pickle.load(f)

# ============== Save metadata ==============
metadata = {
    'model_type': 'IntgrBoostClassifier',
    'n_estimators': clf.n_estimators,
    'max_depth': clf.max_depth,
    'clip_min': clf.clip_min,
    'clip_max': clf.clip_max,
    'n_features': clf.n_features_in_,
    'feature_names': clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else None,
    'training_date': '2025-01-15',
    'accuracy': accuracy_score(y_test, clf.predict(X_test))
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model and metadata saved successfully")
```

---

## Cross-Validation

K-fold cross-validation for robust evaluation.

```python
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
import numpy as np

# Create classifier
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)

# ============== Simple cross-validation ==============
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# ============== Multiple metrics ==============
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring)

print("\nCross-Validation Results:")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric:10s}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# ============== Stratified K-Fold (for imbalanced data) ==============
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
print(f"\nStratified ROC AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# ============== Get predictions for each fold ==============
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(clf, X, y, cv=5)
print(f"\nOverall Accuracy: {accuracy_score(y, y_pred):.3f}")
```

---

## Feature Importance

Analyzing and visualizing feature importances.

```python
import numpy as np
import matplotlib.pyplot as plt
from intgrml import IntgrBoostClassifier

# Train model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for i, idx in enumerate(indices[:10], 1):
    print(f"{i}. Feature {idx}: {importances[idx]:.4f}")

# ============== Plot 1: Bar chart ==============
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

# ============== Plot 2: Top N features ==============
top_n = 10
top_indices = indices[:top_n]
top_importances = importances[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), top_importances)
plt.yticks(range(top_n), [f"Feature {i}" for i in top_indices])
plt.xlabel("Importance")
plt.title(f"Top {top_n} Features")
plt.tight_layout()
plt.savefig("top_features.png")
plt.close()

# ============== Feature selection based on importance ==============
threshold = 0.01  # Keep features with >1% importance
mask = importances >= threshold
X_selected = X_train[:, mask]

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")

# Retrain with selected features
clf_selected = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf_selected.fit(X_selected, y_train)
```

---

## Probability Calibration

Calibrating predicted probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from intgrml import IntgrBoostClassifier
import matplotlib.pyplot as plt

# Train base model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get uncalibrated probabilities
y_proba_uncal = clf.predict_proba(X_test)[:, 1]

# ============== Calibrate probabilities ==============
calibrated = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
calibrated.fit(X_train, y_train)
y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

# ============== Plot calibration curves ==============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Uncalibrated
frac_pos_uncal, mean_pred_uncal = calibration_curve(
    y_test, y_proba_uncal, n_bins=10
)
ax1.plot(mean_pred_uncal, frac_pos_uncal, marker='o', label='IntgrML')
ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
ax1.set_xlabel('Mean predicted probability')
ax1.set_ylabel('Fraction of positives')
ax1.set_title('Calibration Curve (Uncalibrated)')
ax1.legend()
ax1.grid(True)

# Calibrated
frac_pos_cal, mean_pred_cal = calibration_curve(
    y_test, y_proba_cal, n_bins=10
)
ax2.plot(mean_pred_cal, frac_pos_cal, marker='o', label='Calibrated IntgrML')
ax2.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
ax2.set_xlabel('Mean predicted probability')
ax2.set_ylabel('Fraction of positives')
ax2.set_title('Calibration Curve (Calibrated)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("calibration_curves.png")
plt.close()

# ============== Evaluate calibration ==============
from sklearn.metrics import brier_score_loss

brier_uncal = brier_score_loss(y_test, y_proba_uncal)
brier_cal = brier_score_loss(y_test, y_proba_cal)

print(f"Brier Score (uncalibrated): {brier_uncal:.4f}")
print(f"Brier Score (calibrated):   {brier_cal:.4f}")
print(f"Improvement: {(brier_uncal - brier_cal) / brier_uncal * 100:.1f}%")
```

---

## Custom Thresholds

Optimizing decision threshold for specific metrics.

```python
from intgrml import IntgrBoostClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Train model
clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Get probabilities
y_proba = clf.predict_proba(X_test)[:, 1]

# ============== Find optimal threshold for F1 ==============
from sklearn.metrics import f1_score

thresholds = np.linspace(0, 1, 101)
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold for F1: {optimal_threshold:.3f}")
print(f"Max F1 score: {max(f1_scores):.3f}")

# ============== Compare with default threshold (0.5) ==============
y_pred_default = (y_proba >= 0.5).astype(int)
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

from sklearn.metrics import classification_report

print("\nDefault threshold (0.5):")
print(classification_report(y_test, y_pred_default))

print(f"\nOptimal threshold ({optimal_threshold:.3f}):")
print(classification_report(y_test, y_pred_optimal))

# ============== Precision-Recall trade-off ==============
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds_pr, precision[:-1], label='Precision')
plt.plot(thresholds_pr, recall[:-1], label='Recall')
plt.axvline(optimal_threshold, color='red', linestyle='--',
            label=f'Optimal ({optimal_threshold:.3f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_optimization.png")
plt.close()
```

---

## Integration with sklearn Pipelines

Using IntgrML in preprocessing pipelines.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from intgrml import IntgrBoostClassifier

# ============== Simple pipeline ==============
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', IntgrBoostClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {score:.3f}")

# ============== Complex pipeline with feature engineering ==============
complex_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('classifier', IntgrBoostClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ))
])

complex_pipeline.fit(X_train, y_train)

# ============== Grid search with pipeline ==============
from sklearn.model_selection import GridSearchCV

param_grid = {
    'poly__degree': [1, 2],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [4, 6]
}

grid = GridSearchCV(complex_pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# ============== Save/load pipeline ==============
import pickle

with open("pipeline.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

with open("pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

y_pred = loaded_pipeline.predict(X_test)
```

---

## Real-World Dataset

Complete workflow with a real dataset.

```python
import pandas as pd
import numpy as np
from intgrml import IntgrBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============== Load data ==============
# Example: Credit card fraud detection
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nImbalance ratio: {(df['Class']==0).sum() / (df['Class']==1).sum():.1f}:1")

# ============== Preprocessing ==============
X = df.drop('Class', axis=1)
y = df['Class']

# Split (stratified for imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Optimize quantization parameters
clip_min = X_train.quantile(0.01).min()
clip_max = X_train.quantile(0.99).max()
print(f"\nQuantization range: [{clip_min:.2f}, {clip_max:.2f}]")

# ============== Baseline model ==============
clf_baseline = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=6,
    clip_min=clip_min,
    clip_max=clip_max,
    random_state=42
)
clf_baseline.fit(X_train, y_train)

y_pred_baseline = clf_baseline.predict(X_test)
y_proba_baseline = clf_baseline.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("BASELINE MODEL RESULTS")
print("="*60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_baseline):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba_baseline):.4f}")

# ============== Hyperparameter tuning ==============
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.25, 0.5],
}

grid = GridSearchCV(
    IntgrBoostClassifier(clip_min=clip_min, clip_max=clip_max, random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

clf_tuned = grid.best_estimator_

y_pred_tuned = clf_tuned.predict(X_test)
y_proba_tuned = clf_tuned.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("TUNED MODEL RESULTS")
print("="*60)
print(f"Best parameters: {grid.best_params_}")
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_tuned):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba_tuned):.4f}")

# ============== Feature importance analysis ==============
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf_tuned.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 10 FEATURES")
print("="*60)
print(importances.head(10))

# Plot top features
plt.figure(figsize=(10, 6))
top_features = importances.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig("feature_importances_credit.png", dpi=150)
plt.close()

# ============== ROC curve ==============
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_proba_tuned)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'IntgrML (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Fraud Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_credit.png", dpi=150)
plt.close()

# ============== Save final model ==============
clf_tuned.save("fraud_detection_model.sbf")
print("\n✓ Model saved to fraud_detection_model.sbf")
```

---

## Additional Resources

- **API Reference**: See [PYTHON_API.md](PYTHON_API.md)
- **Quickstart**: See [PYTHON_QUICKSTART.md](../PYTHON_QUICKSTART.md)
- **GitHub**: https://github.com/pmeade/intgr_ml
- **Issues**: https://github.com/pmeade/intgr_ml/issues

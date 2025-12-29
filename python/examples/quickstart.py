"""
IntgrML Python Quickstart Example

This example demonstrates:
1. Basic usage with numpy arrays
2. scikit-learn compatible API
3. Model serialization
4. Hyperparameter tuning with GridSearchCV
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import tempfile
import os

print("=" * 60)
print("IntgrML Python Quickstart")
print("=" * 60)

# ============================================================================
# Example 1: Basic Usage with Core API
# ============================================================================
print("\n1. Basic Usage (Core API)")
print("-" * 60)

from intgrml import Boost

# Create simple dataset
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int32)

# Train model
print("Training Boost model (100 trees, depth=6)...")
model = Boost(trees=100, depth=6)
model.fit(X_train, y_train)

# Predict
X_test = np.random.randn(200, 10)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(np.int32)
predictions = model.predict_class(X_test)

accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")

# ============================================================================
# Example 2: scikit-learn Compatible API
# ============================================================================
print("\n2. scikit-learn Compatible API")
print("-" * 60)

from intgrml import IntgrBoostClassifier

# Create dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
print("Training IntgrBoostClassifier...")
clf = IntgrBoostClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.25,
    random_state=42
)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importances
importances = clf.feature_importances_
top_features = np.argsort(importances)[-5:][::-1]
print(f"\nTop 5 Features:")
for i, feat_idx in enumerate(top_features, 1):
    print(f"  {i}. Feature {feat_idx}: {importances[feat_idx]:.3f}")

# ============================================================================
# Example 3: Model Serialization
# ============================================================================
print("\n3. Model Serialization")
print("-" * 60)

with tempfile.NamedTemporaryFile(suffix=".sbf", delete=False) as f:
    model_path = f.name

try:
    # Save model
    print(f"Saving model to {model_path}...")
    clf.save(model_path)

    # Load model
    print("Loading model...")
    clf_loaded = IntgrBoostClassifier()
    clf_loaded.load(model_path)

    # Verify predictions match
    y_pred_loaded = clf_loaded.predict(X_test)
    assert np.array_equal(y_pred, y_pred_loaded), "Predictions don't match!"
    print("✓ Model saved and loaded successfully")
    print(f"✓ Predictions match (verified {len(y_pred)} samples)")

finally:
    if os.path.exists(model_path):
        os.unlink(model_path)

# ============================================================================
# Example 4: Hyperparameter Tuning with GridSearchCV
# ============================================================================
print("\n4. Hyperparameter Tuning (GridSearchCV)")
print("-" * 60)

# Create smaller dataset for faster tuning
X_small, y_small = make_classification(
    n_samples=500,
    n_features=10,
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.25],
}

print("Running grid search...")
clf_gs = IntgrBoostClassifier(random_state=42)
grid = GridSearchCV(
    clf_gs,
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=0
)
grid.fit(X_small, y_small)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.2%}")

# Use best model
best_clf = grid.best_estimator_
print(f"Best model trees: {best_clf.n_estimators}")
print(f"Best model depth: {best_clf.max_depth}")

# ============================================================================
# Example 5: Deterministic Training
# ============================================================================
print("\n5. Deterministic Training")
print("-" * 60)

# Train twice with same seed
clf1 = IntgrBoostClassifier(n_estimators=50, random_state=42)
clf1.fit(X_train, y_train)
pred1 = clf1.predict(X_test)

clf2 = IntgrBoostClassifier(n_estimators=50, random_state=42)
clf2.fit(X_train, y_train)
pred2 = clf2.predict(X_test)

if np.array_equal(pred1, pred2):
    print("✓ Training is deterministic (predictions match)")
else:
    print("✗ Training is NOT deterministic (predictions differ)")

# ============================================================================
# Example 6: Pandas Integration
# ============================================================================
print("\n6. Pandas Integration")
print("-" * 60)

try:
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    y_series = pd.Series(y_train, name="target")

    # Train with DataFrame
    clf_pandas = IntgrBoostClassifier(n_estimators=50, random_state=42)
    clf_pandas.fit(df, y_series)

    # Check feature names preserved
    print(f"✓ Trained with pandas DataFrame")
    print(f"✓ Feature names preserved: {clf_pandas.feature_names_in_[:3]}...")

    # Predict with DataFrame
    df_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    pred_pandas = clf_pandas.predict(df_test)
    print(f"✓ Predictions: {len(pred_pandas)} samples")

except ImportError:
    print("pandas not installed, skipping pandas integration example")

# ============================================================================
print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)

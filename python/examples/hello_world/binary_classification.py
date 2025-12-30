#!/usr/bin/env python3
"""
IntgrML Hello World: Binary Classification

This is a minimal example demonstrating IntgrML's gradient boosting classifier
with a tiny, hard-coded dataset. No external data files or sklearn required.

Prerequisites:
    pip install intgrml

Run with:
    cd python/examples/hello_world
    python binary_classification.py

Expected output:
    - Model training message
    - Predictions on test samples
    - Training accuracy
"""

import numpy as np
from intgrml import IntgrBoostClassifier

def main():
    print("=" * 50)
    print("IntgrML Hello World: Binary Classification")
    print("=" * 50)

    # ========================================
    # Tiny inline dataset (no external files)
    # ========================================
    # Features: [feature_1, feature_2]
    # Label: 0 = class A, 1 = class B
    #
    # Pattern: class B tends to have larger values

    # Generate a small but learnable dataset
    # Class 0: centered around (-2, -2)
    # Class 1: centered around (+2, +2)
    np.random.seed(42)
    n_per_class = 25

    X_class0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2.0, -2.0])
    X_class1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([2.0, 2.0])

    X_train = np.vstack([X_class0, X_class1]).astype(np.float64)
    y_train = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int32)

    # ========================================
    # Create and train the model
    # ========================================
    print("\nTraining IntgrBoostClassifier...")
    print(f"  Dataset: {len(y_train)} samples, {X_train.shape[1]} features")

    clf = IntgrBoostClassifier(
        n_estimators=20,      # Small number of trees for speed
        max_depth=3,          # Shallow trees
        learning_rate=0.25,
        random_state=42       # Deterministic results
    )

    clf.fit(X_train, y_train)
    print("  Training complete!")

    # ========================================
    # Make predictions on training data
    # ========================================
    y_pred = clf.predict(X_train)
    accuracy = (y_pred == y_train).mean()

    print(f"\nTraining Accuracy: {accuracy:.0%}")

    # ========================================
    # Predict on new samples
    # ========================================
    print("\nPredictions on new samples:")
    X_new = np.array([
        [-1.0, -1.0],   # Should be class 0
        [0.0, 0.0],     # Boundary case
        [1.5, 1.5],     # Should be class 1
    ], dtype=np.float64)

    predictions = clf.predict(X_new)

    for i, (features, pred) in enumerate(zip(X_new, predictions)):
        class_name = "Class A" if pred == 0 else "Class B"
        print(f"  Sample {i+1}: {features} -> {class_name} (label={pred})")

    # ========================================
    # Success message
    # ========================================
    print("\n" + "=" * 50)
    print("Hello World complete!")
    print("IntgrML is working correctly.")
    print("=" * 50)


if __name__ == "__main__":
    main()

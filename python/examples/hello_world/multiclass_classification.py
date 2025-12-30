#!/usr/bin/env python3
"""
IntgrML Hello World: Multiclass Classification

This example demonstrates multiclass classification (3 classes) using
IntgrML. Since IntgrML v1.2.1 supports binary classification natively,
we use a One-vs-Rest (OvR) approach with multiple binary classifiers.

Prerequisites:
    pip install intgrml

Run with:
    cd python/examples/hello_world
    python multiclass_classification.py

Note: Full multiclass support in the Python API is coming in v1.3.
This example shows the pattern for 3-class problems.
"""

import numpy as np
from intgrml import IntgrBoostClassifier

def main():
    print("=" * 50)
    print("IntgrML Hello World: Multiclass Classification")
    print("=" * 50)

    # ========================================
    # 3-class dataset with well-separated clusters
    # ========================================
    # Features: [x, y] coordinates
    # Classes: 0=bottom-left, 1=top, 2=bottom-right
    #
    # Visual pattern:
    #         1 1
    #        1   1
    #      0 0   2 2
    #      0 0   2 2

    np.random.seed(42)
    n_per_class = 20

    # Class 0: bottom-left cluster, centered at (-3, -2)
    X_class0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-3.0, -2.0])

    # Class 1: top cluster, centered at (0, 3)
    X_class1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([0.0, 3.0])

    # Class 2: bottom-right cluster, centered at (3, -2)
    X_class2 = np.random.randn(n_per_class, 2) * 0.5 + np.array([3.0, -2.0])

    X_train = np.vstack([X_class0, X_class1, X_class2]).astype(np.float64)
    y_train = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class, dtype=np.int32)

    print(f"\nDataset: {len(y_train)} samples, {X_train.shape[1]} features, 3 classes")

    # ========================================
    # One-vs-Rest (OvR) Multiclass Strategy
    # ========================================
    # Train one binary classifier per class
    # Each classifier learns: "Is this sample class K or not?"

    print("\nTraining One-vs-Rest classifiers...")

    classes = [0, 1, 2]
    classifiers = {}

    for k in classes:
        # Binary labels: 1 if class k, 0 otherwise
        y_binary = (y_train == k).astype(np.int32)

        clf = IntgrBoostClassifier(
            n_estimators=15,
            max_depth=3,
            learning_rate=0.3,
            random_state=42 + k  # Different seed per class for diversity
        )
        clf.fit(X_train, y_binary)
        classifiers[k] = clf
        print(f"  Classifier for class {k}: trained")

    print("Training complete!")

    # ========================================
    # Predict: choose class with highest score
    # ========================================
    def predict_multiclass(X, classifiers):
        """Predict class by taking argmax of OvR scores."""
        scores = np.zeros((len(X), len(classifiers)))
        for k, clf in classifiers.items():
            # Use decision_function (raw logits) for scoring
            scores[:, k] = clf.decision_function(X)
        return np.argmax(scores, axis=1)

    y_pred = predict_multiclass(X_train, classifiers)
    accuracy = (y_pred == y_train).mean()

    print(f"\nTraining Accuracy: {accuracy:.0%}")

    # ========================================
    # Predict on new samples
    # ========================================
    print("\nPredictions on new samples:")

    X_new = np.array([
        [-3.0, -2.0],   # Should be class 0 (bottom-left)
        [0.0, 3.0],     # Should be class 1 (top)
        [3.0, -2.0],    # Should be class 2 (bottom-right)
        [0.0, 0.0],     # Ambiguous center point
    ], dtype=np.float64)

    predictions = predict_multiclass(X_new, classifiers)

    class_names = {0: "bottom-left", 1: "top", 2: "bottom-right"}
    for i, (features, pred) in enumerate(zip(X_new, predictions)):
        print(f"  Sample {i+1}: ({features[0]:+.1f}, {features[1]:+.1f}) -> Class {pred} ({class_names[pred]})")

    # ========================================
    # Success message
    # ========================================
    print("\n" + "=" * 50)
    print("Multiclass Hello World complete!")
    print("3-class OvR classification working correctly.")
    print("=" * 50)


if __name__ == "__main__":
    main()

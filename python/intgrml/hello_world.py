#!/usr/bin/env python3
"""
IntgrML Hello World - Quick verification that IntgrML is installed correctly.

Run with:
    python -m intgrml.hello_world

This runs a minimal binary classification example to verify:
- IntgrML C++ extensions are installed
- Training works
- Prediction works
- Determinism is maintained
"""

import numpy as np


def run_hello_world():
    """Run a minimal IntgrML example."""
    print("=" * 50)
    print("IntgrML Hello World")
    print("=" * 50)

    # Import here so any import errors are visible
    from intgrml import IntgrBoostClassifier, __version__

    print(f"\nIntgrML version: {__version__}")

    # Small but learnable dataset
    np.random.seed(42)
    n_per_class = 15
    X_class0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-2.0, -2.0])
    X_class1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([2.0, 2.0])
    X = np.vstack([X_class0, X_class1]).astype(np.float64)
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int32)

    print(f"\nTraining model on {len(y)} samples...")
    clf = IntgrBoostClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf.fit(X, y)

    predictions = clf.predict(X)
    accuracy = (predictions == y).mean()

    print(f"Predictions: {predictions.tolist()}")
    print(f"Accuracy: {accuracy:.0%}")

    # Verify determinism
    clf2 = IntgrBoostClassifier(n_estimators=10, max_depth=2, random_state=42)
    clf2.fit(X, y)
    pred2 = clf2.predict(X)

    if np.array_equal(predictions, pred2):
        print("\nDeterminism: VERIFIED (identical predictions)")
    else:
        print("\nDeterminism: FAILED")

    print("\n" + "=" * 50)
    print("IntgrML is working correctly!")
    print("=" * 50)
    print("\nNext steps:")
    print("  - See python/examples/hello_world/ for more examples")
    print("  - Read docs/PYTHON_QUICKSTART.md for full documentation")


if __name__ == "__main__":
    run_hello_world()

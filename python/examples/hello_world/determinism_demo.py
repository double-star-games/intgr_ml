#!/usr/bin/env python3
"""
IntgrML Hello World: Determinism Demo

This example demonstrates IntgrML's key differentiator: bit-exact determinism.
When trained with the same data and random_state, IntgrML produces identical
models with identical predictions every time.

Why determinism matters:
- Reproducible experiments and debugging
- Regulatory compliance (finance, healthcare)
- Auditable ML systems
- Bit-exact parity between desktop training and embedded deployment

Run with:
    python python/examples/hello_world/determinism_demo.py

Expected output:
    - Two models trained independently
    - Verification that predictions are bit-identical
    - Optional: model file comparison (if serialization is available)
"""

import numpy as np
from intgrml import IntgrBoostClassifier
import tempfile
import os

def main():
    print("=" * 50)
    print("IntgrML Hello World: Determinism Demo")
    print("=" * 50)

    # ========================================
    # Create a small dataset
    # ========================================
    print("\nCreating dataset...")

    X_train = np.array([
        [-2.0, -1.5], [-1.5, -2.0], [-1.0, -0.5], [-0.5, -1.0],
        [-1.2, -1.8], [-0.8, -0.3], [-1.6, -0.9], [-0.4, -1.4],
        [1.0, 1.5], [1.5, 1.0], [2.0, 2.5], [2.5, 2.0],
        [1.2, 1.8], [0.8, 1.3], [1.6, 0.9], [0.4, 1.4],
    ], dtype=np.float64)

    y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

    X_test = np.array([
        [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0],
        [-0.5, 0.5], [0.5, -0.5], [1.5, 1.5],
    ], dtype=np.float64)

    print(f"  Training set: {len(y_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # ========================================
    # Train Model 1
    # ========================================
    print("\n" + "-" * 50)
    print("Training Model 1...")

    clf1 = IntgrBoostClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.25,
        random_state=42  # <-- Key: same random state
    )
    clf1.fit(X_train, y_train)
    pred1 = clf1.predict(X_test)
    logits1 = clf1.decision_function(X_test)

    print("  Model 1 trained.")
    print(f"  Predictions: {pred1.tolist()}")

    # ========================================
    # Train Model 2 (independent training)
    # ========================================
    print("\nTraining Model 2 (independent run, same settings)...")

    clf2 = IntgrBoostClassifier(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.25,
        random_state=42  # <-- Same random state
    )
    clf2.fit(X_train, y_train)
    pred2 = clf2.predict(X_test)
    logits2 = clf2.decision_function(X_test)

    print("  Model 2 trained.")
    print(f"  Predictions: {pred2.tolist()}")

    # ========================================
    # Verify Determinism
    # ========================================
    print("\n" + "-" * 50)
    print("DETERMINISM VERIFICATION")
    print("-" * 50)

    # Check predictions
    predictions_match = np.array_equal(pred1, pred2)
    print(f"\nPredictions identical: {'YES' if predictions_match else 'NO'}")

    # Check raw logits (more sensitive test)
    logits_match = np.array_equal(logits1, logits2)
    print(f"Raw logits identical:  {'YES' if logits_match else 'NO'}")

    if predictions_match and logits_match:
        print("\nBIT-EXACT DETERMINISM VERIFIED")
        print("Both models produce identical predictions and logits.")
    else:
        print("\nWARNING: Models differ! This should not happen.")
        print("Prediction diff:", pred1 - pred2)
        print("Logits diff:", logits1 - logits2)

    # ========================================
    # Optional: Model file comparison
    # ========================================
    print("\n" + "-" * 50)
    print("MODEL SERIALIZATION CHECK")
    print("-" * 50)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "model1.sbf")
            path2 = os.path.join(tmpdir, "model2.sbf")

            clf1.save(path1)
            clf2.save(path2)

            # Compare file sizes
            size1 = os.path.getsize(path1)
            size2 = os.path.getsize(path2)

            print(f"\nModel 1 file size: {size1} bytes")
            print(f"Model 2 file size: {size2} bytes")
            print(f"Sizes match: {'YES' if size1 == size2 else 'NO'}")

            # Compare file contents
            with open(path1, "rb") as f1, open(path2, "rb") as f2:
                bytes1 = f1.read()
                bytes2 = f2.read()

            files_identical = (bytes1 == bytes2)
            print(f"File contents identical: {'YES' if files_identical else 'NO'}")

            if files_identical:
                print("\nSERIALIZED MODELS ARE BIT-IDENTICAL")
            else:
                print("\nNote: Model files differ (may include timestamps)")

    except Exception as e:
        print(f"\nModel serialization check skipped: {e}")

    # ========================================
    # Why this matters
    # ========================================
    print("\n" + "=" * 50)
    print("WHY DETERMINISM MATTERS")
    print("=" * 50)
    print("""
IntgrML's bit-exact determinism enables:

1. REPRODUCIBILITY
   - Same training data + seed = identical model, every time
   - Debug issues by reproducing exact conditions

2. REGULATORY COMPLIANCE
   - Auditable ML systems for finance/healthcare
   - Prove model behavior is consistent

3. EMBEDDED DEPLOYMENT
   - Train on desktop, deploy to MCU
   - Bit-identical predictions on both platforms

4. TESTING
   - Unit tests can verify exact outputs
   - CI/CD can catch any behavioral changes

Most float-based ML libraries cannot guarantee this.
IntgrML's integer-only arithmetic makes it possible.
""")

    print("=" * 50)
    print("Determinism demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

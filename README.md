# IntgrML

**Integer-Only Machine Learning for Edge Devices**

IntgrML is a production-ready machine learning library that performs training and inference using pure integer arithmetic. Designed for deterministic, reproducible results across platforms, IntgrML enables deployment on embedded systems, microcontrollers, and environments where floating-point hardware is unavailable or undesirable.

---

## Key Features

- **Integer-only arithmetic**: All training and inference uses fixed-point math (Q8.8 values, Q16.16 accumulator) with no floating-point operations at runtime
- **Bit-exact determinism**: Same inputs + same seed = identical results across runs, platforms, and compilers
- **Edge/embedded ready**: Models run identically on desktop, MCU, and WebAssembly targets
- **Efficient memory layout**: Compact model formats optimized for constrained environments
- **Multiple algorithms**: Gradient boosting (IntgrBoost), random forests (IntgrForest), linear models (IntgrLinear), Naive Bayes (IntgrBayes), PCA (IntgrReduce)
- **Multiclass support**: One-vs-Rest classification for K-class problems
- **65,535 feature limit**: Intentional design constraint for u16 indexing efficiency; use IntgrReduce for high-dimensional data
- **Three interfaces**: C++ API, command-line tool (`intgrmlc`), and Python package (`intgrml`)

---

## Downloads

Prebuilt Python wheels for Linux and Windows are available from the [GitHub Releases](https://github.com/pmeade/intgr_ml/releases) page.

| Platform | Wheel Example |
|----------|---------------|
| Linux x86_64 | `intgrml-1.0.0-cp310-cp310-linux_x86_64.whl` |
| Windows x64 | `intgrml-1.0.0-cp313-cp313-win_amd64.whl` |

Download the wheel for your platform and Python version, then install with pip (see Installation below).

---

## Supported Platforms

| Platform | Status | Wheels | Notes |
|----------|--------|--------|-------|
| **Linux x86_64** | ✅ Tier 1 | ✅ Available | SIMD enabled where available |
| **Windows x64** | ✅ Tier 1 | ✅ Available | SIMD disabled for v1 |
| macOS x86_64 | 🟡 CI-tested | 🟡 Planned | C++ builds pass in CI; wheel pending |
| macOS ARM64 | 🟡 CI-tested | 🟡 Planned | Apple Silicon; C++ builds pass in CI |
| Linux ARM64 | 🟡 Planned | 🟡 Planned | Cross-compiled in CI; native testing pending |

**Tier 1** platforms have validated Python wheels available in [GitHub Releases](https://github.com/pmeade/intgr_ml/releases).

**CI-tested** platforms have C++ builds/tests passing in CI but Python wheels are not yet available.

**Planned** platforms have build infrastructure but require additional validation before wheel release.

---

## Installation

### Recommended: Install from Wheel (Most Users)

Download the appropriate wheel for your platform from [GitHub Releases](https://github.com/pmeade/intgr_ml/releases), then install:

```bash
# Download the wheel for your platform and Python version, then:
pip install intgrml-1.0.0-cp310-cp310-linux_x86_64.whl   # Linux example
# or
pip install intgrml-1.0.0-cp313-cp313-win_amd64.whl      # Windows example
```

Verify the installation:

```python
import intgrml
print(intgrml.__version__)
```

### C++ Headers Only

For C++ integration, clone this repository to access the public headers in `include/`. Link against your own build of the engine or use the Python bindings for training.

### Build from Source (Advanced)

Building from source requires the full development repository. Contact us for access if you need custom builds.

**Prerequisites for source builds:**
- CMake 3.18+
- C++20 compiler (GCC 11+, Clang 15+, MSVC 2022+)
- OpenSSL development libraries
- Python 3.10+ with NumPy and pybind11

---

## Quickstart

### Python

```python
import numpy as np
from intgrml import IntgrBoostClassifier

# Create sample data
X_train = np.random.randn(1000, 20)
y_train = (X_train[:, 0] > 0).astype(np.int32)

# Train a model
clf = IntgrBoostClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_train)
accuracy = (predictions == y_train).mean()
print(f"Training accuracy: {accuracy:.2%}")

# Save and load models
clf.save("model.sbf")
```

### CLI

```bash
# Show available commands
intgrmlc --help

# Train a gradient boosting model
intgrmlc boost train --data train.csv --out model.sbf --trees 100 --depth 6

# Make predictions
intgrmlc boost predict --model model.sbf --data test.csv --out predictions.csv

# Verify deterministic training (trains 3x, compares hashes)
intgrmlc boost determinism --data train.csv --out model.sbf --runs 3
```

---

## Documentation

- [Product Overview](PRODUCT_V1.md) — What IntgrML is and isn't
- [Python API Reference](docs/PYTHON_API.md) — Complete Python documentation
- [CLI Reference](docs/CLI_REFERENCE.md) — Command-line tool documentation
- [Python Quickstart](docs/PYTHON_QUICKSTART.md) — Getting started with Python
- [Examples](docs/EXAMPLES.md) — Usage patterns and recipes

---

## Licensing & Commercial Use

IntgrML is licensed under a **Commercial & Community License**.

### Free Community Tier
- Annual Revenue ≤ $200,000
- Total Funding ≤ $2,000,000
- Includes personal, research, and early-stage commercial use

### Commercial Tiers
- **Tier 1**: Annual Revenue ≤ $10,000,000
- **Tier 2**: Annual Revenue ≤ $50,000,000
- **Enterprise/OEM**: Custom agreements for larger deployments

### Downloads vs. License Purchase

**Downloads are free and public.** Prebuilt wheels are available directly from [GitHub Releases](https://github.com/pmeade/intgr_ml/releases) without authentication or payment.

**Licensing is based on usage.** Compliance with the license terms is determined by your organization's revenue, funding, and use case—not by how you obtained the software.

**FastSpring is for license purchases only.** When your organization exceeds the Free Community thresholds, purchase a commercial license via [FastSpring](https://doublestar.net/licensing). FastSpring does not host downloads; it processes license payments.

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full terms and [LICENSING_FAQ.md](LICENSING_FAQ.md) for plain-English explanations.

---

## Contact

- **Sales inquiries**: sales@doublestar.net
- **General questions**: hello@doublestar.net
- **Technical support**: support@doublestar.net
- **Website**: https://doublestar.net
- **Issues**: https://github.com/pmeade/intgr_ml/issues

---

*IntgrML: Integer-only ML that runs everywhere, identically.*

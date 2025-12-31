# IntgrML

**Version 1.2.7** — Integer-Only Machine Learning for Edge Devices

IntgrML is a production-ready machine learning library that performs training and inference using pure integer arithmetic. Designed for deterministic, reproducible results across platforms, IntgrML enables deployment on embedded systems, microcontrollers, and environments where floating-point hardware is unavailable or undesirable.

---

## About This Repository

This repository contains the **SDK distribution** of IntgrML:

- **Python wrappers** (`python/intgrml/`) — sklearn-compatible API
- **Public C++ headers** (`include/`) — for embedding in C++ projects
- **Documentation** (`docs/`) — API reference, quickstart, examples
- **Example scripts** (`python/examples/`) — usage demonstrations

The **core C++ engine and compiled extensions are not stored here**. IntgrML uses a binary distribution model — the compiled `_core` module is distributed via wheels.

**To use IntgrML**, install via pip or download a wheel from GitHub Releases:

```bash
pip install intgrml
```

This is intentional: IntgrML's core implementation is closed-source, but the SDK, headers, and documentation are provided here for reference and integration.

---

## Key Features

- **Integer-only arithmetic**: All training and inference uses fixed-point math (Q8.8 values, Q16.16 accumulator) with no floating-point operations at runtime. (Note: Some initialization steps like building lookup tables may use floats internally, but on-device model evaluation is strictly integer-only.)
- **Bit-exact determinism**: Same inputs + same seed = identical results across runs, platforms, and compilers
- **Edge/embedded ready**: Models run identically on desktop, MCU, and WebAssembly targets
- **Efficient memory layout**: Compact model formats optimized for constrained environments
- **Multiple algorithms**: Gradient boosting (IntgrBoost), random forests (IntgrForest), linear models (IntgrLinear), Naive Bayes (IntgrBayes), PCA (IntgrReduce)
- **Multiclass support**: One-vs-Rest classification for K-class problems
- **65,535 feature limit**: Intentional design constraint for u16 indexing efficiency; use IntgrReduce for high-dimensional data
- **Three interfaces**: C++ API, command-line tool (`intgrmlc`), and Python package (`intgrml`)

---

## Downloads

All release artifacts are available from [GitHub Releases](https://github.com/double-star-games/intgr_ml/releases).

### What's in a Release

Each GitHub Release includes:

**Python Wheels**
| Platform | Wheel |
|----------|-------|
| Linux x86_64 | `intgrml-X.Y.Z-cpXX-cpXX-manylinux_2_28_x86_64.whl` |
| Linux ARM64 | `intgrml-X.Y.Z-cpXX-cpXX-manylinux_2_28_aarch64.whl` |
| Windows x64 | `intgrml-X.Y.Z-cpXX-cpXX-win_amd64.whl` |

Linux wheels are built as `manylinux_2_28` for broad distro compatibility (Ubuntu 22.04+, etc.).

**CLI Binaries** (in SDK archives)
| Platform | Binary |
|----------|--------|
| Linux x86_64 | `intgrmlc` |
| Linux ARM64 | `intgrmlc` |
| Windows x64 | `intgrmlc.exe` |
| macOS ARM64 | `intgrmlc` |
| macOS x86_64 | `intgrmlc` |

**C++ SDK Archives**
| Platform | Archive |
|----------|---------|
| Linux x86_64 | `intgrml-sdk-linux-x86_64.tar.gz` |
| Linux ARM64 | `intgrml-sdk-linux-arm64.tar.gz` |
| Windows x64 | `intgrml-sdk-windows-x64.zip` |
| macOS ARM64 | `intgrml-sdk-macos-arm64.tar.gz` |
| macOS x86_64 | `intgrml-sdk-macos-x86_64.tar.gz` |

Each SDK archive contains:
- `lib/` — Static library (`libintgr_ml.a` or `intgr_ml.lib`)
- `bin/` — CLI tool (`intgrmlc` or `intgrmlc.exe`)
- `include/` — Header files (also available in this repository)

### Headers

Public C++ headers are available in this repository under `include/`. These are the same headers bundled in the SDK archives.

---

## Supported Platforms

IntgrML prioritizes **determinism, correctness, and honesty** in platform claims.
Every platform listed below is validated through real builds and execution testing.

| Platform | Status | Wheel Support | Notes |
|----------|--------|---------------|-------|
| **Linux x86_64** | Tier 1 | Yes | Built as manylinux_2_28, deterministic verified |
| **Windows x64** | Tier 1 | Yes | Native validated, deterministic verified |
| **Linux ARM64 (aarch64)** | Tier 1 | Yes | Built as manylinux_2_28, container-executed + deterministic verified |
| **macOS (Intel)** | Planned | No | CI builds pass, wheels pending |
| **macOS (Apple Silicon)** | Planned | No | CI builds pass, wheels pending |

### What "Tier 1" Means

Tier 1 platforms meet all of the following:
- Builds cleanly
- Imports successfully
- Runs core functionality
- Passes determinism checks
- Has a downloadable wheel

### Linux ARM64 Status

Linux ARM64 (aarch64) is **fully supported** via:
- Official `manylinux_2_28_aarch64` toolchain
- Wheel executes successfully under QEMU
- Deterministic equality **verified**

### Determinism Guarantee

All Tier 1 platforms preserve bit-exact determinism:
- Identical logits
- Identical predictions
- Identical file outputs
- Verified hash parity

Determinism is a core design value of IntgrML.

---

## Installation

### Recommended: Install from Wheel (Most Users)

Download the appropriate wheel for your platform from [GitHub Releases](https://github.com/double-star-games/intgr_ml/releases), then install:

```bash
# Download the wheel for your platform and Python version, then:
pip install intgrml-X.Y.Z-cpXX-cpXX-manylinux_2_28_x86_64.whl   # Linux x86_64
pip install intgrml-X.Y.Z-cpXX-cpXX-win_amd64.whl               # Windows x64
pip install intgrml-X.Y.Z-cpXX-cpXX-manylinux_2_28_aarch64.whl  # Linux ARM64
```

### Linux Compatibility Note

Both Linux wheels (x86_64 and ARM64) are built as `manylinux_2_28`, ensuring compatibility with:
- Ubuntu 22.04+
- Debian 11+
- RHEL/CentOS 8+
- Most modern Linux distributions

This avoids GLIBCXX version errors that can occur with wheels built on newer host systems.

The Linux CLI binary (`intgrmlc`) and C++ SDK are also built with the manylinux toolchain for maximum compatibility. The CLI bundles OpenSSL 1.1 shared libraries to avoid `libcrypto.so.1.1` dependency issues on systems with OpenSSL 3.x.

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

> **Note:** The examples below assume you have installed IntgrML via `pip install intgrml` or from a wheel. This repository does not contain the compiled C++ core — it's distributed as prebuilt wheels.

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

The `intgrmlc` command-line tool is included in the SDK archives on [GitHub Releases](https://github.com/double-star-games/intgr_ml/releases). Download the SDK for your platform, extract, and add `bin/` to your PATH.

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

- [Python API Reference](docs/PYTHON_API.md) — Complete Python documentation
- [CLI Reference](docs/CLI_REFERENCE.md) — Command-line tool documentation
- [Python Quickstart](docs/PYTHON_QUICKSTART.md) — Getting started with Python
- [Examples](docs/EXAMPLES.md) — Usage patterns and recipes

---

## Licensing & Commercial Use

IntgrML is licensed under a **Commercial & Community License**.

### Free Community Tier
- Annual Revenue <= $200,000
- Total Funding <= $2,000,000
- Includes personal, research, and early-stage commercial use

### Commercial Tiers
- **Tier 1**: Annual Revenue <= $10,000,000
- **Tier 2**: Annual Revenue <= $50,000,000
- **Enterprise/OEM**: Custom agreements for larger deployments

### Downloads vs. License Purchase

**Downloads are free and public.** Prebuilt wheels are available directly from [GitHub Releases](https://github.com/double-star-games/intgr_ml/releases) without authentication or payment.

**Licensing is based on usage.** Compliance with the license terms is determined by your organization's revenue, funding, and use case—not by how you obtained the software.

**FastSpring is for license purchases only.** When your organization exceeds the Free Community thresholds, purchase a commercial license via [FastSpring](https://doublestar.net/licensing). FastSpring does not host downloads; it processes license payments.

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full terms and [LICENSING_FAQ.md](LICENSING_FAQ.md) for plain-English explanations.

---

## Contact

- **Sales inquiries**: sales@doublestar.net
- **General questions**: hello@doublestar.net
- **Technical support**: support@doublestar.net
- **Website**: https://doublestar.net
- **Issues**: https://github.com/double-star-games/intgr_ml/issues

---

*IntgrML ships verified wheels for Linux x86_64, Windows x64, and Linux ARM64 — with deterministic guarantees.*

*Integer-only ML that runs everywhere, identically.*

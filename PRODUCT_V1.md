# intgr_ml — v1.0 Product Definition (Canonical)

_Last updated: 2025-12-29_

This document defines **exactly what intgr_ml v1.0 is** and **what it is not**.  
It is the canonical reference for product scope, supported environments, licensing posture, and repo topology.

---

## 1. Product Overview

**intgr_ml** is an **integer-only machine learning library** focused on:

- Deterministic, integer-based inference and training
- Small, efficient models suitable for **edge and embedded** use
- A clean C++ API, a CLI tool, and first-class **Python support**

intgr_ml v1.0 is a **production-capable runtime and tooling** release, not a research demo and not an open-source framework.

---

## 2. Deliverables in v1.0

### 2.1 Core Engine

**Form:** Compiled libraries + public headers

- C++ core implementing integer-only ML primitives and runtime
- Public C++ API surface (headers) for:
  - Model construction / loading
  - Inference
  - (If supported) training / fine-tuning
- Internal implementation remains private and is **not** shipped as source

**Linking:**

- Primary: static libraries (`.a` / `.lib`)  
- (Optional: shared libs if supported by the build system)

The engine is the heart of the product and must be stable, deterministic, and test-backed.

---

### 2.2 CLI Tool

**Form:** Binaries (plus thin source wrapper if needed)

- Command-line interface for:
  - Running inference
  - Inspecting / converting models
  - Simple benchmarking / diagnostics
- May be shipped:
  - As prebuilt binaries for common platforms
  - With CLI **orchestration source** that calls into the engine (no internal algorithms exposed)

The CLI is a **thin layer** over the engine, safe to share as source if convenient for build/distribution.

---

### 2.3 Python Support

**Form:** Python package (`intgr_ml`)

- Python bindings over the core engine
- Clean installation experience (e.g. `pip install intgr_ml` for supported platforms)
- Pythonic API for:
  - Loading models
  - Running inference
  - (If supported) integrating into training/inference workflows

Python users are a primary audience for v1.0.

---

### 2.4 Documentation & Metadata

- `PRODUCT_V1.md` — this document (canonical product definition)
- `RUNBOOK.md` — maintained by Quartermaster, describes **exact build & release steps**
- `COMMERCIAL_LICENSE.md` — full license text
- `LICENSING_FAQ.md` — plain-English licensing FAQ
- `README.md` (public repo) — concise overview, install, quickstart
- Minimal but complete API/usage docs for:
  - C++ API
  - Python API
  - CLI usage

---

## 3. Non-Goals for v1.0

The following are **explicitly out of scope** for the v1.0 release:

- GPU acceleration
- Distributed training
- Advanced model architectures beyond the current supported set
- Full model zoo or high-level training workflows
- Public exposure of internal engine source code
- Stable plugin API for third-party extensions

v1.0 is a **focused, integer-only ML runtime and tooling** release, not a general-purpose deep learning platform.

---

## 4. Supported Environments (Initial Target)

These are the **officially supported** v1.0 environments (subject to adjustment as we refine build/tests):

### 4.1 Host Platforms

- **Linux x86_64** — Tier 1 Supported Platform
  - Officially supported in v1.0
  - Builds must pass
  - Core functionality, CLI, and Python bindings must work
  - Toolchain: modern GCC/Clang (document exact versions in `RUNBOOK.md`)
  - Prebuilt binaries and/or Python wheels may be provided where feasible

- **Windows x64** — Tier 1 Supported Platform
  - Officially supported in v1.0
  - Builds must pass under MSVC toolchain
  - Core functionality, CLI, and Python bindings must work
  - Determinism guarantees must hold (bit-exact reproducibility)
  - Toolchain: MSVC (document exact version in `RUNBOOK.md`)
  - Prebuilt binaries and/or Python wheels are desirable but not guaranteed for v1.0
  - No promise of deep Windows ecosystem integration (NuGet, UWP, etc.) in v1.0

**Note:** macOS and other platforms are NOT Tier 1 unless explicitly promoted later.

### 4.2 Languages / APIs

- **C++17 or later** for the core engine and public API
- **Python 3.9+** for Python bindings (document exact minor versions supported)

### 4.3 Embedded / Edge Targets

- intgr_ml is designed for embedded/edge use where integer-only inference is valuable.
- v1.0 **does not promise** direct toolchain support for every MCU/SoC.
- Embedded support for v1.0 is:
  - "Manually integrable" via the C++/static library
  - Documented via example(s) where feasible
  - Full embedded porting guides may come later

### 4.4 Feature Count Limit

intgr_ml v1.0 supports datasets with **up to 65,535 features** (the maximum value of a 16-bit unsigned integer).

- Training or loading models with more than 65,535 features will produce a clear error message
- For high-dimensional datasets, consider:
  - **IntgrReduce (PCA)** for dimensionality reduction before training
  - Feature selection/engineering in your preprocessing pipeline
- This limit is enforced consistently across C++, CLI, and Python APIs

---

## 5. Licensing & Pricing (High-Level Summary)

intgr_ml is **not open source**.

It is licensed under a **Commercial & Community License** with these principles:

- **Free Community Tier**
  - Annual Revenue ≤ \$200,000
  - Total Funding ≤ \$2,000,000
  - Allows personal, research, and early-stage commercial use, including limited deployment

- **Commercial Tier 1**
  - Annual Revenue ≤ \$10,000,000
  - Allows full commercial deployment in software products and Devices

- **Commercial Tier 2**
  - Annual Revenue ≤ \$50,000,000
  - Allows large-scale deployment and enterprise use

- **OEM / Enterprise**
  - Annual Revenue > \$50,000,000, or high-volume Devices, or regulated/safety-critical use
  - Requires a separate negotiated agreement

**Canonical reference:**  
- `COMMERCIAL_LICENSE.md` — full legal text  
- `LICENSING_FAQ.md` — developer-friendly explanation

---

## 6. Repo Topology & Publication Model

There are **two GitHub repositories**:

1. `pmeade/intgr_ml_dev` (local: `~/dev/intgr_ml_dev`)
   - _Private / internal development repo_
   - Contains:
     - Full engine implementation
     - All internal tools, experiments, and support code
     - Alexandria, Quartermaster, Kepler cards and scripts
   - Treated as the **source of truth** for implementation and development history.

2. `pmeade/intgr_ml` (local: `~/prod/intgr_ml`)
   - _Public / product-facing repo_
   - Contains only:
     - Public headers and binaries (or build artifacts instructions)
     - Public Python bindings
     - CLI tooling (binaries and/or safe source)
     - Documentation: README, PRODUCT_V1.md copy, LICENSE, FAQ, etc.
   - No internal implementation or sensitive code.

### 6.1 Publication Process (High-Level)

- All development happens in `intgr_ml_dev`
- A **publish pipeline** (scripted) is responsible for:
  - Building release artifacts
  - Copying public headers, bindings, CLI code (if applicable), and docs into `intgr_ml`
  - Ensuring that only intended files are exposed
- The publication pipeline is documented in `RUNBOOK.md` and should be reproducible.

### 6.2 Distribution & Downloads

**Model B: Headers + Prebuilt Wheels from GitHub Releases**

IntgrML uses a distribution model where:

- **Prebuilt Python wheels** (Linux x86_64, Windows x64) are attached to GitHub Releases
- **C++ headers** are available directly in this repository (`include/`)
- **Pure Python wrappers** are available in `python/intgrml/`
- **Source code** for the core engine is NOT distributed (proprietary)

**Where to get IntgrML:**

| Artifact | Location |
|----------|----------|
| Python wheels | [GitHub Releases](https://github.com/pmeade/intgr_ml/releases) |
| C++ headers | `include/` directory in this repo |
| Documentation | This repo + [docs/](docs/) |

**Licensing vs. Distribution:**

- **Downloads are ungated.** Users can download wheels from GitHub Releases without authentication or payment.
- **Licensing is based on usage.** The [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) defines who must pay and when.
- **FastSpring handles license purchases only.** It does NOT host downloads or gate access to binaries.

This model allows:
- Easy evaluation and adoption (download and try immediately)
- Clear legal terms (license defines compliance, not download access)
- Simple distribution (GitHub handles hosting and versioning)

---

## 7. Versioning & Release Policy

- intgr_ml uses **Semantic Versioning**:
  - **MAJOR.MINOR.PATCH**
  - v1.0.0 is the first public production release.
- For v1.x:
  - **PATCH**: bugfixes, no API/ABI break
  - **MINOR**: new features, backwards-compatible API additions
  - **MAJOR**: breaking API changes or significant architecture shifts

Each public release:
- Is tagged in git (`v1.0.0`, `v1.0.1`, etc.)
- Has a corresponding GitHub Release with:
  - Release notes
  - Binaries / wheels (where applicable)
  - Reference to the exact commit

---

## 8. Quartermaster & Documentation as Code

Quartermaster maintains a canonical `RUNBOOK.md` in `intgr_ml_dev` that MUST always be correct for:

- Full clean build from scratch
- Test execution
- Release artifact generation
- Publication steps from `intgr_ml_dev` → `intgr_ml`

Changes to build or release processes **must** be reflected in `RUNBOOK.md` as part of the same change.

Alexandria and Kepler are used to:

- Map and document “what is” (Alexandria)
- Understand project history and major epochs (Kepler)

They support, but do not replace, the canonical product definition and runbook.

---

_End of v1.0 Product Definition_

---

_Windows and Linux Tier-1 platforms validated. Distribution via GitHub Releases; licensing via FastSpring._

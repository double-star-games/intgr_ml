# IntgrML Python Package

This directory contains the IntgrML Python package.

## Installation

```bash
pip install intgrml
```

## Structure

- `intgrml/` - Main package directory
  - `__init__.py` - Package initialization
  - `sklearn_api.py` - scikit-learn compatible API
  - `_core.so` - Compiled C++ bindings (platform-specific)

## Usage

```python
from intgrml import IntgrBoostClassifier

clf = IntgrBoostClassifier(n_estimators=100, max_depth=6)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

See [PYTHON_API.md](../docs/PYTHON_API.md) for complete documentation.

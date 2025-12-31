"""
IntgrML: Integer-only machine learning for edge devices

IntgrML provides gradient boosting and random forest implementations using
pure integer arithmetic, enabling deployment on embedded systems without
floating-point hardware.

Quick Start
-----------
>>> from intgrml import Boost
>>> model = Boost(trees=100, depth=6)
>>> model.fit(X_train, y_train)
>>> predictions = model.predict_class(X_test)  # Returns 0/1 labels

For scikit-learn compatibility:
>>> from intgrml import IntgrBoostClassifier
>>> model = IntgrBoostClassifier(n_estimators=100, max_depth=6)
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
"""

__version__ = "1.2.6"
__author__ = "Patrick Meade"

# Import core C++ bindings
try:
    from intgrml._core import Boost as _CoreBoost
    from intgrml._core import Forest as _CoreForest
except ImportError as e:
    raise ImportError(
        "IntgrML C++ extensions not found. "
        "Please install with: pip install intgrml"
    ) from e

# Import sklearn-compatible API
from intgrml.sklearn_api import (
    IntgrBoostClassifier,
    IntgrBoostRegressor,
    IntgrForestClassifier,
)

# Simple aliases
Boost = _CoreBoost
Forest = _CoreForest

__all__ = [
    # Core classes
    "Boost",
    "Forest",
    # sklearn-compatible classes
    "IntgrBoostClassifier",
    "IntgrBoostRegressor",
    "IntgrForestClassifier",
    # Version
    "__version__",
]

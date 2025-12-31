"""
scikit-learn compatible API for IntgrML

Implements the standard sklearn API to enable ecosystem compatibility with
GridSearchCV, Pipeline, and other sklearn utilities.

This module uses sklearn's BaseEstimator and ClassifierMixin base classes
(BSD license) to provide a familiar interface for users of XGBoost, LightGBM,
and other sklearn-compatible libraries.

IntgrML provides unique advantages for edge deployment and deterministic systems
while maintaining API compatibility with the Python ML ecosystem.
"""

import numpy as np
from typing import Optional, Union
import numbers

# =============================================================================
# Constants
# =============================================================================

# Maximum number of features supported by IntgrML (uint16 feature indices)
MAX_FEATURES = 65535

# Default temperature for logit-to-probability conversion
# IntgrML uses Q8.8 fixed-point logits; dividing by 100 produces well-calibrated
# probabilities for typical gradient boosting outputs.
DEFAULT_TEMPERATURE = 100.0


# =============================================================================
# Parameter Validation Helpers
# =============================================================================

def _validate_n_estimators(n_estimators, name="n_estimators"):
    """Validate n_estimators parameter.

    Must be a positive integer.
    """
    if not isinstance(n_estimators, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(n_estimators).__name__}")
    if n_estimators < 1:
        raise ValueError(f"{name} must be >= 1, got {n_estimators}")
    if n_estimators > 100000:
        raise ValueError(f"{name}={n_estimators} is unreasonably large (max 100,000)")
    return int(n_estimators)


def _validate_max_depth(max_depth, name="max_depth"):
    """Validate max_depth parameter.

    Must be an integer between 1 and 15 (IntgrML tree depth limit).
    """
    if not isinstance(max_depth, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(max_depth).__name__}")
    if not 1 <= max_depth <= 15:
        raise ValueError(f"{name} must be between 1 and 15, got {max_depth}")
    return int(max_depth)


def _validate_learning_rate(learning_rate, name="learning_rate"):
    """Validate learning_rate parameter.

    Must be a positive float, typically in (0, 1].
    """
    if not isinstance(learning_rate, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(learning_rate).__name__}")
    if learning_rate <= 0:
        raise ValueError(f"{name} must be > 0, got {learning_rate}")
    if learning_rate > 10:
        raise ValueError(f"{name}={learning_rate} is unusually large (typical range: 0.01-1.0)")
    return float(learning_rate)


def _validate_min_samples_leaf(min_samples_leaf, name="min_samples_leaf"):
    """Validate min_samples_leaf parameter.

    Must be a positive integer.
    """
    if not isinstance(min_samples_leaf, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(min_samples_leaf).__name__}")
    if min_samples_leaf < 1:
        raise ValueError(f"{name} must be >= 1, got {min_samples_leaf}")
    return int(min_samples_leaf)


def _validate_n_bins(n_bins, name="n_bins"):
    """Validate n_bins parameter.

    Must be an integer between 2 and 256.
    """
    if not isinstance(n_bins, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(n_bins).__name__}")
    if not 2 <= n_bins <= 256:
        raise ValueError(f"{name} must be between 2 and 256, got {n_bins}")
    return int(n_bins)


def _validate_clip_bounds(clip_min, clip_max):
    """Validate clip_min and clip_max parameters.

    Both must be numeric, and clip_min must be strictly less than clip_max.
    """
    if not isinstance(clip_min, numbers.Real):
        raise TypeError(f"clip_min must be a number, got {type(clip_min).__name__}")
    if not isinstance(clip_max, numbers.Real):
        raise TypeError(f"clip_max must be a number, got {type(clip_max).__name__}")
    if clip_min >= clip_max:
        raise ValueError(f"clip_min must be < clip_max, got clip_min={clip_min}, clip_max={clip_max}")
    return float(clip_min), float(clip_max)


def _validate_subsample(subsample, name="subsample"):
    """Validate subsample parameter.

    Must be a float in (0, 1].
    """
    if not isinstance(subsample, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(subsample).__name__}")
    if not 0 < subsample <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {subsample}")
    return float(subsample)


def _validate_temperature(temperature, name="temperature"):
    """Validate temperature parameter for logit scaling.

    Must be a positive float. Higher values produce more uniform probabilities,
    lower values produce more confident (peaked) probabilities.
    """
    if not isinstance(temperature, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(temperature).__name__}")
    if temperature <= 0:
        raise ValueError(f"{name} must be > 0, got {temperature}")
    return float(temperature)


def _validate_random_state(random_state, name="random_state"):
    """Validate random_state parameter.

    Accepts None (uses default seed), integers, or numpy RandomState/Generator.
    Returns an integer seed for the native code.
    """
    if random_state is None:
        return None  # Will use default in native code
    if isinstance(random_state, numbers.Integral):
        return int(random_state)
    if hasattr(random_state, 'randint'):  # numpy RandomState or Generator
        # Generate a seed from the random state
        return int(random_state.randint(0, 2**31 - 1))
    raise TypeError(
        f"{name} must be None, an integer, or a numpy RandomState/Generator, "
        f"got {type(random_state).__name__}"
    )


def _validate_features(X, name="X"):
    """Validate feature array dimensions.

    Checks that n_features does not exceed IntgrML's limit of 65,535.
    """
    n_features = X.shape[1] if X.ndim > 1 else 1
    if n_features > MAX_FEATURES:
        raise ValueError(
            f"IntgrML supports up to {MAX_FEATURES:,} features; "
            f"got {n_features:,}. Consider feature selection or dimensionality reduction."
        )
    return n_features

try:
    # sklearn base classes enable ecosystem compatibility (BSD license)
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
    _SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback for environments without sklearn
    _SKLEARN_AVAILABLE = False

    # Create minimal base classes to avoid duplicate base class errors
    class BaseEstimator:
        """Minimal base estimator for sklearn-free environments"""
        pass

    class ClassifierMixin:
        """Minimal classifier mixin for sklearn-free environments"""
        pass

    class RegressorMixin:
        """Minimal regressor mixin for sklearn-free environments"""
        pass

    # Create minimal validation functions for sklearn-free environments
    def check_X_y(X, y, **kwargs):
        """Minimal validation when sklearn is not available"""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        return X, y

    def check_array(X, **kwargs):
        """Minimal array validation when sklearn is not available"""
        return np.asarray(X)

    def check_is_fitted(estimator, attributes):
        """Minimal fitted check when sklearn is not available"""
        for attr in attributes:
            if not hasattr(estimator, attr):
                raise ValueError(f"This {type(estimator).__name__} instance is not fitted yet")

from intgrml._core import Boost as _CoreBoost, Forest as _CoreForest, OvR as _CoreOvR


class IntgrBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Integer-only gradient boosted trees for binary classification.

    This is a scikit-learn compatible interface to IntgrML's IntgrBoost algorithm,
    which uses pure integer arithmetic for training and inference. Perfect for
    deployment on embedded systems without floating-point hardware.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to build). Must be >= 1.

    max_depth : int, default=6
        Maximum tree depth (1-15). Deeper trees can model more complex patterns
        but may overfit.

    learning_rate : float, default=0.25
        Boosting learning rate (shrinkage). Lower values require more trees
        but may generalize better. Must be > 0, typical range: 0.01-1.0.

    min_samples_leaf : int, default=8
        Minimum number of samples required in a leaf node. Must be >= 1.

    n_bins : int, default=256
        Number of histogram bins for feature quantization (2-256). Higher values
        preserve more information but increase memory usage.

    clip_min : float, default=-10.0
        Minimum value for feature clipping during quantization.
        Must be < clip_max.

    clip_max : float, default=10.0
        Maximum value for feature clipping during quantization.
        Must be > clip_min.

    temperature : float, default=100.0
        Temperature scaling for logit-to-probability conversion in predict_proba().
        IntgrML returns Q8.8 fixed-point logits; dividing by temperature before
        sigmoid produces calibrated probabilities. Higher values → more uniform,
        lower values → more confident predictions. Must be > 0.

    random_state : int or None, default=None
        Random seed for reproducibility. IntgrML guarantees bit-exact
        reproducibility when the same seed is used. If None, uses internal
        default (42 for backward compatibility).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit

    feature_importances_ : ndarray of shape (n_features_,)
        Feature importances (normalized to sum to 1.0)

    classes_ : ndarray of shape (n_classes,)
        Class labels (always [0, 1] for binary classification)

    Examples
    --------
    >>> from intgrml import IntgrBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = IntgrBoostClassifier(n_estimators=100, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> score = clf.score(X_test, y_test)

    Notes
    -----
    IntgrML uses Q8.8 fixed-point arithmetic with a Q16.16 accumulator, ensuring
    no precision loss during gradient accumulation while maintaining deterministic
    integer-only operations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.25,
        min_samples_leaf: int = 8,
        n_bins: int = 256,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        temperature: float = DEFAULT_TEMPERATURE,
        random_state: Optional[int] = None,
    ):
        # Store parameters as-is (sklearn convention for clone/get_params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.temperature = temperature
        self.random_state = random_state

    def _validate_params(self):
        """Validate all parameters. Called at fit() time."""
        _validate_n_estimators(self.n_estimators)
        _validate_max_depth(self.max_depth)
        _validate_learning_rate(self.learning_rate)
        _validate_min_samples_leaf(self.min_samples_leaf)
        _validate_n_bins(self.n_bins)
        _validate_clip_bounds(self.clip_min, self.clip_max)
        _validate_temperature(self.temperature)
        _validate_random_state(self.random_state)

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, "pd.Series"],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "IntgrBoostClassifier":
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        y : array-like of shape (n_samples,)
            Target values. For binary: 0 or 1. For multiclass: 0, 1, 2, ...

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights (currently not supported, reserved for future use)

        Returns
        -------
        self : object
            Fitted estimator

        Notes
        -----
        For multiclass problems (K > 2 classes), this uses One-vs-Rest (OvR)
        strategy internally, training K binary classifiers in parallel.
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not yet supported")

        # Validate parameters before proceeding
        self._validate_params()

        # Handle pandas DataFrames
        if hasattr(X, "values"):  # pandas DataFrame
            self.feature_names_in_ = list(X.columns)
            X = X.values
        if hasattr(y, "values"):  # pandas Series
            y = y.values

        # Validate input
        if _SKLEARN_AVAILABLE:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Validate feature count
        _validate_features(X)

        # Store number of features
        self.n_features_in_ = X.shape[1]

        # Ensure y is int32
        y = y.astype(np.int32)

        # Detect number of classes and store class labels
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.n_classes_ = len(unique_classes)

        # Validate labels are contiguous starting from 0
        if unique_classes[0] != 0 or unique_classes[-1] != self.n_classes_ - 1:
            # Remap labels to 0..K-1
            label_map = {label: i for i, label in enumerate(unique_classes)}
            y = np.array([label_map[label] for label in y], dtype=np.int32)

        # Get validated random_state (None -> use internal default)
        seed = _validate_random_state(self.random_state)
        if seed is None:
            seed = 42  # IntgrML's internal default for reproducibility

        if self.n_classes_ == 2:
            # Binary classification: use existing Boost
            self._is_multiclass = False
            self.model_ = _CoreBoost(
                trees=int(self.n_estimators),
                depth=int(self.max_depth),
                bins=int(self.n_bins),
                learning_rate=float(self.learning_rate),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=seed,
            )
            self.model_.fit(X, y, clip_min=float(self.clip_min), clip_max=float(self.clip_max))
        else:
            # Multiclass: use One-vs-Rest (OvR)
            self._is_multiclass = True
            self.model_ = _CoreOvR(
                trees=int(self.n_estimators),
                depth=int(self.max_depth),
                bins=int(self.n_bins),
                learning_rate=float(self.learning_rate),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=seed,
            )
            self.model_.fit(X, y, clip_min=float(self.clip_min), clip_max=float(self.clip_max))

        return self

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        # Handle pandas DataFrames
        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self._is_multiclass:
            # Multiclass: OvR returns class indices directly
            preds = self.model_.predict(X)
        else:
            # Binary: use predict_class which applies threshold
            preds = self.model_.predict_class(X)

        # Map back to original class labels
        return self.classes_[preds]

    def predict_proba(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.

        Notes
        -----
        For binary classification, applies sigmoid to logits.
        For multiclass, applies softmax to K logits from OvR heads.
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self._is_multiclass:
            # Multiclass: get logits for all K classes
            logits = self.model_.predict_logits(X)  # (n_samples, n_classes)
            logits = logits.astype(np.float64) / self.temperature

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return proba
        else:
            # Binary: single logit, apply sigmoid
            logits = self.model_.predict(X)
            proba_class_1 = 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / self.temperature))
            proba_class_0 = 1.0 - proba_class_1
            return np.column_stack([proba_class_0, proba_class_1])

    def decision_function(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Compute the decision function of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        decision : ndarray
            For binary: shape (n_samples,), raw logits
            For multiclass: shape (n_samples, n_classes), raw logits per class
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self._is_multiclass:
            return self.model_.predict_logits(X).astype(np.float64)
        else:
            return self.model_.predict(X).astype(np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Feature importances (normalized).

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Normalized feature importances (sum to 1.0)
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        return self.model_.feature_importances_

    def score(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        y : array-like of shape (n_samples,)
            True labels for X

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y
        """
        y_pred = self.predict(X)
        if hasattr(y, "values"):
            y = y.values
        y = np.asarray(y)
        return float((y_pred == y).mean())

    def save(self, path: str) -> None:
        """Save model to file.

        Uses .sbf format for binary, .ovr format for multiclass.
        A sidecar .meta file stores quantization parameters.
        """
        self.model_.save(path)

    def load(self, path: str) -> None:
        """Load model from file.

        Detects format from file extension (.sbf for binary, .ovr for multiclass).
        """
        seed = _validate_random_state(self.random_state)
        if seed is None:
            seed = 42

        # Detect format from extension
        if path.endswith('.ovr'):
            # Multiclass model
            self._is_multiclass = True
            self.model_ = _CoreOvR(
                trees=int(self.n_estimators),
                depth=int(self.max_depth),
                bins=int(self.n_bins),
                learning_rate=float(self.learning_rate),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=seed,
            )
            self.model_.load(path)
            self.n_features_in_ = self.model_.n_features
            self.n_classes_ = self.model_.n_classes
            self.classes_ = np.arange(self.n_classes_)
        else:
            # Binary model (.sbf)
            self._is_multiclass = False
            self.model_ = _CoreBoost(
                trees=int(self.n_estimators),
                depth=int(self.max_depth),
                bins=int(self.n_bins),
                learning_rate=float(self.learning_rate),
                min_samples_leaf=int(self.min_samples_leaf),
                random_state=seed,
            )
            self.model_.load(path)
            self.n_features_in_ = self.model_.n_features
            self.n_classes_ = 2
            self.classes_ = np.array([0, 1])


class IntgrBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Integer-only gradient boosted trees for regression.

    Similar to IntgrBoostClassifier but for continuous target values.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds. Must be >= 1.

    max_depth : int, default=6
        Maximum tree depth (1-15).

    learning_rate : float, default=0.25
        Boosting learning rate. Must be > 0, typical range: 0.01-1.0.

    min_samples_leaf : int, default=8
        Minimum samples per leaf. Must be >= 1.

    n_bins : int, default=256
        Number of histogram bins (2-256).

    clip_min : float, default=-10.0
        Minimum value for feature clipping. Must be < clip_max.

    clip_max : float, default=10.0
        Maximum value for feature clipping. Must be > clip_min.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> from intgrml import IntgrBoostRegressor
    >>> reg = IntgrBoostRegressor(n_estimators=100)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)

    Notes
    -----
    This class is a placeholder for future implementation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.25,
        min_samples_leaf: int = 8,
        n_bins: int = 256,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        random_state: Optional[int] = None,
    ):
        # Store parameters as-is (sklearn convention)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the model."""
        raise NotImplementedError("IntgrBoostRegressor coming in a future release")

    def predict(self, X):
        """Predict continuous values."""
        raise NotImplementedError("IntgrBoostRegressor coming in a future release")


class IntgrForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Integer-only random forest for classification.

    scikit-learn compatible interface to IntgrML's IntgrForest algorithm.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest. Must be >= 1.

    max_depth : int, default=8
        Maximum tree depth (1-15).

    min_samples_leaf : int, default=10
        Minimum samples per leaf. Must be >= 1.

    n_bins : int, default=256
        Number of histogram bins (2-256).

    subsample : float, default=0.8
        Fraction of samples to use for each tree. Must be in (0, 1].

    clip_min : float, default=-10.0
        Minimum value for feature clipping. Must be < clip_max.

    clip_max : float, default=10.0
        Maximum value for feature clipping. Must be > clip_min.

    temperature : float, default=100.0
        Temperature scaling for logit-to-probability conversion in predict_proba().
        Higher values → more uniform, lower values → more confident predictions.
        Must be > 0.

    random_state : int or None, default=None
        Random seed for reproducibility. If None, uses internal default (42).

    Examples
    --------
    >>> from intgrml import IntgrForestClassifier
    >>> clf = IntgrForestClassifier(n_estimators=100)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 10,
        n_bins: int = 256,
        subsample: float = 0.8,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
        temperature: float = DEFAULT_TEMPERATURE,
        random_state: Optional[int] = None,
    ):
        # Store parameters as-is (sklearn convention for clone/get_params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.subsample = subsample
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.temperature = temperature
        self.random_state = random_state

    def _validate_params(self):
        """Validate all parameters. Called at fit() time."""
        _validate_n_estimators(self.n_estimators)
        _validate_max_depth(self.max_depth)
        _validate_min_samples_leaf(self.min_samples_leaf)
        _validate_n_bins(self.n_bins)
        _validate_subsample(self.subsample)
        _validate_clip_bounds(self.clip_min, self.clip_max)
        _validate_temperature(self.temperature)
        _validate_random_state(self.random_state)

    def fit(self, X, y, sample_weight=None):
        """Fit the random forest model."""
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not yet supported")

        # Validate parameters before proceeding
        self._validate_params()

        if hasattr(X, "values"):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        if _SKLEARN_AVAILABLE:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Validate feature count
        _validate_features(X)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        y = y.astype(np.int32)

        # Get validated random_state (None -> use internal default)
        seed = _validate_random_state(self.random_state)
        if seed is None:
            seed = 42  # IntgrML's internal default for reproducibility

        self.model_ = _CoreForest(
            trees=int(self.n_estimators),
            depth=int(self.max_depth),
            bins=int(self.n_bins),
            min_samples_leaf=int(self.min_samples_leaf),
            subsample=float(self.subsample),
            random_state=seed,
        )

        self.model_.fit(X, y, clip_min=float(self.clip_min), clip_max=float(self.clip_max))
        return self

    def predict(self, X):
        """Predict class labels"""
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        logits = self.model_.predict(X)
        return (logits >= 0).astype(np.int32)

    def predict_proba(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Column 0 is P(y=0), column 1 is P(y=1)

        Notes
        -----
        IntgrML returns integer logits. This method applies sigmoid transformation
        to convert logits to probabilities.
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Get logits
        logits = self.model_.predict(X)

        # Apply sigmoid: p = 1 / (1 + exp(-logit / temperature))
        # IntgrML returns integer logits; temperature scaling converts
        # to well-calibrated probabilities.
        proba_class_1 = 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / self.temperature))
        proba_class_0 = 1.0 - proba_class_1

        return np.column_stack([proba_class_0, proba_class_1])

    def decision_function(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Compute the decision function of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        decision : ndarray of shape (n_samples,)
            Decision function values (raw logits)
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        return self.model_.predict(X).astype(np.float64)

    def score(self, X: Union[np.ndarray, "pd.DataFrame"], y: Union[np.ndarray, "pd.Series"]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        y : array-like of shape (n_samples,)
            True labels for X

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y
        """
        y_pred = self.predict(X)
        if hasattr(y, "values"):
            y = y.values
        y = np.asarray(y)
        return float((y_pred == y).mean())

    def save(self, path: str) -> None:
        """Save model to file"""
        self.model_.save(path)

    def load(self, path: str) -> None:
        """Load model from file."""
        # Create model instance if not already fitted
        if not hasattr(self, "model_"):
            seed = _validate_random_state(self.random_state)
            if seed is None:
                seed = 42
            self.model_ = _CoreForest(
                trees=int(self.n_estimators),
                depth=int(self.max_depth),
                bins=int(self.n_bins),
                min_samples_leaf=int(self.min_samples_leaf),
                subsample=float(self.subsample),
                random_state=seed,
            )
        self.model_.load(path)
        # Set fitted attributes
        self.n_features_in_ = self.model_.n_features
        self.classes_ = np.array([0, 1])

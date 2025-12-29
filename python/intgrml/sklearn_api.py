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

from intgrml._core import Boost as _CoreBoost, Forest as _CoreForest


class IntgrBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Integer-only gradient boosted trees for binary classification.

    This is a scikit-learn compatible interface to IntgrML's IntgrBoost algorithm,
    which uses pure integer arithmetic for training and inference. Perfect for
    deployment on embedded systems without floating-point hardware.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees to build)

    max_depth : int, default=6
        Maximum tree depth. Deeper trees can model more complex patterns
        but may overfit

    learning_rate : float, default=0.25
        Boosting learning rate (shrinkage). Lower values require more trees
        but may generalize better. Range: 0.0-1.0

    min_samples_leaf : int, default=8
        Minimum number of samples required in a leaf node

    n_bins : int, default=256
        Number of histogram bins for feature quantization. Higher values
        preserve more information but increase memory usage

    clip_min : float, default=-10.0
        Minimum value for feature clipping during quantization

    clip_max : float, default=10.0
        Maximum value for feature clipping during quantization

    random_state : int, default=None
        Random seed for reproducibility. IntgrML guarantees bit-exact
        reproducibility when the same seed is used

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
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state if random_state is not None else 42

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
            Target values (binary: 0 or 1)

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights (currently not supported, reserved for future use)

        Returns
        -------
        self : object
            Fitted estimator
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not yet supported in v1.2.1")

        # Handle pandas DataFrames
        if hasattr(X, "values"):  # pandas DataFrame
            self.feature_names_in_ = list(X.columns)
            X = X.values
        if hasattr(y, "values"):  # pandas Series
            y = y.values

        # Validate input
        if _SKLEARN_AVAILABLE:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Store number of features
        self.n_features_in_ = X.shape[1]

        # Store class labels
        self.classes_ = np.array([0, 1])

        # Ensure y is int32
        y = y.astype(np.int32)

        # Create and fit core model
        self.model_ = _CoreBoost(
            trees=self.n_estimators,
            depth=self.max_depth,
            bins=self.n_bins,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        self.model_.fit(X, y, clip_min=self.clip_min, clip_max=self.clip_max)

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
            Predicted class labels (0 or 1)
        """
        if _SKLEARN_AVAILABLE:
            check_is_fitted(self, ["model_"])

        # Handle pandas DataFrames
        if hasattr(X, "values"):
            X = X.values

        if _SKLEARN_AVAILABLE:
            X = check_array(X, accept_sparse=False, dtype=np.float64)

        return self.model_.predict_class(X)

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

        # Apply sigmoid: p = 1 / (1 + exp(-logit))
        # Use stable sigmoid computation
        proba_class_1 = 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / 100.0))
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

    def save(self, path: str) -> None:
        """Save model to file (.sbf format)"""
        self.model_.save(path)

    def load(self, path: str) -> None:
        """Load model from file (.sbf format)"""
        # Create model instance if not already fitted
        if not hasattr(self, "model_"):
            self.model_ = _CoreBoost(
                trees=self.n_estimators,
                depth=self.max_depth,
                bins=self.n_bins,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        self.model_.load(path)
        # Set fitted attributes
        self.n_features_in_ = self.model_.n_features
        self.classes_ = np.array([0, 1])


class IntgrBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Integer-only gradient boosted trees for regression.

    Similar to IntgrBoostClassifier but for continuous target values.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds

    max_depth : int, default=6
        Maximum tree depth

    learning_rate : float, default=0.25
        Boosting learning rate

    min_samples_leaf : int, default=8
        Minimum samples per leaf

    n_bins : int, default=256
        Number of histogram bins

    clip_min : float, default=-10.0
        Minimum feature value

    clip_max : float, default=10.0
        Maximum feature value

    random_state : int, default=None
        Random seed

    Examples
    --------
    >>> from intgrml import IntgrBoostRegressor
    >>> reg = IntgrBoostRegressor(n_estimators=100)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state if random_state is not None else 42

    def fit(self, X, y, sample_weight=None):
        """Fit the model (same as IntgrBoostClassifier)"""
        # Similar implementation to classifier
        # (For now, delegate to classifier - full implementation pending)
        raise NotImplementedError("IntgrBoostRegressor coming in v1.3.0")

    def predict(self, X):
        """Predict continuous values"""
        raise NotImplementedError("IntgrBoostRegressor coming in v1.3.0")


class IntgrForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Integer-only random forest for classification.

    scikit-learn compatible interface to IntgrML's IntgrForest algorithm.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest

    max_depth : int, default=8
        Maximum tree depth

    min_samples_leaf : int, default=10
        Minimum samples per leaf

    n_bins : int, default=256
        Number of histogram bins

    subsample : float, default=0.8
        Fraction of samples to use for each tree

    clip_min : float, default=-10.0
        Minimum feature value

    clip_max : float, default=10.0
        Maximum feature value

    random_state : int, default=None
        Random seed

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
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.subsample = subsample
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state if random_state is not None else 42

    def fit(self, X, y, sample_weight=None):
        """Fit the random forest model"""
        if sample_weight is not None:
            raise NotImplementedError("sample_weight not yet supported")

        if hasattr(X, "values"):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        if _SKLEARN_AVAILABLE:
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        y = y.astype(np.int32)

        self.model_ = _CoreForest(
            trees=self.n_estimators,
            depth=self.max_depth,
            bins=self.n_bins,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        self.model_.fit(X, y, clip_min=self.clip_min, clip_max=self.clip_max)
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

    def save(self, path: str) -> None:
        """Save model to file"""
        self.model_.save(path)

    def load(self, path: str) -> None:
        """Load model from file"""
        # Create model instance if not already fitted
        if not hasattr(self, "model_"):
            self.model_ = _CoreForest(
                trees=self.n_estimators,
                depth=self.max_depth,
                bins=self.n_bins,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                random_state=self.random_state,
            )
        self.model_.load(path)
        # Set fitted attributes
        self.n_features_in_ = self.model_.n_features
        self.classes_ = np.array([0, 1])

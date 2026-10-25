from typing import Any, Sequence, Tuple
import numpy as np
import numpy.typing as npt


def _check_1d_array(X: npt.ArrayLike) -> npt.NDArray[Any]:
    """
    Ensure the input is a 1D NumPy array.

    Parameters
    ----------
    X : npt.ArrayLike
        Input data.

    Returns
    -------
    npt.NDArray[Any]
        A 1D NumPy array.

    Raises
    ------
    ValueError
        If the input is not 1D.
    """
    X_arr = np.asarray(X)
    if X_arr.ndim != 1:
        raise ValueError(f"X must be a 1D array. Received shape {X_arr.shape}.")
    return X_arr


def _check_is_fitted(estimator: Any, attributes: Sequence[str]) -> None:
    """Check if the estimator has been fitted by verifying necessary attributes."""
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(
                f"This {estimator.__class__.__name__} instance is not fitted yet. Call 'fit' first."
            )


class StandardScaler:
    """
    Standardize 1D features by removing the mean and scaling to unit variance.
    """

    epsilon: float
    mean_: float
    scale_: float

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def fit(self, X: npt.ArrayLike) -> "StandardScaler":
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        self : StandardScaler
            The fitted scaler.

        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        if X_arr.size == 0:
            raise ValueError("StandardScaler cannot fit on an empty array.")

        self.mean_ = float(np.mean(X_arr))
        self.scale_ = float(np.std(X_arr))

        # Prevent division by zero if the feature is completely constant
        if self.scale_ == 0.0:
            self.scale_ = 1.0

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Apply standardization by centering and scaling.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            Standardized 1D array, shape (n_samples,).

        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ["mean_", "scale_"])
        X_arr = _check_1d_array(X)

        return (X_arr - self.mean_) / (self.scale_ + self.epsilon)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            Standardized 1D array, shape (n_samples,).
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Transform 1D features by scaling to a given range [0, 1].
    """

    feature_range: Tuple[float, float]
    data_min_: float
    data_max_: float
    data_range_: float

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range

    def fit(self, X: npt.ArrayLike) -> "MinMaxScaler":
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        self : MinMaxScaler
            The fitted scaler.

        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        if X_arr.size == 0:
            raise ValueError("MinMaxScaler cannot fit on an empty array.")

        self.data_min_ = float(np.min(X_arr))
        self.data_max_ = float(np.max(X_arr))
        self.data_range_ = self.data_max_ - self.data_min_

        # Prevent division by zero for constant features
        if self.data_range_ == 0.0:
            self.data_range_ = 1.0

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Scale features of X according to feature_range.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            Scaled 1D array, shape (n_samples,).

        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ["data_min_", "data_range_"])
        X_arr = _check_1d_array(X)

        X_std = (X_arr - self.data_min_) / self.data_range_
        min_val, max_val = self.feature_range
        return X_std * (max_val - min_val) + min_val

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            Scaled 1D array, shape (n_samples,).
        """
        return self.fit(X).transform(X)


class Imputer:
    """
    Imputation transformer for completing missing values received directly in a 1D array.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy ('mean', 'median', or 'constant').
    fill_value : Any, default=0.0
        The constant value to replace missing values with when strategy='constant'.
    missing_values : Any, default=np.nan
        The placeholder for the missing values received in the input data.
    """

    strategy: str
    fill_value: Any
    missing_values: Any
    statistics_: Any

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Any = 0.0,
        missing_values: Any = np.nan,
    ) -> None:
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError("Strategy must be 'mean', 'median', or 'constant'")
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

    def _get_missing_mask(self, X_arr: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        """Return a boolean mask indicating where missing_values are located."""
        if isinstance(self.missing_values, (float, np.floating)) and np.isnan(self.missing_values):
            if X_arr.dtype.kind in "biufc":
                return np.isnan(X_arr)
            if X_arr.dtype.kind in "USO":
                res = np.zeros(X_arr.size, dtype=bool)
                for idx, val in enumerate(X_arr):
                    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)) or (isinstance(val, str) and val.strip().casefold() in {"", "nan"}):
                        res[idx] = True
                return res
            return np.isnan(X_arr.astype(float, copy=False))
        return X_arr == self.missing_values

    def fit(self, X: npt.ArrayLike) -> "Imputer":
        """
        Learn the imputation scalar value from the input 1D array containing missing values.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array containing missing values directly, shape (n_samples,).

        Returns
        -------
        self : Imputer
            The fitted imputer.

        Complexity
        ----------
        Time: O(n_samples) for mean/constant, O(n_samples log n_samples) for median
        Space: O(n_samples)
        """
        X_arr = _check_1d_array(X)
        if X_arr.size == 0:
            raise ValueError("Imputer cannot fit on an empty array.")

        mask = self._get_missing_mask(X_arr)
        valid_data = X_arr[~mask]

        # If a non-numeric fill_value is provided with default strategy='mean',
        # automatically switch to 'constant' to support string pipelines.
        actual_strategy = self.strategy
        if valid_data.size > 0 and valid_data.dtype.kind not in "biufc":
            if actual_strategy in ["mean", "median"]:
                if not isinstance(self.fill_value, (int, float, np.number)):
                    actual_strategy = "constant"
                else:
                    raise ValueError(
                        f"Cannot compute strategy='{actual_strategy}' on non-numeric string data."
                    )

        if actual_strategy == "constant" or (valid_data.size > 0 and valid_data.dtype.kind not in "biufc"):
            self.statistics_ = self.fill_value
        elif valid_data.size == 0:
            self.statistics_ = float(self.fill_value)
        else:
            if actual_strategy == "mean":
                self.statistics_ = float(np.mean(valid_data.astype(float)))
            elif actual_strategy == "median":
                self.statistics_ = float(np.median(valid_data.astype(float)))

        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Replace missing values in the input 1D array with the learned statistic.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array containing missing values directly, shape (n_samples,).

        Returns
        -------
        npt.NDArray[Any]
            Imputed 1D array, shape (n_samples,).

        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ["statistics_"])
        X_arr = _check_1d_array(X)

        mask = self._get_missing_mask(X_arr)
        return np.where(mask, self.statistics_, X_arr)

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data with missing values, shape (n_samples,).

        Returns
        -------
        npt.NDArray[Any]
            Imputed 1D array, shape (n_samples,).
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Encode a 1D categorical feature array into a 2D one-hot matrix.
    Supports both non-negative integer categories (contiguous or non-contiguous)
    and string categories using 100% vectorized NumPy operations.
    """

    categories_: int
    unique_categories_: npt.NDArray[Any]
    _category_type: str

    def fit(self, X: npt.ArrayLike) -> "OneHotEncoder":
        """
        Learn the sorted unique categories present in a 1D feature.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D categorical data array, shape (n_samples,).

        Returns
        -------
        self : OneHotEncoder
            The fitted encoder.

        Complexity
        ----------
        Time: O(n_samples log n_samples)
        Space: O(n_categories)
        """
        X_arr = _check_1d_array(X)
        if X_arr.size == 0:
            raise ValueError("OneHotEncoder requires at least one category.")

        if np.issubdtype(X_arr.dtype, np.integer):
            if np.any(X_arr < 0):
                raise ValueError("Categories must be non-negative integers.")
            self._category_type = "integer"
        elif X_arr.dtype.kind in "USO":
            self._category_type = "string"
        else:
            raise ValueError(
                "OneHotEncoder expects integer categorical data or strings."
            )

        self.unique_categories_ = np.unique(X_arr)
        self.categories_ = int(self.unique_categories_.size)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Transform the 1D data to a 2D one-hot encoded matrix using vectorized binary search.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D categorical data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            One-hot encoded 2D array, shape (n_samples, n_categories).

        Raises
        ------
        ValueError
            If unseen categories or mismatched data types are encountered.

        Complexity
        ----------
        Time: O(n_samples log n_categories + n_samples * n_categories)
        Space: O(n_samples * n_categories)
        """
        _check_is_fitted(self, ["categories_", "unique_categories_", "_category_type"])
        X_arr = _check_1d_array(X)

        if self._category_type == "integer" and not np.issubdtype(X_arr.dtype, np.integer):
            raise ValueError("OneHotEncoder expects integer categorical data.")
        if self._category_type == "string" and X_arr.dtype.kind not in "USO":
            raise ValueError("OneHotEncoder expects string categorical data.")

        if X_arr.size == 0:
            return np.empty((0, self.categories_), dtype=float)

        # Vectorized binary search maps each category to its column index without Python loops
        indices = np.searchsorted(self.unique_categories_, X_arr)

        # Clip indices to safely verify exact match against learned unique_categories_
        safe_indices = np.clip(indices, 0, self.categories_ - 1)
        unknown_mask = (
            (indices < 0)
            | (indices >= self.categories_)
            | (self.unique_categories_[safe_indices] != X_arr)
        )
        if np.any(unknown_mask):
            raise ValueError("Found unknown categories during transform.")

        # Allocate only the output matrix and set 1.0 at each sample's category column
        encoded = np.zeros((X_arr.size, self.categories_), dtype=float)
        encoded[np.arange(X_arr.size), indices] = 1.0
        return encoded

    def fit_transform(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : npt.ArrayLike
            Input 1D data array, shape (n_samples,).

        Returns
        -------
        npt.NDArray[np.floating[Any]]
            Scaled 1D array, shape (n_samples,).
        """
        return self.fit(X).transform(X)
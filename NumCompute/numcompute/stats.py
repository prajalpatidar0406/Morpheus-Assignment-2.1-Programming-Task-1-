
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def _as_numeric_array(
    data: npt.ArrayLike,
) -> NDArray[np.float64]:
    """
    Convert input to a numeric NumPy array.
    
    Parameters
    ----------
    data : array-like
        Input data containing real numeric values.

    Returns
    -------
    numpy.ndarray
        Floating-point array containing the input data.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = np.asarray(data)

    if arr.dtype.kind not in "iuf":
        raise TypeError("Data must contain real numeric values")

    return arr.astype(float, copy=False)


def _validate_data(
    data: npt.ArrayLike,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Validate numeric input and optionally allow empty arrays.
    Parameters
    ----------
    data : array-like
        Input data to validate.
    allow_empty : bool, default=False
        Whether to allow an empty array.

    Returns
    -------
    numpy.ndarray
        Validated floating-point array.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the array is empty and allow_empty is False.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = _as_numeric_array(data)

    if arr.size == 0 and not allow_empty:
        raise ValueError("Data must not be empty")

    return arr


def mean(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the arithmetic mean.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the mean.
        None calculates the mean over all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Mean value or array of mean values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmean(arr, axis=axis)

    return np.mean(arr, axis=axis)


def median(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the median.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the median.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Median value or array of median values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmedian(arr, axis=axis)

    return np.median(arr, axis=axis)


def std(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the standard deviation of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate standard deviation.
    ddof : int, default=0
        Delta degrees of freedom. Zero computes population standard
        deviation; one computes sample standard deviation.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Standard deviation value or array of values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty, ddof is invalid, or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")

    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)

    return np.std(arr, axis=axis, ddof=ddof)


def minimum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the minimum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the minimum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Minimum value or array of minimum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmin(arr, axis=axis)

    return np.min(arr, axis=axis)


def maximum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the maximum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the maximum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Maximum value or array of maximum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.    
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmax(arr, axis=axis)

    return np.max(arr, axis=axis)


def quantile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested quantile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Quantile value(s) between 0 and 1, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate quantiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Quantile value or array of quantile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 1], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    if skipna:
        return np.nanquantile(arr, q_arr, axis=axis)

    return np.quantile(arr, q_arr, axis=axis)


def percentile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested percentile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Percentile value(s) between 0 and 100, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate percentiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Percentile value or array of percentile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 100], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(data, q_arr / 100.0, axis=axis, skipna=skipna)


def histogram(
    data: npt.ArrayLike,
    bins: int | str | npt.ArrayLike = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    skipna: bool = False,
) -> tuple[NDArray[np.int64] | NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a histogram of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    bins : int, str, or array-like, default=10
        Number of bins, a supported binning strategy, or explicit
        bin edges.
    range : tuple of float, optional
        Lower and upper range of the bins.
    density : bool, default=False
        If True, return normalized density values instead of counts.
    skipna : bool, default=False
        If True, ignore NaN values. Infinite values are always rejected.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing histogram counts or densities and bin edges.

    Raises
    ------
    TypeError
        If the data contains non-numeric or complex values.
    ValueError
        If the data has no valid elements, contains NaN without
        skipna=True, contains infinity, or the bin configuration is invalid.

    Complexity: O(n + b) time, where n is the number of observations
    and b is the number of bins.
    """
    arr = _validate_data(data, allow_empty=True).ravel()

    if np.any(np.isinf(arr)):
        raise ValueError("Data must not contain infinite values")

    if skipna:
        arr = arr[~np.isnan(arr)]
    elif np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN; use skipna=True to ignore it")

    if arr.size == 0:
        raise ValueError("Data must contain at least one valid value")

    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("Bins must be a positive integer")
    elif not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "Bin edges must be a 1D array with at least 2 values"
            )

        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Bin edges must be finite and strictly increasing"
            )

    counts, edges = np.histogram(
        arr,
        bins=bins,
        range=range,
        density=density,
    )

    return counts, edges

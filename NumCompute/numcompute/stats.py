import numpy as np

def _as_numeric_array(data):
    """Convert input to a numeric NumPy array."""
    arr = np.asarray(data)

    if arr.dtype.kind not in "iuf":
        raise TypeError("data must contain real numeric values")

    return arr.astype(float, copy=False)


def _validate_data(data, allow_empty=False):
    """Validate numeric input and optionally allow empty arrays."""
    arr = _as_numeric_array(data)

    if arr.size == 0 and not allow_empty:
        raise ValueError("data must not be empty")

    return arr


def mean(data, axis=None, skipna=False):
    """Return the arithmetic mean."""
    arr = _validate_data(data)

    if skipna:
        return np.nanmean(arr, axis=axis)

    return np.mean(arr, axis=axis)


def median(data, axis=None, skipna=False):
    """Return the median."""
    arr = _validate_data(data)

    if skipna:
        return np.nanmedian(arr, axis=axis)

    return np.median(arr, axis=axis)


def std(data, axis=None, ddof=0, skipna=False):
    """Return the standard deviation.

    ddof=0 computes population standard deviation.
    ddof=1 computes sample standard deviation.
    """
    arr = _validate_data(data)

    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")

    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)

    return np.std(arr, axis=axis, ddof=ddof)


def minimum(data, axis=None, skipna=False):
    """Return the minimum value."""
    arr = _validate_data(data)

    if skipna:
        return np.nanmin(arr, axis=axis)

    return np.min(arr, axis=axis)


def maximum(data, axis=None, skipna=False):
    """Return the maximum value."""
    arr = _validate_data(data)

    if skipna:
        return np.nanmax(arr, axis=axis)

    return np.max(arr, axis=axis)


def quantile(data, q, axis=None, skipna=False):
    """Return quantile(s), where q lies between 0 and 1.

    Examples:
        quantile([1, 2, 3, 4], 0.5) -> 2.5
        quantile([1, 2, 3, 4], [0.25, 0.75]) -> [1.75, 3.25]
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


def percentile(data, q, axis=None, skipna=False):
    """Return percentile(s), where q lies between 0 and 100."""
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(data, q_arr / 100.0, axis=axis, skipna=skipna)


def histogram(data, bins=10, range=None, density=False, skipna=False):
    """Return histogram counts and bin edges.

    NaN values raise an error unless skipna=True.
    Infinite values are rejected.
    """
    arr = _validate_data(data, allow_empty=True).ravel()

    if np.any(np.isinf(arr)):
        raise ValueError("data must not contain infinite values")

    if skipna:
        arr = arr[~np.isnan(arr)]
    elif np.any(np.isnan(arr)):
        raise ValueError("data contains NaN; use skipna=True to ignore it")

    if arr.size == 0:
        raise ValueError("data must contain at least one valid value")

    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
    elif not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin edges must be a 1D array with at least 2 values")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be finite and strictly increasing")

    counts, edges = np.histogram(
        arr, bins=bins, range=range, density=density
    )
    return counts, edges

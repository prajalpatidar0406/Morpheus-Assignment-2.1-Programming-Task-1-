from typing import Any, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt


def stable_sort(arr: npt.ArrayLike, descending: bool = False) -> npt.NDArray[Any]:
    """
    Sort a 1D numpy array using stable sorting.

    Parameters
    ----------
    arr : npt.ArrayLike
        1D array to be sorted, shape (n_samples,).
    descending : bool, default=False
        If True, sort in descending order.

    Returns
    -------
    npt.NDArray[Any]
        The sorted 1D array.

    Complexity
    ----------
    Time: O(n_samples * log(n_samples))
    Space: O(n_samples)
    """
    arr_np = np.asarray(arr)
    if arr_np.ndim != 1:
        raise ValueError("stable_sort expects a 1D array.")

    # Start with a stable ascending order so ties keep their original order.
    sort_indices = np.argsort(arr_np, kind="stable")

    if descending and len(sort_indices):
        sorted_values = arr_np[sort_indices]
        equal_adjacent = sorted_values[1:] == sorted_values[:-1]
        if sorted_values.dtype.kind in "fc":
            equal_adjacent |= np.isnan(sorted_values[1:]) & np.isnan(sorted_values[:-1])
        # Reverse group order, not item order, so ties remain stable.
        group_ids = np.cumsum(np.r_[True, ~equal_adjacent]) - 1
        sort_indices = sort_indices[np.argsort(-group_ids, kind="stable")]

    return arr_np[sort_indices]


def multi_key_sort(
    keys: Sequence[npt.ArrayLike],
    descending: bool = False,
) -> npt.NDArray[np.intp]:
    """
    Sort based on multiple 1D arrays (multi-key sorting).

    Parameters
    ----------
    keys : Sequence[npt.ArrayLike]
        Tuple or list of 1D arrays to sort by, ordered from primary to least significant key.
    descending : bool, default=False
        If True, reverse the final sorted order.

    Returns
    -------
    npt.NDArray[np.intp]
        The 1D array of indices that would sort the arrays.

    Complexity
    ----------
    Time: O(n_samples * log(n_samples))
    Space: O(n_samples)
    """
    # lexsort treats its last key as primary, so reverse the caller's key order.
    if len(keys) == 0:
        raise ValueError("keys must contain at least one array.")
    key_arrays: Tuple[npt.NDArray[Any], ...] = tuple(
        np.asarray(k) for k in reversed(keys)
    )
    for k_arr in key_arrays:
        if k_arr.ndim != 1:
            raise ValueError("All keys in multi_key_sort must be 1D arrays.")
    if len({array.size for array in key_arrays}) != 1:
        raise ValueError("All keys in multi_key_sort must have the same length.")

    sort_indices = np.lexsort(key_arrays)

    if descending:
        sort_indices = sort_indices[::-1]

    return sort_indices


def topk(
    values: npt.ArrayLike,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
) -> npt.NDArray[Any]:
    """
    Retrieve the top-k elements from a 1D array.
    Uses np.argpartition for an O(n) partial sort.

    Parameters
    ----------
    values : npt.ArrayLike
        Input 1D array, shape (n_samples,).
    k : int
        Number of elements to retrieve.
    largest : bool, default=True
        If True, retrieve the largest elements. If False, the smallest.
    return_indices : bool, default=True
        If True, returns indices instead of values.

    Returns
    -------
    npt.NDArray[Any]
        Array of top-k elements or indices, shape (k,).
        Note: The returned k elements are NOT fully sorted among themselves.

    Complexity
    ----------
    Time: O(n_samples) average time.
    Space: O(n_samples) for the partition indices.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"topk expects a 1D array. Received shape {arr.shape}.")

    n = len(arr)
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer.")
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and the number of values ({n}).")

    if largest:
        # Partition puts the largest k entries at the end without sorting all n.
        idx = np.argpartition(arr, -k)[-k:]
    else:
        # For the smallest values, keep the first k entries after partitioning.
        idx = np.argpartition(arr, k - 1)[:k]

    if return_indices:
        return idx
    return arr[idx]


def quickselect(
    values: npt.ArrayLike,
    k: int,
    largest: bool = False,
) -> Any:
    """
    Finds the k-th smallest (or largest) element in an unordered 1D array.
    Uses NumPy boolean masks to partition values around a pivot.

    Parameters
    ----------
    values : npt.ArrayLike
        Input 1D array, shape (n_samples,).
    k : int
        The rank of the element to find (1-indexed). e.g., k=1 means the absolute smallest/largest.
    largest : bool, default=False
        If True, returns the k-th largest element. If False, the k-th smallest.

    Returns
    -------
    Any
        The k-th element scalar.

    Complexity
    ----------
    Time: O(n_samples) average time, O(n_samples^2) worst case.
    Space: O(n_samples) for array slicing masks.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")

    n = len(arr)
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer.")
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and the number of values ({n}).")
    if arr.dtype.kind not in "iuf":
        raise TypeError("quickselect expects real numeric values.")
    if np.any(np.isnan(arr)):
        raise ValueError("quickselect does not support NaN values.")

    if largest:
        # Convert the requested largest-item rank to its equivalent ascending rank.
        k = n - k + 1

    remaining = arr
    while True:
        # Keep only the side of the pivot where the requested rank can still be.
        pivot = remaining[len(remaining) // 2]
        left = remaining[remaining < pivot]
        equal = remaining[remaining == pivot]

        if k <= len(left):
            remaining = left
        elif k <= len(left) + len(equal):
            return pivot.item() if isinstance(pivot, np.generic) else pivot
        else:
            k -= len(left) + len(equal)
            remaining = remaining[remaining > pivot]


def binary_search(
    sorted_array: npt.ArrayLike,
    x: Union[int, float, npt.ArrayLike],
) -> Tuple[Union[int, npt.NDArray[np.intp]], Union[bool, npt.NDArray[np.bool_]]]:
    """
    Finds the insertion indices for targets in a sorted 1D array,
    and returns whether the target exactly exists in the array.

    Parameters
    ----------
    sorted_array : npt.ArrayLike
        A 1D array that is ALREADY sorted in ascending order.
    x : Union[int, float, npt.ArrayLike]
        The value(s) to search for.

    Returns
    -------
    Tuple[Union[int, npt.NDArray[np.intp]], Union[bool, npt.NDArray[np.bool_]]]
        insertion_indices: int or np.ndarray. Where the value should be inserted.
        exists: bool or np.ndarray. True if the exact value already exists.

    Complexity
    ----------
    Time: O(m * log n) where n=len(sorted_array), m=len(x)
    Space: O(m) for the output array.
    """
    arr = np.asarray(sorted_array)
    if arr.ndim != 1:
        raise ValueError("Binary search requires a 1D sorted array.")

    # Convert scalar inputs to a 1D array for vectorized processing
    targets = np.atleast_1d(x)

    # np.searchsorted performs O(log N) binary search for every target element in C
    insertion_indices = np.searchsorted(arr, targets, side="left")

    # Check for out-of-bounds indices (if target is greater than all elements)
    valid_mask = insertion_indices < len(arr)

    # Check if the value at the valid insertion index exactly matches the target
    exists = np.zeros_like(targets, dtype=bool)
    exists[valid_mask] = arr[insertion_indices[valid_mask]] == targets[valid_mask]

    # Return scalars if the original input was a scalar, else return arrays
    if np.isscalar(x):
        return int(insertion_indices[0]), bool(exists[0])

    return insertion_indices, exists

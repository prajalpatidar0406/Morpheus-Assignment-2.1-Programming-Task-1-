"""Check sorting, partial selection, quickselect, and binary search."""

import numpy as np
import pytest
from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    quickselect,
    binary_search,
)


# Sorting


def test_stable_sort_ascending():
    """Sort values from smallest to largest."""
    arr = np.array([3, 1, 4, 1, 5, 9, 2])
    sorted_arr = stable_sort(arr, descending=False)
    np.testing.assert_array_equal(sorted_arr, np.array([1, 1, 2, 3, 4, 5, 9]))


def test_stable_sort_descending():
    """Sort largest-first while retaining the order of equal items."""
    arr = np.array([3, 1, 4, 1, 5])
    sorted_arr = stable_sort(arr, descending=True)
    np.testing.assert_array_equal(sorted_arr, np.array([5, 4, 3, 1, 1]))

    class Item:
        def __init__(self, key, name):
            self.key = key
            self.name = name

        def __lt__(self, other):
            return self.key < other.key

        def __eq__(self, other):
            return self.key == other.key

    items = np.array([Item(2, "first"), Item(1, "middle"), Item(2, "last")])
    result = stable_sort(items, descending=True)
    assert [item.name for item in result] == ["first", "last", "middle"]


def test_stable_sort_negatives_and_all_equal():
    """Handle negative values and arrays where every value is tied."""
    arr_neg = np.array([-5.0, -1.0, -10.0, 0.0])
    np.testing.assert_array_equal(
        stable_sort(arr_neg), np.array([-10.0, -5.0, -1.0, 0.0])
    )

    arr_const = np.array([4, 4, 4, 4])
    np.testing.assert_array_equal(stable_sort(arr_const), arr_const)


def test_multi_key_sort_ascending():
    """Use the first key as primary and the next key to break ties."""
    primary = np.array([2, 1, 2, 1])
    secondary = np.array([20, 30, 10, 15])

    indices = multi_key_sort((primary, secondary), descending=False)
    # The result refers to original row positions, not sorted value pairs.
    np.testing.assert_array_equal(indices, np.array([3, 1, 2, 0]))


def test_multi_key_sort_descending():
    """Sort multiple keys in descending order."""
    primary = np.array([1, 2, 1])
    secondary = np.array([10, 5, 20])

    indices = multi_key_sort((primary, secondary), descending=True)
    np.testing.assert_array_equal(indices, np.array([1, 2, 0]))


def test_multi_key_sort_validates_keys():
    with pytest.raises(ValueError, match="at least one"):
        multi_key_sort([])
    with pytest.raises(ValueError, match="same length"):
        multi_key_sort(([1, 2], [1]))
    with pytest.raises(ValueError, match="1D arrays"):
        multi_key_sort(([1, 2], [[1, 2]]))


def test_stable_sort_handles_empty_descending_input():
    np.testing.assert_array_equal(stable_sort([], descending=True), [])


def test_stable_sort_descending_preserves_nan_tie_order():
    values = np.array([2.0, np.nan, 1.0, np.nan])

    result = stable_sort(values, descending=True)

    np.testing.assert_array_equal(result, [np.nan, np.nan, 2.0, 1.0])


# Top-k selection


def test_topk_largest_indices_and_values():
    """Return both the selected indices and their corresponding values."""
    arr = np.array([10, 50, 20, 40, 30])

    idx = topk(arr, k=2, largest=True, return_indices=True)
    vals = topk(arr, k=2, largest=True, return_indices=False)

    assert len(idx) == 2
    assert set(idx) == {1, 3}
    assert set(vals) == {40, 50}


def test_topk_smallest_values():
    """Select the smallest values when largest is disabled."""
    arr = np.array([100, 5, 20, 1, 50])
    vals = topk(arr, k=3, largest=False, return_indices=False)

    assert len(vals) == 3
    assert set(vals) == {1, 5, 20}


def test_topk_extreme_k_bounds():
    """Support selecting one item or every item in the array."""
    arr = np.array([8, 2, 15, 4])

    top_1 = topk(arr, k=1, largest=True, return_indices=False)
    np.testing.assert_array_equal(top_1, np.array([15]))

    top_all = topk(arr, k=len(arr), largest=True, return_indices=False)
    assert set(top_all) == {2, 4, 8, 15}


def test_topk_invalid_k():
    """Reject counts outside the range of available values."""
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="k must be between"):
        topk(arr, k=0)
    with pytest.raises(ValueError, match="k must be between"):
        topk(arr, k=4)
    with pytest.raises(ValueError, match="integer"):
        topk(arr, k=1.5)
    with pytest.raises(ValueError, match="integer"):
        topk(arr, k=True)


# Quickselect


def test_quickselect_kth_smallest():
    """Find the minimum, a middle-ranked value, and the maximum."""
    arr = np.array([7, 10, 4, 3, 20])
    assert quickselect(arr, k=1, largest=False) == 3
    assert quickselect(arr, k=3, largest=False) == 7
    assert quickselect(arr, k=5, largest=False) == 20


def test_quickselect_kth_largest_with_duplicates():
    """Count duplicate values separately when finding a ranked item."""
    arr = np.array([5, 1, 5, 3, 5, 2])
    # The three copies of 5 occupy the first three positions in descending order.
    assert quickselect(arr, k=2, largest=True) == 5
    assert quickselect(arr, k=4, largest=True) == 3
    assert quickselect(arr, k=6, largest=True) == 1


def test_quickselect_invalid_k():
    """Reject invalid ranks, missing numbers, and non-numeric input."""
    arr = np.array([10, 20, 30])
    with pytest.raises(ValueError, match="k must be between"):
        quickselect(arr, k=0)
    with pytest.raises(ValueError, match="k must be between"):
        quickselect(arr, k=5)
    with pytest.raises(ValueError, match="NaN"):
        quickselect([1.0, np.nan], k=1)
    with pytest.raises(TypeError, match="real numeric"):
        quickselect(["a", "b"], k=1)


# Binary search and input edge cases


def test_binary_search_scalar():
    """Return an insertion position and whether a scalar target exists."""
    sorted_arr = np.array([10, 20, 30, 40, 50])

    idx_found, exists_found = binary_search(sorted_arr, 30)
    assert idx_found == 2
    assert exists_found is True

    idx_missing, exists_missing = binary_search(sorted_arr, 25)
    assert idx_missing == 2
    assert exists_missing is False


def test_binary_search_vector_and_out_of_bounds():
    """Search several values, including targets outside the array's range."""
    sorted_arr = np.array([10, 20, 30])
    targets = np.array([5, 10, 25, 30, 100])

    indices, exists = binary_search(sorted_arr, targets)
    np.testing.assert_array_equal(indices, np.array([0, 0, 2, 2, 3]))
    np.testing.assert_array_equal(exists, np.array([False, True, False, True, False]))


def test_sort_search_non_contiguous_and_invalid_dimensions():
    """Accept strided one-dimensional arrays and reject multidimensional input."""
    base = np.array([50, 0, 10, 0, 30, 0, 20, 0, 40])
    strided = base[::2]  # A non-contiguous view containing [50, 10, 30, 20, 40].
    assert not strided.flags["C_CONTIGUOUS"]

    np.testing.assert_array_equal(stable_sort(strided), np.array([10, 20, 30, 40, 50]))
    assert quickselect(strided, k=3) == 30

    # All operations in this module expect a single sorted or sortable sequence.
    arr_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        stable_sort(arr_2d)
    with pytest.raises(ValueError):
        topk(arr_2d, k=1)
    with pytest.raises(ValueError):
        quickselect(arr_2d, k=1)
    with pytest.raises(ValueError):
        binary_search(arr_2d, 2)

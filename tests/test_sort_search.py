import numpy as np
import pytest
from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    quickselect,
    binary_search,
)


# =====================================================================
# Sorting Tests: stable_sort & multi_key_sort (Tests 1 - 5)
# =====================================================================

def test_stable_sort_ascending():
    """1. Test basic sorting in ascending order."""
    arr = np.array([3, 1, 4, 1, 5, 9, 2])
    sorted_arr = stable_sort(arr, descending=False)
    np.testing.assert_array_equal(sorted_arr, np.array([1, 1, 2, 3, 4, 5, 9]))


def test_stable_sort_descending():
    """2. Test sorting in descending order."""
    arr = np.array([3, 1, 4, 1, 5])
    sorted_arr = stable_sort(arr, descending=True)
    np.testing.assert_array_equal(sorted_arr, np.array([5, 4, 3, 1, 1]))


def test_stable_sort_negatives_and_all_equal():
    """3. Test stable_sort with negative values and an all-equal array."""
    arr_neg = np.array([-5.0, -1.0, -10.0, 0.0])
    np.testing.assert_array_equal(stable_sort(arr_neg), np.array([-10.0, -5.0, -1.0, 0.0]))

    arr_const = np.array([4, 4, 4, 4])
    np.testing.assert_array_equal(stable_sort(arr_const), arr_const)


def test_multi_key_sort_ascending():
    """4. Test sorting by primary key first, then breaking ties with secondary key."""
    primary = np.array([2, 1, 2, 1])
    secondary = np.array([20, 30, 10, 15])

    indices = multi_key_sort((primary, secondary), descending=False)
    # Expected order: (1, 15) at idx 3, (1, 30) at idx 1, (2, 10) at idx 2, (2, 20) at idx 0
    np.testing.assert_array_equal(indices, np.array([3, 1, 2, 0]))


def test_multi_key_sort_descending():
    """5. Test multi_key_sort in descending order."""
    primary = np.array([1, 2, 1])
    secondary = np.array([10, 5, 20])

    indices = multi_key_sort((primary, secondary), descending=True)
    np.testing.assert_array_equal(indices, np.array([1, 2, 0]))


# =====================================================================
# Top-K Partial Sort Tests (Tests 6 - 9)
# =====================================================================

def test_topk_largest_indices_and_values():
    """6. Test retrieving top-k largest indices and values."""
    arr = np.array([10, 50, 20, 40, 30])

    idx = topk(arr, k=2, largest=True, return_indices=True)
    vals = topk(arr, k=2, largest=True, return_indices=False)

    assert len(idx) == 2
    assert set(idx) == {1, 3}
    assert set(vals) == {40, 50}


def test_topk_smallest_values():
    """7. Test retrieving top-k smallest elements (largest=False)."""
    arr = np.array([100, 5, 20, 1, 50])
    vals = topk(arr, k=3, largest=False, return_indices=False)

    assert len(vals) == 3
    assert set(vals) == {1, 5, 20}


def test_topk_extreme_k_bounds():
    """8. Test topk edge cases where k=1 and k=len(arr)."""
    arr = np.array([8, 2, 15, 4])

    top_1 = topk(arr, k=1, largest=True, return_indices=False)
    np.testing.assert_array_equal(top_1, np.array([15]))

    top_all = topk(arr, k=len(arr), largest=True, return_indices=False)
    assert set(top_all) == {2, 4, 8, 15}


def test_topk_invalid_k():
    """9. Test topk raising ValueError when k < 1 or k > len(arr)."""
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="k must be between"):
        topk(arr, k=0)
    with pytest.raises(ValueError, match="k must be between"):
        topk(arr, k=4)


# =====================================================================
# Quickselect Tests (Tests 10 - 12)
# =====================================================================

def test_quickselect_kth_smallest():
    """10. Test quickselect for minimum (k=1), median (k=3), and maximum (k=5)."""
    arr = np.array([7, 10, 4, 3, 20])
    assert quickselect(arr, k=1, largest=False) == 3
    assert quickselect(arr, k=3, largest=False) == 7
    assert quickselect(arr, k=5, largest=False) == 20


def test_quickselect_kth_largest_with_duplicates():
    """11. Test quickselect finding k-th largest in an array with duplicate ties."""
    arr = np.array([5, 1, 5, 3, 5, 2])
    # Sorted descending: [5, 5, 5, 3, 2, 1]
    assert quickselect(arr, k=2, largest=True) == 5
    assert quickselect(arr, k=4, largest=True) == 3
    assert quickselect(arr, k=6, largest=True) == 1


def test_quickselect_invalid_k():
    """12. Test quickselect raising ValueError for out-of-bounds k."""
    arr = np.array([10, 20, 30])
    with pytest.raises(ValueError, match="k must be between"):
        quickselect(arr, k=0)
    with pytest.raises(ValueError, match="k must be between"):
        quickselect(arr, k=5)


# =====================================================================
# Binary Search & General Edge Case Tests (Tests 13 - 15)
# =====================================================================

def test_binary_search_scalar():
    """13. Test binary_search with scalar targets (both existing and missing)."""
    sorted_arr = np.array([10, 20, 30, 40, 50])

    idx_found, exists_found = binary_search(sorted_arr, 30)
    assert idx_found == 2
    assert exists_found is True

    idx_missing, exists_missing = binary_search(sorted_arr, 25)
    assert idx_missing == 2
    assert exists_missing is False


def test_binary_search_vector_and_out_of_bounds():
    """14. Test vectorized binary_search including targets smaller than min and larger than max."""
    sorted_arr = np.array([10, 20, 30])
    targets = np.array([5, 10, 25, 30, 100])

    indices, exists = binary_search(sorted_arr, targets)
    np.testing.assert_array_equal(indices, np.array([0, 0, 2, 2, 3]))
    np.testing.assert_array_equal(exists, np.array([False, True, False, True, False]))


def test_sort_search_non_contiguous_and_invalid_dimensions():
    """15. Test non-contiguous 1D slices and ensure 2D arrays raise ValueError."""
    base = np.array([50, 0, 10, 0, 30, 0, 20, 0, 40])
    strided = base[::2]  # [50, 10, 30, 20, 40]
    assert not strided.flags['C_CONTIGUOUS']

    np.testing.assert_array_equal(stable_sort(strided), np.array([10, 20, 30, 40, 50]))
    assert quickselect(strided, k=3) == 30

    # 2D inputs must raise ValueError across functions
    arr_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        stable_sort(arr_2d)
    with pytest.raises(ValueError):
        topk(arr_2d, k=1)
    with pytest.raises(ValueError):
        quickselect(arr_2d, k=1)
    with pytest.raises(ValueError):
        binary_search(arr_2d, 2)
import numpy as np
import pytest
from numcompute.sort_search import stable_sort, topk, quickselect

def test_stable_sort_ascending():
    """Test basic sorting in ascending order."""
    arr = np.array([3, 1, 4, 1, 5])
    sorted_arr = stable_sort(arr)
    np.testing.assert_array_equal(sorted_arr, np.array([1, 1, 3, 4, 5]))

def test_stable_sort_descending():
    """Test sorting in descending order."""
    arr = np.array([3, 1, 4, 1, 5])
    sorted_arr = stable_sort(arr, descending=True)
    np.testing.assert_array_equal(sorted_arr, np.array([5, 4, 3, 1, 1]))

def test_topk_largest_indices():
    """Test finding the indices of the top 2 largest elements."""
    arr = np.array([10, 50, 20, 40, 30])
    # Should find the indices of 50 and 40 (which are 1 and 3)
    idx = topk(arr, k=2, largest=True, return_indices=True)
    
    assert len(idx) == 2
    assert 1 in idx and 3 in idx

def test_quickselect_median():
    """Test finding the 3rd smallest element (median) in an array of 5."""
    arr = np.array([7, 10, 4, 3, 20])
    # 3rd smallest means k=3
    kth_element = quickselect(arr, k=3, largest=False)
    
    # Sorted it would be [3, 4, 7, 10, 20], so the 3rd is 7
    assert kth_element == 7
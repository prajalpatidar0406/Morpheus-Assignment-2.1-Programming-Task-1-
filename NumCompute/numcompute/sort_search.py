import numpy as np

def stable_sort(arr, descending=False):
    """
    Sort a 1D numpy array using stable sorting.
    
    Parameters
    ----------
    arr : np.ndarray
        1D array to be sorted, shape (n_samples,).
    descending : bool, default=False
        If True, sort in descending order.
        
    Returns
    -------
    np.ndarray
        The sorted 1D array.
        
    Complexity
    ----------
    Time: O(n_samples * log(n_samples))
    Space: O(n_samples)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("stable_sort expects a 1D array.")
        
    # Using stable sort wrapper as per guidelines
    sort_indices = np.argsort(arr, kind='stable')
    
    if descending:
        sort_indices = sort_indices[::-1]
        
    return arr[sort_indices]

def multi_key_sort(keys, descending=False):
    """
    Sort based on multiple 1D arrays (multi-key sorting).
    
    Parameters
    ----------
    keys : tuple or list of np.ndarray
        Tuple of 1D arrays to sort by, ordered from primary to least significant key.
    descending : bool, default=False
        If True, reverse the final sorted order.
        
    Returns
    -------
    np.ndarray
        The 1D array of indices that would sort the arrays.
        
    Complexity
    ----------
    Time: O(n_samples * log(n_samples))
    Space: O(n_samples)
    """
    for k in keys:
        if np.asarray(k).ndim != 1:
            raise ValueError("All keys in multi_key_sort must be 1D arrays.")
            
    # np.lexsort sorts sequentially. It applies sorting from the last array to the first.
    # We reverse the tuple so the first key provided is the primary sort key.
    sort_indices = np.lexsort(keys[::-1])
    
    if descending:
        sort_indices = sort_indices[::-1]
        
    return sort_indices

def topk(values, k, largest=True, return_indices=True):
    """
    Retrieve the top-k elements from a 1D array.
    Uses np.argpartition for an O(n) partial sort.
    
    Parameters
    ----------
    values : np.ndarray
        Input 1D array, shape (n_samples,).
    k : int
        Number of elements to retrieve.
    largest : bool, default=True
        If True, retrieve the largest elements. If False, the smallest.
    return_indices : bool, default=True
        If True, returns indices instead of values.
        
    Returns
    -------
    np.ndarray
        Array of top-k elements or indices, shape (k,). 
        Note: The returned k elements are NOT fully sorted among themselves.
        
    Complexity
    ----------
    Time: O(n_samples) average time.
    Space: O(k) for output
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"topk expects a 1D array. Received shape {arr.shape}.")
        
    n = len(arr)
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and the number of values ({n}).")

    if largest:
        # Move the largest k elements to the end of the array
        idx = np.argpartition(arr, -k)[-k:]
    else:
        # Move the smallest k elements to the start of the array
        idx = np.argpartition(arr, k - 1)[:k]
        
    if return_indices:
        return idx
    return arr[idx]

def quickselect(values, k, largest=False):
    """
    Finds the k-th smallest (or largest) element in an unordered 1D array.
    Implemented as an educational recursive function using NumPy boolean masks
    to avoid pure Python `for` loops.
    
    Parameters
    ----------
    values : np.ndarray
        Input 1D array, shape (n_samples,).
    k : int
        The rank of the element to find (1-indexed). e.g., k=1 means the absolute smallest/largest.
    largest : bool, default=False
        If True, returns the k-th largest element. If False, the k-th smallest.
        
    Returns
    -------
    scalar
        The k-th element.
        
    Complexity
    ----------
    Time: O(n_samples) average time, O(n_samples^2) worst case.
    Space: O(n_samples) for array slicing masks in the recursion stack.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")
        
    n = len(arr)
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and the number of values ({n}).")
        
    # If looking for the k-th largest, it is equivalent to the (n - k + 1)-th smallest
    if largest:
        k = n - k + 1

    # Choose a pivot (middle element strategy)
    pivot = arr[n // 2]
    
    # Vectorized boolean partitioning
    left = arr[arr < pivot]
    mid = arr[arr == pivot]
    right = arr[arr > pivot]
    
    # Recursive search based on partitioned sizes
    if k <= len(left):
        return quickselect(left, k, largest=False)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        # Subtract the length of the left and middle portions for the recursive call
        return quickselect(right, k - len(left) - len(mid), largest=False)

def binary_search(sorted_array, x):
    """
    Finds the insertion indices for targets in a sorted 1D array,
    and returns whether the target exactly exists in the array.
    
    Parameters
    ----------
    sorted_array : np.ndarray
        A 1D array that is ALREADY sorted in ascending order.
    x : scalar or np.ndarray
        The value(s) to search for.
        
    Returns
    -------
    tuple (insertion_indices, exists)
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
    insertion_indices = np.searchsorted(arr, targets, side='left')
    
    # Check for out-of-bounds indices (if target is greater than all elements)
    valid_mask = insertion_indices < len(arr)
    
    # Check if the value at the valid insertion index exactly matches the target
    exists = np.zeros_like(targets, dtype=bool)
    exists[valid_mask] = (arr[insertion_indices[valid_mask]] == targets[valid_mask])
    
    # Return scalars if the original input was a scalar, else return arrays
    if np.isscalar(x):
        return int(insertion_indices[0]), bool(exists[0])
        
    return insertion_indices, exists
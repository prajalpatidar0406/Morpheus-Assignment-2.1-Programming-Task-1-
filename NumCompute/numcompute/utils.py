"""Reusable numeric helpers for distances, activations, selection, and batching."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute Euclidean distance between equal-shaped numeric arrays.

    Returns a scalar; shape mismatch raises ValueError. Complexity is O(n)
    time and O(n) temporary space for n elements.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.linalg.norm(left - right))


def manhattan_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute Manhattan distance between equal-shaped numeric arrays.

    Returns a scalar; shape mismatch raises ValueError. Complexity is O(n)
    time and O(n) temporary space.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.sum(np.abs(left - right)))


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute cosine similarity of equal-shaped numeric arrays.

    Returns zero if either vector has zero norm. Shape mismatch raises
    ValueError. Complexity is O(n) time and O(n) temporary space.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denominator == 0 else float(np.dot(left.ravel(), right.ravel()) / denominator)


def relu(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Apply ReLU elementwise and return an array with x's shape."""
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def sigmoid(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Apply stable logistic sigmoid elementwise, preserving the input shape."""
    values = np.asarray(x, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def softmax(x: npt.ArrayLike, axis: int = -1) -> npt.NDArray[np.float64]:
    """Apply stable softmax along axis; output has the same shape as x.

    Empty inputs raise ValueError. Positive-infinity logits share probability
    uniformly; an all-negative-infinity slice also maps to a uniform vector.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("x must not be empty.")
    maximum = np.max(values, axis=axis, keepdims=True)
    positive_infinity = np.isposinf(values)
    positive_count = positive_infinity.sum(axis=axis, keepdims=True)
    negative_infinity_only = np.isneginf(maximum)
    shifted = np.where(
        positive_infinity,
        0.0,
        values - np.where(np.isfinite(maximum), maximum, 0.0),
    )
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=axis, keepdims=True)
    probabilities = np.divide(
        exponentials,
        denominator,
        out=np.zeros_like(values),
        where=denominator > 0,
    )
    inf_probabilities = np.divide(
        positive_infinity,
        positive_count,
        out=np.zeros_like(values),
        where=positive_count > 0,
    )
    uniform = np.full_like(values, 1.0 / values.shape[axis])
    probabilities = np.where(positive_count > 0, inf_probabilities, probabilities)
    probabilities = np.where(negative_infinity_only, uniform, probabilities)
    return np.where(np.isnan(maximum), np.nan, probabilities)


def logsumexp(x: npt.ArrayLike, axis: int | None = None, keepdims: bool = False) -> npt.NDArray[np.float64] | np.float64:
    """Compute log(sum(exp(x))) stably along the selected axis.

    Positive- and negative-infinite input maxima return their corresponding
    infinite result rather than generating ``inf - inf`` NaNs.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("x must not be empty.")
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_shift = np.where(np.isfinite(maximum), maximum, 0.0)
    shifted = np.where(np.isfinite(maximum), values - finite_shift, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        finite_result = finite_shift + np.log(
            np.sum(np.exp(shifted), axis=axis, keepdims=True)
        )
    result = np.where(np.isposinf(maximum), np.inf, finite_result)
    result = np.where(np.isneginf(maximum), -np.inf, result)
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    elif not keepdims and axis is None:
        result = np.squeeze(result)
    return result


def topk_indices(values: npt.ArrayLike, k: int, largest: bool = True) -> npt.NDArray[np.intp]:
    """Return ordered indices of k largest/smallest values in a 1D array.

    Requires ``1 <= k <= len(values)``. Expected selection work is O(n), plus
    O(k log k) sorting, with O(n) workspace.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not isinstance(k, (int, np.integer)) or k < 1 or k > array.size:
        raise ValueError("k must be between 1 and the number of values.")
    split = array.size - k if largest else k - 1
    candidates = np.argpartition(array, split)[-k:] if largest else np.argpartition(array, split)[:k]
    order = np.argsort(array[candidates], kind="stable")
    if largest:
        order = order[::-1]
    return candidates[order]


def batches(data: npt.ArrayLike, batch_size: int) -> Iterator[npt.NDArray[np.generic]]:
    """Yield slices along data's first axis, including a final partial batch.

    Requires data with at least one dimension and positive integer batch_size.
    Each yielded array has at most ``batch_size`` rows and preserves trailing
    dimensions. Slicing itself is O(1) view construction per batch.
    """
    values = np.asarray(data)
    if values.ndim == 0:
        raise ValueError("data must have at least one dimension.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]

import numpy as np
import pytest

from numcompute.utils import (
    batches,
    cosine_similarity,
    euclidean_distance,
    logsumexp,
    manhattan_distance,
    relu,
    sigmoid,
    softmax,
    topk_indices,
)


def test_distances():
    assert euclidean_distance([0, 0], [3, 4]) == 5.0
    assert manhattan_distance([0, 0], [3, 4]) == 7.0


def test_distance_rejects_different_shapes():
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance([1], [1, 2])


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert cosine_similarity([1, 0], [2, 0]) == 1.0


def test_activation_functions():
    np.testing.assert_array_equal(relu([-1, 0, 2]), [0.0, 0.0, 2.0])
    np.testing.assert_allclose(sigmoid([0]), [0.5])
    assert np.isfinite(sigmoid([-1000, 1000])).all()


def test_softmax_normalizes_per_row():
    result = softmax([[1, 2], [3, 4]])
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])


def test_logsumexp_matches_stable_expected_value():
    assert np.isclose(logsumexp([1000.0, 1000.0]), 1000.0 + np.log(2.0))


def test_logsumexp_and_softmax_handle_infinite_values():
    assert logsumexp([np.inf, 0.0]) == np.inf
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    np.testing.assert_array_equal(softmax([np.inf, 3.0, np.inf]), [0.5, 0.0, 0.5])
    np.testing.assert_allclose(softmax([-np.inf, -np.inf]), [0.5, 0.5])


def test_topk_indices_returns_largest_sorted_first():
    indices = topk_indices([2, 8, 3, 7], 2)
    np.testing.assert_array_equal(indices, [1, 3])


def test_batches_include_final_partial_batch():
    result = list(batches(np.arange(5), 2))
    assert [part.tolist() for part in result] == [[0, 1], [2, 3], [4]]


def test_batches_reject_nonpositive_size():
    with pytest.raises(ValueError, match="positive integer"):
        list(batches([1, 2], 0))

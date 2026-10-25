import numpy as np
import pytest

from numcompute.stats import (
    histogram,
    maximum,
    mean,
    median,
    minimum,
    percentile,
    quantile,
    std
)


def test_mean_of_a_simple_list() -> None:
    assert mean([1, 2, 3, 4]) == pytest.approx(2.5)


def test_mean_reduces_along_the_requested_axis() -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]])

    np.testing.assert_allclose(mean(data, axis=0), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(mean(data, axis=1), [2.0, 5.0])


def test_mean_propagates_nan_unless_skipna_is_set() -> None:
    assert np.isnan(mean([1, np.nan, 3]))
    assert mean([1, np.nan, 3], skipna=True) == pytest.approx(2.0)


def test_median_handles_odd_and_even_lengths() -> None:
    assert median([3, 1, 2]) == pytest.approx(2.0)
    assert median([4, 1, 3, 2]) == pytest.approx(2.5)


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_matches_numpy_for_each_ddof(ddof: int) -> None:
    data = [1, 2, 3, 4]

    assert std(data, ddof=ddof) == pytest.approx(np.std(data, ddof=ddof))


def test_minimum_and_maximum_of_a_simple_list() -> None:
    assert minimum([3, 1, 2]) == 1
    assert maximum([3, 1, 2]) == 3


def test_quantile_and_percentile_match_known_values() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    np.testing.assert_allclose(quantile([1, 2, 3, 4], [0.25, 0.75]), [1.75, 3.25])
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_histogram_returns_counts_and_bin_edges() -> None:
    counts, edges = histogram([1, 2, 2, 3, 3, 3], bins=[0, 2, 4])

    np.testing.assert_array_equal(counts, [1, 5])
    np.testing.assert_allclose(edges, [0, 2, 4])


def test_invalid_and_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean([])

    with pytest.raises(TypeError, match="real numeric values"):
        mean(["a", "b"])

    with pytest.raises(ValueError, match="ddof must be a non-negative integer"):
        std([1, 2, 3], ddof=-1)

    with pytest.raises(ValueError, match="q must be between 0 and 1"):
        quantile([1, 2, 3], 2)

    with pytest.raises(ValueError, match="infinite values"):
        histogram([1, np.inf])
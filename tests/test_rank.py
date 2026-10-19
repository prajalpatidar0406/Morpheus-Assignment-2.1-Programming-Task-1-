import numpy as np
import pytest
from numcompute.rank import Ranker


# ---------------------------------------------------------
# FIXTURE
# ---------------------------------------------------------

@pytest.fixture
def ranker():
    return Ranker()


# ---------------------------------------------------------
# VALID INPUT TESTS — RANK
# ---------------------------------------------------------

def test_rank_average_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="average")
    expected = np.array([1.0, 2.5, 2.5, 4.0])
    assert np.allclose(result, expected)


def test_rank_dense_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="dense")
    expected = np.array([1, 2, 2, 3])
    assert np.allclose(result, expected)


def test_rank_ordinal_basic(ranker):
    data = [10, 20, 20, 30]
    result = ranker.rank(data, method="ordinal")
    expected = np.array([1, 2, 3, 4])
    assert np.allclose(result, expected)


# ---------------------------------------------------------
# MIXED DATA TESTS — RANK
# ---------------------------------------------------------

def test_rank_mixed_types(ranker):
    data = [10, "20", "abc", None, 5.5]
    result = ranker.rank(data, method="average")
    assert len(result) == 5
    # "abc" and None become NaN → sentinel → lowest rank
    assert result[2] == 1.0
    assert result[3] == 1.0


def test_rank_with_nan(ranker):
    data = [10, np.nan, 5]
    result = ranker.rank(data, method="dense")
    assert len(result) == 3
    assert result[1] == 1  # NaN → sentinel → lowest rank


# ---------------------------------------------------------
# EXCEPTION HANDLING — RANK
# ---------------------------------------------------------

def test_rank_invalid_method(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([1, 2, 3], method="weird")
    assert "Invalid ranking method" in str(exc.value)


def test_rank_invalid_data_none(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank(None)
    assert "cannot be None" in str(exc.value)


def test_rank_invalid_data_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank("hello")
    assert "single string" in str(exc.value)


def test_rank_invalid_data_multidimensional(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([[1, 2], [3, 4]])
    assert "must be 1D" in str(exc.value)


def test_rank_empty_array(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.rank([])
    assert "empty" in str(exc.value)


# ---------------------------------------------------------
# VALID INPUT TESTS — PERCENTILE
# ---------------------------------------------------------

def test_percentile_linear(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 50, interpolation="linear")
    assert result == 25.0


def test_percentile_lower(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="lower")
    assert result == 10


def test_percentile_higher(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="higher")
    assert result == 20


def test_percentile_midpoint(ranker):
    data = [10, 20, 30, 40]
    result = ranker.percentile(data, 25, interpolation="midpoint")
    assert result == 15


# ---------------------------------------------------------
# MIXED DATA TESTS — PERCENTILE
# ---------------------------------------------------------

def test_percentile_mixed_types(ranker):
    data = [10, "20", "abc", None, 5.5]
    result = ranker.percentile(data, 50)
    assert np.isclose(result, np.median([10, 20, 5.5]))


def test_percentile_with_nan(ranker):
    data = [10, np.nan, 20]
    result = ranker.percentile(data, 50)
    assert result == 15.0


# ---------------------------------------------------------
# EXCEPTION HANDLING — PERCENTILE
# ---------------------------------------------------------

def test_percentile_invalid_interpolation(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], 50, interpolation="weird")
    assert "Invalid interpolation" in str(exc.value)


def test_percentile_invalid_q_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], "abc")
    assert "numeric" in str(exc.value)


def test_percentile_invalid_q_range(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([1, 2, 3], 150)
    assert "between 0 and 100" in str(exc.value)


def test_percentile_invalid_data_none(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile(None, 50)
    assert "cannot be None" in str(exc.value)


def test_percentile_invalid_data_string(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile("hello", 50)
    assert "single string" in str(exc.value)


def test_percentile_invalid_data_multidimensional(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([[1, 2], [3, 4]], 50)
    assert "must be 1D" in str(exc.value)


def test_percentile_empty_array(ranker):
    with pytest.raises(ValueError) as exc:
        ranker.percentile([], 50)
    assert "empty" in str(exc.value)

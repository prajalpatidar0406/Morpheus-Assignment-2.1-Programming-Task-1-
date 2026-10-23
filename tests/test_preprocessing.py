import numpy as np
import pytest
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


# =====================================================================
# StandardScaler Tests (Tests 1 - 4)
# =====================================================================

def test_standard_scaler_normal():
    """1. Test standard scaling on a normal 1D array."""
    X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.mean(X_scaled), 0.0)
    assert np.isclose(np.std(X_scaled), 1.0)


def test_standard_scaler_constant():
    """2. Test scaling a constant array (zero variance should not divide by zero)."""
    X = np.array([5.0, 5.0, 5.0, 5.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert scaler.scale_ == 1.0
    np.testing.assert_allclose(X_scaled, np.zeros_like(X))


def test_standard_scaler_separate_train_test():
    """3. Test that transform uses training parameters on unseen test data."""
    X_train = np.array([0.0, 10.0])  # mean = 5.0, std = 5.0
    X_test = np.array([15.0, 25.0])

    scaler = StandardScaler(epsilon=0.0)
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.testing.assert_allclose(X_test_scaled, np.array([2.0, 4.0]))


def test_standard_scaler_unfitted_error():
    """4. Test that calling transform before fit raises a ValueError."""
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted yet"):
        scaler.transform(np.array([1.0, 2.0, 3.0]))


# =====================================================================
# MinMaxScaler Tests (Tests 5 - 7)
# =====================================================================

def test_minmax_scaler_default_range():
    """5. Test MinMaxScaler scaling to the default [0, 1] range."""
    X = np.array([-10.0, 0.0, 10.0, 30.0])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.min(X_scaled), 0.0)
    assert np.isclose(np.max(X_scaled), 1.0)
    np.testing.assert_allclose(X_scaled, np.array([0.0, 0.25, 0.5, 1.0]))


def test_minmax_scaler_custom_range():
    """6. Test MinMaxScaler scaling to a custom feature_range=(-1, 1)."""
    X = np.array([0.0, 5.0, 10.0])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled, np.array([-1.0, 0.0, 1.0]))


def test_minmax_scaler_constant():
    """7. Test MinMaxScaler on a constant array (range = 0 should not crash)."""
    X = np.array([7.0, 7.0, 7.0])
    scaler = MinMaxScaler(feature_range=(2.0, 5.0))
    X_scaled = scaler.fit_transform(X)

    assert scaler.data_range_ == 1.0
    np.testing.assert_allclose(X_scaled, np.array([2.0, 2.0, 2.0]))


# =====================================================================
# Imputer Tests (Tests 8 - 12)
# =====================================================================

def test_imputer_mean_nan():
    """8. Test Imputer replacing np.nan with the mean of valid elements."""
    X = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    imputer = Imputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([1.0, 3.0, 3.0, 3.0, 5.0]))


def test_imputer_median_nan():
    """9. Test Imputer replacing np.nan with the median (robust to outliers)."""
    X = np.array([1.0, 2.0, 100.0, np.nan])
    imputer = Imputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Median of [1.0, 2.0, 100.0] is 2.0
    np.testing.assert_allclose(X_imputed, np.array([1.0, 2.0, 100.0, 2.0]))


def test_imputer_custom_missing_values_placeholder():
    """10. Test Imputer when missing values are passed as a custom placeholder (-999.0)."""
    X = np.array([10.0, -999.0, 20.0, -999.0])
    imputer = Imputer(strategy='mean', missing_values=-999.0)
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([10.0, 15.0, 20.0, 15.0]))


def test_imputer_constant_and_all_missing_fallback():
    """11. Test Imputer constant strategy and fallback when all values are missing."""
    X_all_nan = np.array([np.nan, np.nan, np.nan])
    imputer = Imputer(strategy='mean', fill_value=-1.0)
    X_imputed = imputer.fit_transform(X_all_nan)

    np.testing.assert_allclose(X_imputed, np.array([-1.0, -1.0, -1.0]))


def test_imputer_invalid_strategy():
    """12. Test Imputer raising ValueError when initialized with an unknown strategy."""
    with pytest.raises(ValueError, match="Strategy must be"):
        Imputer(strategy='most_frequent')


# =====================================================================
# OneHotEncoder & General Edge Case Tests (Tests 13 - 15)
# =====================================================================

def test_onehot_encoder_normal():
    """13. Test OneHotEncoder converting 1D integer categories into a 2D binary matrix."""
    X = np.array([0, 2, 1, 0])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    assert X_encoded.shape == (4, 3)
    np.testing.assert_array_equal(X_encoded, expected)


def test_onehot_encoder_invalid_inputs():
    """14. Test OneHotEncoder rejecting floats, negative categories, and unseen categories."""
    encoder = OneHotEncoder()

    # Rejects floats
    with pytest.raises(ValueError, match="integer categorical data"):
        encoder.fit(np.array([0.0, 1.5, 2.0]))

    # Rejects negative integers
    with pytest.raises(ValueError, match="non-negative integers"):
        encoder.fit(np.array([-1, 0, 1]))

    # Rejects unseen categories during transform
    encoder.fit(np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(np.array([0, 1, 2]))


def test_preprocessing_non_contiguous_and_invalid_dimensions():
    """15. Test non-contiguous 1D array strides and rejection of 2D arrays."""
    # Non-contiguous slice (step=2)
    base = np.array([10.0, 99.0, 20.0, 99.0, 30.0, 99.0])
    X_strided = base[::2]  # [10.0, 20.0, 30.0]
    assert not X_strided.flags['C_CONTIGUOUS']

    scaler = MinMaxScaler()
    np.testing.assert_allclose(scaler.fit_transform(X_strided), np.array([0.0, 0.5, 1.0]))

    # 2D array must raise ValueError
    X_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="X must be a 1D array"):
        scaler.fit(X_2d)
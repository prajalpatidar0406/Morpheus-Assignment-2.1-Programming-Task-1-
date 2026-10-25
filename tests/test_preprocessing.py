"""Check scaling, missing-value imputation, and categorical encoding."""

import numpy as np
import pytest
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


# StandardScaler


def test_standard_scaler_normal():
    """Standardize values so their mean is zero and standard deviation is one."""
    X = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.mean(X_scaled), 0.0)
    assert np.isclose(np.std(X_scaled), 1.0)


def test_standard_scaler_constant():
    """A constant feature should become zeros without dividing by zero."""
    X = np.array([5.0, 5.0, 5.0, 5.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert scaler.scale_ == 1.0
    np.testing.assert_allclose(X_scaled, np.zeros_like(X))


def test_standard_scaler_separate_train_test():
    """Transform new values with statistics learned from the training data."""
    X_train = np.array([0.0, 10.0])  # mean = 5.0, std = 5.0
    X_test = np.array([15.0, 25.0])

    scaler = StandardScaler(epsilon=0.0)
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.testing.assert_allclose(X_test_scaled, np.array([2.0, 4.0]))


def test_standard_scaler_unfitted_error():
    """Reject transforming data before the scaler has been fitted."""
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted yet"):
        scaler.transform(np.array([1.0, 2.0, 3.0]))


# MinMaxScaler


def test_minmax_scaler_default_range():
    """Scale the observed minimum and maximum to the default range."""
    X = np.array([-10.0, 0.0, 10.0, 30.0])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.min(X_scaled), 0.0)
    assert np.isclose(np.max(X_scaled), 1.0)
    np.testing.assert_allclose(X_scaled, np.array([0.0, 0.25, 0.5, 1.0]))


def test_minmax_scaler_custom_range():
    """Allow scaling into a caller-selected range."""
    X = np.array([0.0, 5.0, 10.0])
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled, np.array([-1.0, 0.0, 1.0]))


def test_minmax_scaler_constant():
    """A constant feature should map to the lower end of the selected range."""
    X = np.array([7.0, 7.0, 7.0])
    scaler = MinMaxScaler(feature_range=(2.0, 5.0))
    X_scaled = scaler.fit_transform(X)

    assert scaler.data_range_ == 1.0
    np.testing.assert_allclose(X_scaled, np.array([2.0, 2.0, 2.0]))


# Imputer


def test_imputer_mean_nan():
    """Replace missing values with the mean of observed values."""
    X = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    imputer = Imputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([1.0, 3.0, 3.0, 3.0, 5.0]))


def test_imputer_median_nan():
    """Use the median when outliers could skew the replacement value."""
    X = np.array([1.0, 2.0, 100.0, np.nan])
    imputer = Imputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Median of [1.0, 2.0, 100.0] is 2.0
    np.testing.assert_allclose(X_imputed, np.array([1.0, 2.0, 100.0, 2.0]))


def test_imputer_custom_missing_values_placeholder():
    """Recognize a caller-provided marker for missing numeric values."""
    X = np.array([10.0, -999.0, 20.0, -999.0])
    imputer = Imputer(strategy="mean", missing_values=-999.0)
    X_imputed = imputer.fit_transform(X)

    np.testing.assert_allclose(X_imputed, np.array([10.0, 15.0, 20.0, 15.0]))


def test_imputer_recognizes_string_and_none_missing_values():
    values = np.array(["  NaN ", None, "kept"], dtype=object)
    imputer = Imputer(strategy="constant", fill_value="missing")

    np.testing.assert_array_equal(
        imputer.fit_transform(values), ["missing", "missing", "kept"]
    )


def test_imputer_constant_and_all_missing_fallback():
    """Use the constant value both by strategy and when no observations exist."""
    X_all_nan = np.array([np.nan, np.nan, np.nan])
    imputer = Imputer(strategy="mean", fill_value=-1.0)
    X_imputed = imputer.fit_transform(X_all_nan)

    np.testing.assert_allclose(X_imputed, np.array([-1.0, -1.0, -1.0]))


def test_imputer_invalid_strategy():
    """Reject strategies that the imputer does not support."""
    with pytest.raises(ValueError, match="Strategy must be"):
        Imputer(strategy="most_frequent")


# One-hot encoding and input edge cases


def test_onehot_encoder_normal():
    """Turn integer categories into one binary indicator column per category."""
    X = np.array([0, 2, 1, 0])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    assert X_encoded.shape == (4, 3)
    np.testing.assert_array_equal(X_encoded, expected)


def test_onehot_encoder_supports_object_integer_categories():
    values = np.array([10, 1000000, 10], dtype=object)

    encoded = OneHotEncoder().fit_transform(values)

    assert encoded.shape == (3, 2)
    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1], [1, 0]])


def test_onehot_encoder_vectorizes_string_category_lookup():
    encoder = OneHotEncoder().fit(["red", "blue", "red"])

    encoded = encoder.transform(["blue", "red"])

    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(["green"])


def test_onehot_encoder_invalid_inputs():
    """Reject unsupported numeric categories and categories not seen during fitting."""
    encoder = OneHotEncoder()

    # Categories must be integers, not values that only happen to look numeric.
    with pytest.raises(ValueError, match="integer categorical data"):
        encoder.fit(np.array([0.0, 1.5, 2.0]))

    # Negative values are outside the encoder's supported category range.
    with pytest.raises(ValueError, match="non-negative integers"):
        encoder.fit(np.array([-1, 0, 1]))

    # Transform cannot assign a column to a category that was never learned.
    encoder.fit(np.array([0, 1]))
    with pytest.raises(ValueError, match="unknown categories"):
        encoder.transform(np.array([0, 1, 2]))

    encoder.fit(np.array(["red", "blue"]))
    with pytest.raises(ValueError, match="string categorical data"):
        encoder.transform(np.array([1, 2]))


def test_preprocessing_non_contiguous_and_invalid_dimensions():
    """Handle strided input while still rejecting arrays with multiple dimensions."""
    # This view skips every other item rather than copying into a new array.
    base = np.array([10.0, 99.0, 20.0, 99.0, 30.0, 99.0])
    X_strided = base[::2]  # [10.0, 20.0, 30.0]
    assert not X_strided.flags["C_CONTIGUOUS"]

    scaler = MinMaxScaler()
    np.testing.assert_allclose(
        scaler.fit_transform(X_strided), np.array([0.0, 0.5, 1.0])
    )

    # The transformer accepts one feature at a time.
    X_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="X must be a 1D array"):
        scaler.fit(X_2d)

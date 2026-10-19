import numpy as np
import pytest
from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer

def test_standard_scaler_normal():
    """Test standard scaling on a normal 1D array."""
    X = np.array([1.0, 2.0, 3.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Mean should be 0, Std should be 1
    assert np.isclose(np.mean(X_scaled), 0.0)
    assert np.isclose(np.std(X_scaled), 1.0)

def test_standard_scaler_constant():
    """Test scaling an array where all values are the same (should not divide by zero)."""
    X = np.array([5.0, 5.0, 5.0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Because of our epsilon, it shouldn't crash, and should return 0s
    assert np.allclose(X_scaled, 0.0)

def test_imputer_mean():
    """Test filling NaNs with the mean of the array."""
    X = np.array([1.0, np.nan, 3.0])
    imputer = Imputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # The mean of [1.0, 3.0] is 2.0
    np.testing.assert_array_equal(X_imputed, np.array([1.0, 2.0, 3.0]))

def test_imputer_invalid_dimensions():
    """Test that our input validation catches 2D arrays."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    imputer = Imputer()
    
    # Expect a ValueError because we enforce 1D arrays
    with pytest.raises(ValueError):
        imputer.fit(X)
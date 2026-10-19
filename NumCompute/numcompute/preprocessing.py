import numpy as np

def _check_1d_array(X):
    """
    Ensure the input is a 1D NumPy array.
    
    Parameters
    ----------
    X : array_like
        Input data.
        
    Returns
    -------
    np.ndarray
        A 1D NumPy array.
        
    Raises
    ------
    ValueError
        If the input is not 1D.
    """
    X_arr = np.array(X, copy=False)
    if X_arr.ndim != 1:
        raise ValueError(f"X must be a 1D array. Received shape {X_arr.shape}.")
    return X_arr

def _check_is_fitted(estimator, attributes):
    """Check if the estimator has been fitted by verifying necessary attributes."""
    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(f"This {estimator.__class__.__name__} instance is not fitted yet. Call 'fit' first.")

#Remove base estimator
class BaseEstimator:
    """Base class for all estimators in the NumCompute framework."""
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data.
            
        Returns
        -------
        np.ndarray
            Transformed array.
        """
        return self.fit(X).transform(X)

class StandardScaler(BaseEstimator):
    """
    Standardize 1D features by removing the mean and scaling to unit variance.
    """
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data array, shape (n_samples,).
            
        Returns
        -------
        self : StandardScaler
            The fitted scaler.
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        
        # Calculate scalar mean and std, ignoring NaNs to ensure stability
        self.mean_ = np.nanmean(X_arr)
        self.scale_ = np.nanstd(X_arr)
        
        # Prevent division by zero if the feature is completely constant
        if self.scale_ == 0.0:
            self.scale_ = 1.0
            
        return self

    def transform(self, X):
        """
        Apply standardization by centering and scaling.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data array, shape (n_samples,).
            
        Returns
        -------
        np.ndarray
            Standardized 1D array, shape (n_samples,).
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ['mean_', 'scale_'])
        X_arr = _check_1d_array(X)
        
        return (X_arr - self.mean_) / (self.scale_ + self.epsilon)

class MinMaxScaler(BaseEstimator):
    """
    Transform 1D features by scaling to a given range [0, 1].
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data array, shape (n_samples,).
            
        Returns
        -------
        self : MinMaxScaler
            The fitted scaler.
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        
        self.data_min_ = np.nanmin(X_arr)
        self.data_max_ = np.nanmax(X_arr)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Prevent division by zero for constant features
        if self.data_range_ == 0.0:
            self.data_range_ = 1.0
            
        return self

    def transform(self, X):
        """
        Scale features of X according to feature_range.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data array, shape (n_samples,).
            
        Returns
        -------
        np.ndarray
            Scaled 1D array, shape (n_samples,).
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ['data_min_', 'data_range_'])
        X_arr = _check_1d_array(X)
            
        X_std = (X_arr - self.data_min_) / self.data_range_
        min_val, max_val = self.feature_range
        return X_std * (max_val - min_val) + min_val

class Imputer(BaseEstimator):
    """
    Imputation transformer for completing missing values (NaNs) in a 1D array.
    """
    def __init__(self, strategy='mean', fill_value=0.0):
        if strategy not in ['mean', 'median', 'constant']:
            raise ValueError("Strategy must be 'mean', 'median', or 'constant'")
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        """
        Learn the imputation scalar value from the training data.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data with missing values, shape (n_samples,).
            
        Returns
        -------
        self : Imputer
            The fitted imputer.
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X_arr)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X_arr)
        elif self.strategy == 'constant':
            self.statistics_ = float(self.fill_value)
            
        # Handle case where the entire 1D array is NaN
        if np.isnan(self.statistics_):
            self.statistics_ = 0.0
            
        return self

    def transform(self, X):
        """
        Replace missing values with the learned statistic.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D data with missing values, shape (n_samples,).
            
        Returns
        -------
        np.ndarray
            Imputed 1D array, shape (n_samples,).
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples)
        """
        _check_is_fitted(self, ['statistics_'])
        X_arr = _check_1d_array(X).astype(float, copy=True)
        
        # Vectorized replacement of NaNs without Python loops
        mask = np.isnan(X_arr)
        return np.where(mask, self.statistics_, X_arr)

class OneHotEncoder(BaseEstimator):
    """
    Encode a 1D categorical integer feature array into a 2D one-hot matrix.
    Expects input to be non-negative integer categories.
    """
    def fit(self, X):
        """
        Determine the maximum number of categories.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D categorical data array, shape (n_samples,).
            
        Returns
        -------
        self : OneHotEncoder
            The fitted encoder.
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(1)
        """
        X_arr = _check_1d_array(X)
        
        if not np.issubdtype(X_arr.dtype, np.integer):
            raise ValueError("OneHotEncoder expects integer categorical data.")
        if np.any(X_arr < 0):
            raise ValueError("Categories must be non-negative integers.")
            
        # Determine number of categories (0-indexed, so we add 1 to the max value)
        self.categories_ = np.max(X_arr) + 1
        return self

    def transform(self, X):
        """
        Transform the 1D data to a 2D one-hot encoded matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Input 1D categorical data array, shape (n_samples,).
            
        Returns
        -------
        np.ndarray
            One-hot encoded 2D array, shape (n_samples, n_categories).
            
        Raises
        ------
        ValueError
            If unseen categories are encountered.
            
        Complexity
        ----------
        Time: O(n_samples)
        Space: O(n_samples * n_categories)
        """
        _check_is_fitted(self, ['categories_'])
        X_arr = _check_1d_array(X)
        
        if np.any(X_arr >= self.categories_) or np.any(X_arr < 0):
            raise ValueError("Found unknown categories during transform.")
            
        # Instant 1-hot creation via advanced NumPy indexing mapping to an identity matrix
        return np.eye(self.categories_)[X_arr]
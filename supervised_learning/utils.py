"""
Utility Functions for Supervised Learning
==========================================

Data preprocessing and evaluation utilities.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into train and test subsets.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Features
    y : ndarray of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : ndarrays
        Split data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Attributes:
    -----------
    mean_ : ndarray
        Mean of training data
    scale_ : ndarray
        Standard deviation of training data
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        """
        Compute mean and standard deviation.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Standardize data.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_scaled : ndarray
            Standardized data
        """
        # Avoid division by zero by adding small epsilon to scale
        scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return (X - self.mean_) / scale
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_scaled : ndarray
            Standardized data
        """
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """
    Scale features to a given range (default [0, 1]).
    
    Parameters:
    -----------
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data
    
    Attributes:
    -----------
    min_ : ndarray
        Minimum value per feature in training data
    max_ : ndarray
        Maximum value per feature in training data
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        
    def fit(self, X):
        """
        Compute minimum and maximum.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Scale features to range.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_scaled : ndarray
            Scaled data
        """
        # Avoid division by zero for constant features
        data_range = self.max_ - self.min_
        data_range = np.where(data_range == 0, 1.0, data_range)
        X_std = (X - self.min_) / data_range
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_scaled : ndarray
            Scaled data
        """
        self.fit(X)
        return self.transform(X)


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    accuracy : float
        Accuracy score
    """
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    
    Parameters:
    -----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
        
    Returns:
    --------
    mse : float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Calculate R² score.
    
    Parameters:
    -----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
        
    Returns:
    --------
    r2 : float
        R² score
    """
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Handle case where all y values are identical
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else float('-inf')
    return 1 - (ss_res / ss_tot)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for classification.
    
    Supports both binary and multi-class classification.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix where element [i, j] is the count of samples
        with true label i and predicted label j.
        For binary classification: [[TN, FP], [FN, TP]]
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix

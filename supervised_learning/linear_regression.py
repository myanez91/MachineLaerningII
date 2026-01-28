"""
Linear Regression Implementation
=================================

Simple implementation of linear regression for supervised learning.
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    
    Attributes:
    -----------
    weights : ndarray
        Coefficients of the linear model
    bias : float
        Intercept of the linear model
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
        y : ndarray of shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)

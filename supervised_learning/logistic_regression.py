"""
Logistic Regression Implementation
===================================

Implementation of logistic regression for binary classification.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression for binary classification using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    
    Attributes:
    -----------
    weights : ndarray
        Coefficients of the model
    bias : float
        Intercept of the model
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values (0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear combination
            linear_pred = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            predictions = self._sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        probabilities : ndarray of shape (n_samples,)
            Predicted probabilities
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
        y : ndarray of shape (n_samples,)
            True values
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

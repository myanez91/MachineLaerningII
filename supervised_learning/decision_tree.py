"""
Decision Tree Implementation
=============================

Simple implementation of a decision tree classifier.
"""

import numpy as np
from collections import Counter


class Node:
    """
    A node in the decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier.
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    
    Attributes:
    -----------
    root : Node
        Root node of the decision tree
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        """
        Build the decision tree.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        """
        self.root = self._build_tree(X, y)
        return self
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, n_features):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
                    
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain for a split."""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Weighted child entropy
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """Calculate entropy."""
        unique, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _most_common_label(self, y):
        """Return most common label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make a prediction for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
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

"""
Decision Tree Example
======================

This example demonstrates how to use the DecisionTreeClassifier implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from supervised_learning.decision_tree import DecisionTreeClassifier
from supervised_learning.utils import train_test_split, accuracy_score, confusion_matrix


def main():
    print("Decision Tree Classifier Example")
    print("=" * 50)
    
    # Generate synthetic multi-class classification data
    np.random.seed(42)
    
    # Class 0
    X0 = np.random.randn(40, 2) + np.array([2, 2])
    y0 = np.zeros(40, dtype=int)
    
    # Class 1
    X1 = np.random.randn(40, 2) + np.array([5, 5])
    y1 = np.ones(40, dtype=int)
    
    # Class 2
    X2 = np.random.randn(40, 2) + np.array([8, 2])
    y2 = np.full(40, 2, dtype=int)
    
    # Combine
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Parameters:")
    print(f"  Max Depth: {model.max_depth}")
    print(f"  Min Samples Split: {model.min_samples_split}")
    
    print(f"\nTraining Performance:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()

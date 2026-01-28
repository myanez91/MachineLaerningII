"""
Logistic Regression Example
============================

This example demonstrates how to use the LogisticRegression implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from supervised_learning.logistic_regression import LogisticRegression
from supervised_learning.utils import train_test_split, StandardScaler, accuracy_score, confusion_matrix


def main():
    print("Logistic Regression Example")
    print("=" * 50)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    
    # Class 0
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)
    
    # Class 1
    X1 = np.random.randn(50, 2) + np.array([5, 5])
    y1 = np.ones(50)
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    
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

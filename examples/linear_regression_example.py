"""
Linear Regression Example
==========================

This example demonstrates how to use the LinearRegression implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from supervised_learning.linear_regression import LinearRegression
from supervised_learning.utils import train_test_split, StandardScaler, mean_squared_error, r2_score


def main():
    print("Linear Regression Example")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    
    print(f"\nTraining Performance:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()

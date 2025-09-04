"""
Multi-Layer Perceptron (MLP) for ETA Prediction
Basic neural network with minimal complexity to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_and_prepare_data():
    """Load and prepare the travel dataset"""
    # Load data
    df = pd.read_csv('data/travel_data.csv')
    
    # Encode categorical variable (TravelMode)
    le = LabelEncoder()
    df['TravelMode_encoded'] = le.fit_transform(df['TravelMode'])
    
    # Features and target
    X = df[['Distance', 'Speed', 'TravelMode_encoded']]
    y = df['ETA']
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, le, scaler

def train_mlp():
    """Train a basic Multi-Layer Perceptron"""
    print("Training Multi-Layer Perceptron...")
    
    # Load data
    X, y, label_encoder, scaler = load_and_prepare_data()
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create basic MLP model
    # Keep it simple with few neurons and layers to avoid overfitting
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(16, 8),     # Small hidden layers
        activation='relu',              # ReLU activation
        solver='lbfgs',                 # Good for small datasets
        alpha=0.01,                     # L2 regularization
        max_iter=1000,                  # Maximum iterations
        random_state=42
    )
    
    # Train model
    mlp_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = mlp_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MLP Results:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Training iterations: {mlp_model.n_iter_}")
    
    # Save model, encoder, and scaler
    os.makedirs('utils/saved_models', exist_ok=True)
    joblib.dump(mlp_model, 'utils/saved_models/mlp_model.pkl')
    joblib.dump(label_encoder, 'utils/saved_models/mlp_travel_mode_encoder.pkl')
    joblib.dump(scaler, 'utils/saved_models/mlp_scaler.pkl')
    
    print("Model saved to utils/saved_models/mlp_model.pkl")
    
    return mlp_model, label_encoder, scaler, (mse, mae, r2)

if __name__ == "__main__":
    model, encoder, scaler, metrics = train_mlp()

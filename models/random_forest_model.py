"""
Random Forest Regressor for ETA Prediction
Basic model with minimal complexity to avoid overfitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    
    return X, y, le

def train_random_forest():
    """Train a basic Random Forest Regressor"""
    print("Training Random Forest Regressor...")
    
    # Load data
    X, y, label_encoder = load_and_prepare_data()
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create basic Random Forest model
    # Keep it simple to avoid overfitting with limited features
    rf_model = RandomForestRegressor(
        n_estimators=50,        # Small number of trees
        max_depth=10,           # Limit depth
        min_samples_split=20,   # Require more samples to split
        min_samples_leaf=10,    # Require more samples in leaf
        random_state=42
    )
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Results:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    
    # Feature importance
    feature_names = ['Distance', 'Speed', 'TravelMode']
    importance = rf_model.feature_importances_
    print(f"  Feature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"    {name}: {imp:.3f}")
    
    # Save model and encoder
    os.makedirs('utils/saved_models', exist_ok=True)
    joblib.dump(rf_model, 'utils/saved_models/random_forest_model.pkl')
    joblib.dump(label_encoder, 'utils/saved_models/travel_mode_encoder.pkl')
    
    print("Model saved to utils/saved_models/random_forest_model.pkl")
    
    return rf_model, label_encoder, (mse, mae, r2)

if __name__ == "__main__":
    model, encoder, metrics = train_random_forest()

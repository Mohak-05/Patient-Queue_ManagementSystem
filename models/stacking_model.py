"""
Stacking ensemble model for ETA prediction.

This module implements a StackingRegressor with Random Forest, KNN, 
and Linear Regression as base models for improved ETA prediction accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


class StackingETAPredictor:
    """Stacking ensemble model for ETA prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Stacking ensemble predictor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        # Base models
        self.base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=50, 
                max_depth=10, 
                random_state=random_state
            )),
            ('knn', KNeighborsRegressor(
                n_neighbors=5, 
                weights='distance'
            )),
            ('lr', LinearRegression())
        ]
        
        # Meta-learner
        self.meta_learner = LinearRegression()
        
        # Create stacking regressor
        self.model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['Distance', 'Speed', 'TravelMode_encoded']
        
    def prepare_features(self, df: pd.DataFrame, fit_preprocessing: bool = False) -> np.ndarray:
        """
        Prepare features for training or prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe with Distance, Speed, TravelMode columns
            fit_preprocessing (bool): Whether to fit preprocessing objects (True for training)
        
        Returns:
            np.ndarray: Prepared feature matrix
        """
        features_df = df.copy()
        
        # Encode travel mode
        if fit_preprocessing:
            features_df['TravelMode_encoded'] = self.label_encoder.fit_transform(features_df['TravelMode'])
        else:
            # Handle unseen categories
            try:
                features_df['TravelMode_encoded'] = self.label_encoder.transform(features_df['TravelMode'])
            except ValueError:
                # If there are unseen categories, use the most common category
                most_common = self.label_encoder.classes_[0]
                features_df['TravelMode'] = features_df['TravelMode'].apply(
                    lambda x: x if x in self.label_encoder.classes_ else most_common
                )
                features_df['TravelMode_encoded'] = self.label_encoder.transform(features_df['TravelMode'])
        
        # Prepare feature matrix
        X = features_df[['Distance', 'Speed', 'TravelMode_encoded']].values
        
        # Scale features (important for KNN and Linear Regression)
        if fit_preprocessing:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, data_path: str) -> Dict[str, float]:
        """
        Train the Stacking ensemble model.
        
        Args:
            data_path (str): Path to the training data CSV file
        
        Returns:
            Dict[str, float]: Training metrics
        """
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
        
        # Prepare features and target
        X = self.prepare_features(df, fit_preprocessing=True)
        y = df['ETA'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training Stacking ensemble model...")
        print("Base models: Random Forest, KNN, Linear Regression")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=5, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        metrics['cv_mae'] = -cv_scores.mean()
        
        # Individual base model performance
        base_model_metrics = self._evaluate_base_models(X_train, X_test, y_train, y_test)
        
        print("\nStacking Ensemble Results:")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"R²: {metrics['r2']:.3f}")
        print(f"CV MAE: {metrics['cv_mae']:.3f}")
        
        print("\nBase Model Performance:")
        for model_name, model_metrics in base_model_metrics.items():
            print(f"{model_name}: MAE={model_metrics['mae']:.3f}, R²={model_metrics['r2']:.3f}")
        
        return metrics
    
    def _evaluate_base_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual base models.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
        
        Returns:
            Dict[str, Dict[str, float]]: Metrics for each base model
        """
        base_metrics = {}
        
        for name, model in self.base_models:
            # Train individual model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            base_metrics[name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return base_metrics
    
    def predict(self, distance: float, speed: float, travel_mode: str) -> float:
        """
        Predict ETA for a single patient.
        
        Args:
            distance (float): Distance in kilometers
            speed (float): Speed in km/h
            travel_mode (str): Travel mode ('Walk', 'Bike', 'Car')
        
        Returns:
            float: Predicted ETA in minutes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create DataFrame for consistent preprocessing
        df = pd.DataFrame({
            'Distance': [distance],
            'Speed': [speed],
            'TravelMode': [travel_mode]
        })
        
        X = self.prepare_features(df, fit_preprocessing=False)
        prediction = self.model.predict(X)[0]
        
        return max(0, prediction)  # ETA cannot be negative
    
    def predict_batch(self, distances: list, speeds: list, travel_modes: list) -> np.ndarray:
        """
        Predict ETA for multiple patients.
        
        Args:
            distances (list): List of distances in kilometers
            speeds (list): List of speeds in km/h
            travel_modes (list): List of travel modes
        
        Returns:
            np.ndarray: Predicted ETAs in minutes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        df = pd.DataFrame({
            'Distance': distances,
            'Speed': speeds,
            'TravelMode': travel_modes
        })
        
        X = self.prepare_features(df, fit_preprocessing=False)
        predictions = self.model.predict(X)
        
        return np.maximum(0, predictions)  # ETA cannot be negative
    
    def get_base_model_predictions(self, distance: float, speed: float, travel_mode: str) -> Dict[str, float]:
        """
        Get predictions from individual base models.
        
        Args:
            distance (float): Distance in kilometers
            speed (float): Speed in km/h
            travel_mode (str): Travel mode
        
        Returns:
            Dict[str, float]: Predictions from each base model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        df = pd.DataFrame({
            'Distance': [distance],
            'Speed': [speed],
            'TravelMode': [travel_mode]
        })
        
        X = self.prepare_features(df, fit_preprocessing=False)
        
        predictions = {}
        for name, model in self.base_models:
            pred = model.predict(X)[0]
            predictions[name] = max(0, pred)
        
        # Add ensemble prediction
        predictions['ensemble'] = self.predict(distance, speed, travel_mode)
        
        return predictions
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'base_models': self.base_models,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Stacking model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.base_models = model_data['base_models']
        self.is_trained = model_data['is_trained']
        
        print(f"Stacking model loaded from {filepath}")
    
    def compare_models(self, distance: float, speed: float, travel_mode: str):
        """
        Compare predictions from all base models and ensemble.
        
        Args:
            distance (float): Distance in kilometers
            speed (float): Speed in km/h
            travel_mode (str): Travel mode
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.get_base_model_predictions(distance, speed, travel_mode)
        
        print(f"\nPrediction Comparison for:")
        print(f"Distance: {distance} km, Speed: {speed} km/h, Mode: {travel_mode}")
        print("-" * 50)
        
        for model_name, prediction in predictions.items():
            print(f"{model_name.upper()}: {prediction:.2f} minutes")


def train_and_save_stacking_model(data_path: str, 
                                model_save_path: str = "stacking_model.joblib") -> StackingETAPredictor:
    """
    Train and save a Stacking ensemble model.
    
    Args:
        data_path (str): Path to training data
        model_save_path (str): Path to save the trained model
    
    Returns:
        StackingETAPredictor: Trained model
    """
    predictor = StackingETAPredictor()
    metrics = predictor.train(data_path)
    predictor.save_model(model_save_path)
    
    return predictor


if __name__ == "__main__":
    # Example usage
    data_path = "data/sample_travel_data.csv"
    
    if os.path.exists(data_path):
        predictor = train_and_save_stacking_model(data_path)
        
        # Test single prediction
        test_eta = predictor.predict(distance=2.5, speed=35.0, travel_mode="Car")
        print(f"\nTest prediction: {test_eta:.1f} minutes")
        
        # Compare all models
        predictor.compare_models(distance=2.5, speed=35.0, travel_mode="Car")
        
    else:
        print(f"Data file not found: {data_path}")
        print("Please ensure the sample data file exists.")

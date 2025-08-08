"""
Random Forest model training for ETA prediction.

This module implements a Random Forest regressor to predict patient ETA
based on distance, speed, and travel mode.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestETAPredictor:
    """Random Forest model for ETA prediction."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the Random Forest predictor.
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names = ['Distance', 'Speed', 'TravelMode_encoded']
        
    def prepare_features(self, df: pd.DataFrame, fit_encoder: bool = False) -> np.ndarray:
        """
        Prepare features for training or prediction.
        
        Args:
            df (pd.DataFrame): Input dataframe with Distance, Speed, TravelMode columns
            fit_encoder (bool): Whether to fit the label encoder (True for training)
        
        Returns:
            np.ndarray: Prepared feature matrix
        """
        features_df = df.copy()
        
        if fit_encoder:
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
        
        return features_df[['Distance', 'Speed', 'TravelMode_encoded']].values
    
    def train(self, data_path: str) -> Dict[str, float]:
        """
        Train the Random Forest model.
        
        Args:
            data_path (str): Path to the training data CSV file
        
        Returns:
            Dict[str, float]: Training metrics
        """
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
        
        # Prepare features and target
        X = self.prepare_features(df, fit_encoder=True)
        y = df['ETA'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training Random Forest model...")
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
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        metrics['cv_mae'] = -cv_scores.mean()
        
        print("\nTraining Results:")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"R²: {metrics['r2']:.3f}")
        print(f"CV MAE: {metrics['cv_mae']:.3f}")
        
        return metrics
    
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
        
        X = self.prepare_features(df, fit_encoder=False)
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
        
        X = self.prepare_features(df, fit_encoder=False)
        predictions = self.model.predict(X)
        
        return np.maximum(0, predictions)  # ETA cannot be negative
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_dict = {}
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[feature] = importance
        
        return importance_dict
    
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
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
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
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def plot_feature_importance(self, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            save_path (str): Optional path to save the plot
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_dict = self.get_feature_importance()
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(features, importances)
        plt.title('Random Forest Feature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()


def train_and_save_model(data_path: str, model_save_path: str = "random_forest_model.joblib") -> RandomForestETAPredictor:
    """
    Train and save a Random Forest model.
    
    Args:
        data_path (str): Path to training data
        model_save_path (str): Path to save the trained model
    
    Returns:
        RandomForestETAPredictor: Trained model
    """
    predictor = RandomForestETAPredictor()
    metrics = predictor.train(data_path)
    predictor.save_model(model_save_path)
    
    return predictor


if __name__ == "__main__":
    # Example usage
    data_path = "data/sample_travel_data.csv"
    
    if os.path.exists(data_path):
        predictor = train_and_save_model(data_path)
        
        # Test single prediction
        test_eta = predictor.predict(distance=2.5, speed=35.0, travel_mode="Car")
        print(f"\nTest prediction: {test_eta:.1f} minutes")
        
        # Show feature importance
        importance = predictor.get_feature_importance()
        print("\nFeature Importance:")
        for feature, score in importance.items():
            print(f"  {feature}: {score:.3f}")
            
        # Plot feature importance
        try:
            predictor.plot_feature_importance()
        except:
            print("Could not display plot (running in terminal)")
    else:
        print(f"Data file not found: {data_path}")
        print("Please run the geo_simulation.py first to generate sample data.")

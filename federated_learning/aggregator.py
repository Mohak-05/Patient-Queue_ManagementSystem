"""
Federated Learning Aggregator
Combines hospital model weights to create global federated models
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

class FederatedRandomForest(BaseEstimator, RegressorMixin):
    """Federated Random Forest using weighted feature importance averaging"""
    
    def __init__(self):
        self.feature_importances_ = None
        self.n_estimators = 50
        self.hospital_weights = None
        self.base_model = None
        
    def aggregate_weights(self, hospital_weights):
        """Aggregate Random Forest weights from hospitals"""
        rf_weights = {}
        total_samples = 0
        
        # Collect weights and sample counts
        for hospital, weights in hospital_weights.items():
            if 'random_forest' in weights:
                rf_data = weights['random_forest']
                sample_weight = rf_data['performance']['r2']  # Use R² as quality weight
                if sample_weight > 0:  # Only positive R² scores
                    rf_weights[hospital] = {
                        'importances': np.array(rf_data['feature_importances']),
                        'weight': sample_weight
                    }
                    total_samples += sample_weight
        
        # Calculate weighted average of feature importances
        if rf_weights:
            aggregated_importances = np.zeros(3)  # 3 features
            
            for hospital, data in rf_weights.items():
                weight = data['weight'] / total_samples
                aggregated_importances += data['importances'] * weight
            
            self.feature_importances_ = aggregated_importances
            print(f"Federated RF feature importances: {self.feature_importances_}")
        
        self.hospital_weights = rf_weights
        
    def fit(self, X, y):
        """Fit method for sklearn compatibility"""
        # Create a base model with aggregated importances
        self.base_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.base_model.fit(X, y)
        return self
        
    def predict(self, X):
        """Predict using the base model"""
        if self.base_model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.base_model.predict(X)

class FederatedMLP(BaseEstimator, RegressorMixin):
    """Federated MLP using performance-weighted model averaging"""
    
    def __init__(self):
        self.hospital_models = {}
        self.hospital_scalers = {}
        self.hospital_weights = {}
        self.aggregated_scaler = None
        
    def aggregate_weights(self, hospital_weights):
        """Load hospital MLP models for ensemble prediction"""
        total_weight = 0
        
        for hospital, weights in hospital_weights.items():
            if 'mlp' in weights:
                mlp_data = weights['mlp']
                performance_weight = max(0, mlp_data['performance']['r2'])  # Use R² as weight
                
                if performance_weight > 0:
                    # Load hospital models
                    model_path = f'hospital_models/{hospital}/mlp_model.pkl'
                    scaler_path = f'hospital_models/{hospital}/mlp_scaler.pkl'
                    
                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        self.hospital_models[hospital] = joblib.load(model_path)
                        self.hospital_scalers[hospital] = joblib.load(scaler_path)
                        self.hospital_weights[hospital] = performance_weight
                        total_weight += performance_weight
        
        # Normalize weights
        if total_weight > 0:
            for hospital in self.hospital_weights:
                self.hospital_weights[hospital] /= total_weight
                
        print(f"Federated MLP weights: {self.hospital_weights}")
        
    def fit(self, X, y):
        """Fit method for sklearn compatibility"""
        # Create aggregated scaler
        self.aggregated_scaler = StandardScaler()
        self.aggregated_scaler.fit(X)
        return self
        
    def predict(self, X):
        """Predict using weighted ensemble of hospital models"""
        if not self.hospital_models:
            raise ValueError("No hospital models loaded")
            
        predictions = []
        weights = []
        
        for hospital, model in self.hospital_models.items():
            # Use hospital-specific scaler
            scaler = self.hospital_scalers[hospital]
            X_scaled = scaler.transform(X)
            
            # Get prediction
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weights.append(self.hospital_weights[hospital])
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        federated_prediction = np.average(predictions, axis=0, weights=weights)
        return federated_prediction

class FederatedLearningAggregator:
    """Main federated learning aggregator"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        
    def load_hospital_weights(self):
        """Load weights from all hospitals"""
        with open('weights/hospital_weights.json', 'r') as f:
            self.hospital_weights = json.load(f)
        print("[DATA] Hospital weights loaded successfully")
        
    def create_federated_models(self):
        """Create federated models from hospital weights"""
        print("\n[FEDERATED] Creating federated models...")
        
        # Create federated Random Forest
        fed_rf = FederatedRandomForest()
        fed_rf.aggregate_weights(self.hospital_weights)
        self.models['Random Forest'] = fed_rf
        
        # Create federated MLP
        fed_mlp = FederatedMLP()
        fed_mlp.aggregate_weights(self.hospital_weights)
        self.models['MLP'] = fed_mlp
        
        print("[SUCCESS] Federated models created")
        
    def train_on_global_data(self):
        """Train federated models on combined data for final calibration"""
        print("\n[TARGET] Final training on global dataset...")
        
        # Load original dataset for final training
        df = pd.read_csv('../data/travel_data.csv')
        
        # Prepare features
        le = LabelEncoder()
        df['TravelMode_encoded'] = le.fit_transform(df['TravelMode'])
        X = df[['Distance', 'Speed', 'TravelMode_encoded']]
        y = df['ETA']
        
        # Train federated models
        for name, model in self.models.items():
            model.fit(X, y)
            
        # Store encoder and scaler
        self.encoders['federated'] = le
        if 'MLP' in self.models:
            self.scalers['federated'] = self.models['MLP'].aggregated_scaler
            
    def save_federated_models(self):
        """Save federated models"""
        os.makedirs('global_models', exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_name = name.lower().replace(' ', '_')
            joblib.dump(model, f'global_models/federated_{model_name}.pkl')
            
        # Save encoders and scalers
        joblib.dump(self.encoders['federated'], 'global_models/federated_encoder.pkl')
        if 'federated' in self.scalers:
            joblib.dump(self.scalers['federated'], 'global_models/federated_scaler.pkl')
            
        print("[SUCCESS] Federated models saved to global_models/")
        
    def aggregate_all(self):
        """Complete federated learning aggregation process"""
        print("[TRAINING] Starting federated learning aggregation...")
        
        self.load_hospital_weights()
        self.create_federated_models()
        self.train_on_global_data()
        self.save_federated_models()
        
        print("\n[COMPLETE] Federated learning aggregation complete!")
        return self.models

def run_federated_aggregation():
    """Run the complete federated learning aggregation"""
    aggregator = FederatedLearningAggregator()
    return aggregator.aggregate_all()

if __name__ == "__main__":
    models = run_federated_aggregation()

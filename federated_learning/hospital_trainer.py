"""
Hospital-specific Model Training
Trains individual models for each hospital dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class HospitalModelTrainer:
    def __init__(self, hospital_name):
        self.hospital_name = hospital_name
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.metrics = {}
        
    def load_hospital_data(self):
        """Load hospital-specific dataset"""
        data_path = f'hospital_data/{self.hospital_name}_data.csv'
        self.df = pd.read_csv(data_path)
        print(f"[DATA] {self.hospital_name}: Loaded {len(self.df)} records")
        
    def prepare_features(self):
        """Prepare features for training"""
        # Encode travel mode
        self.le = LabelEncoder()
        self.df['TravelMode_encoded'] = self.le.fit_transform(self.df['TravelMode'])
        
        # Features and target
        self.X = self.df[['Distance', 'Speed', 'TravelMode_encoded']]
        self.y = self.df['ETA']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print(f"Training Random Forest for {self.hospital_name}...")
        
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        rf.fit(self.X_train, self.y_train)
        predictions = rf.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        self.models['random_forest'] = rf
        self.encoders['random_forest'] = self.le
        self.metrics['random_forest'] = {'mse': mse, 'mae': mae, 'r2': r2}
        
        print(f"  Random Forest - MAE: {mae:.3f}, R²: {r2:.3f}")
        
    def train_mlp(self):
        """Train MLP model"""
        print(f"Training MLP for {self.hospital_name}...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        mlp = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation='relu',
            solver='lbfgs',
            alpha=0.01,
            max_iter=500,
            random_state=42
        )
        
        mlp.fit(X_train_scaled, self.y_train)
        predictions = mlp.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        self.models['mlp'] = mlp
        self.encoders['mlp'] = self.le
        self.scalers['mlp'] = scaler
        self.metrics['mlp'] = {'mse': mse, 'mae': mae, 'r2': r2}
        
        print(f"  MLP - MAE: {mae:.3f}, R²: {r2:.3f}")
        
    def save_models(self):
        """Save trained models"""
        model_dir = f'hospital_models/{self.hospital_name}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')
            joblib.dump(self.encoders[model_name], f'{model_dir}/{model_name}_encoder.pkl')
            if model_name in self.scalers:
                joblib.dump(self.scalers[model_name], f'{model_dir}/{model_name}_scaler.pkl')
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(f'{model_dir}/metrics.csv')
        
        print(f"[SUCCESS] Models saved to {model_dir}")
        
    def get_model_weights(self):
        """Extract model weights for federated learning"""
        weights = {}
        
        # Random Forest feature importances
        if 'random_forest' in self.models:
            weights['random_forest'] = {
                'feature_importances': self.models['random_forest'].feature_importances_.tolist(),
                'n_estimators': self.models['random_forest'].n_estimators,
                'performance': self.metrics['random_forest']
            }
        
        # MLP weights (simplified representation)
        if 'mlp' in self.models:
            mlp = self.models['mlp']
            weights['mlp'] = {
                'coefs_shapes': [coef.shape for coef in mlp.coefs_],
                'intercepts_shapes': [intercept.shape for intercept in mlp.intercepts_],
                'n_layers': mlp.n_layers_,
                'performance': self.metrics['mlp']
            }
        
        return weights
        
    def train_all_models(self):
        """Train all models for this hospital"""
        print(f"\n[HOSPITAL] Training models for {self.hospital_name.upper()}")
        print("="*50)
        
        self.load_hospital_data()
        self.prepare_features()
        self.train_random_forest()
        self.train_mlp()
        self.save_models()
        
        return self.get_model_weights()

def train_all_hospitals():
    """Train models for all hospitals"""
    hospitals = ['hospital_a', 'hospital_b', 'hospital_c']
    all_weights = {}
    
    print("[TRAINING] Starting federated learning training phase...")
    
    for hospital in hospitals:
        trainer = HospitalModelTrainer(hospital)
        weights = trainer.train_all_models()
        all_weights[hospital] = weights
    
    # Save aggregated weights
    os.makedirs('weights', exist_ok=True)
    import json
    with open('weights/hospital_weights.json', 'w') as f:
        json.dump(all_weights, f, indent=2)
    
    print(f"\n[SUCCESS] All hospital models trained and weights saved!")
    return all_weights

if __name__ == "__main__":
    weights = train_all_hospitals()

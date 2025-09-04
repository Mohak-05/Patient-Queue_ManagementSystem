"""
Stacking Regressor for ETA Prediction
Combines XGBoost, MLP, and Random Forest with a simple meta-learner
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import xgboost as xgb
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

class ScaledMLPRegressor(RegressorMixin, BaseEstimator):
    """MLP Regressor with automatic feature scaling, sklearn-compatible"""
    
    def __init__(self, hidden_layer_sizes=(12, 6), activation='relu', 
                 solver='lbfgs', alpha=0.01, max_iter=500, random_state=42):
        # Store all parameters as instance attributes (required by sklearn)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit the MLP regressor with feature scaling"""
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Initialize scaler and MLP with stored parameters
        self.scaler_ = StandardScaler()
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        # Scale features and fit MLP
        X_scaled = self.scaler_.fit_transform(X)
        self.mlp_.fit(X_scaled, y)
        
        # Store classes for sklearn compatibility (required for some methods)
        self.n_features_in_ = X.shape[1]
        
        return self
    
    def predict(self, X):
        """Make predictions with feature scaling"""
        # Check if fitted
        check_is_fitted(self, ['scaler_', 'mlp_'])
        
        # Validate input
        X = check_array(X, accept_sparse=False)
        
        # Scale features and predict
        X_scaled = self.scaler_.transform(X)
        return self.mlp_.predict(X_scaled)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (required by sklearn)"""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator (required by sklearn)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _more_tags(self):
        """Additional tags for sklearn compatibility"""
        return {
            'requires_y': True,
            'requires_fit': True,
            'no_validation': False,
            'poor_score': False,
            'requires_positive_X': False
        }

def train_stacking_regressor():
    """Train a Stacking Regressor combining XGBoost, MLP, and Random Forest"""
    print("Training Stacking Regressor...")
    
    # Load data
    X, y, label_encoder = load_and_prepare_data()
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define base estimators (keep them simple)
    base_estimators = [
        ('random_forest', RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )),
        ('xgboost', xgb.XGBRegressor(
            n_estimators=30,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )),
        ('mlp', ScaledMLPRegressor(
            hidden_layer_sizes=(12, 6),
            activation='relu',
            solver='lbfgs',
            alpha=0.01,
            max_iter=500,
            random_state=42
        ))
    ]
    
    # Create stacking regressor with simple linear meta-learner
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression(),  # Simple meta-learner
        cv=3,  # 3-fold cross-validation for stacking
        n_jobs=-1
    )
    
    # Train model (no need for scaling since ScaledMLPRegressor handles it internally)
    stacking_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Stacking Regressor Results:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    
    # Show individual base estimator performance
    print(f"  Base Estimator Performance:")
    for name, estimator in base_estimators:
        if name == 'mlp':
            # Create a fresh ScaledMLPRegressor for individual testing
            test_mlp = ScaledMLPRegressor(
                hidden_layer_sizes=(12, 6),
                activation='relu',
                solver='lbfgs',
                alpha=0.01,
                max_iter=500,
                random_state=42
            )
            test_mlp.fit(X_train, y_train)
            pred = test_mlp.predict(X_test)
        else:
            # Clone the estimator to avoid interference
            test_estimator = clone(estimator)
            test_estimator.fit(X_train, y_train)
            pred = test_estimator.predict(X_test)
        
        base_mse = mean_squared_error(y_test, pred)
        print(f"    {name}: MSE = {base_mse:.3f}")
    
    # Save model and encoder (no separate scaler needed since it's in ScaledMLPRegressor)
    os.makedirs('utils/saved_models', exist_ok=True)
    joblib.dump(stacking_model, 'utils/saved_models/stacking_model.pkl')
    joblib.dump(label_encoder, 'utils/saved_models/stacking_travel_mode_encoder.pkl')
    
    print("Model saved to utils/saved_models/stacking_model.pkl")
    
    return stacking_model, label_encoder, None, (mse, mae, r2)  # No separate scaler

if __name__ == "__main__":
    model, encoder, _, metrics = train_stacking_regressor()  # No scaler returned

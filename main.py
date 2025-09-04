"""
Interactive Model Analysis Script
Analyzes model performance on individual journey GPS tracking data
Compares ETA predictions at 2-minute intervals with actual journey progression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
import glob
from models.stacking_model import ScaledMLPRegressor  # Import for model loading

# Import custom ScaledMLPRegressor for unpickling
sys.path.append('.')
from models.stacking_model import ScaledMLPRegressor

class JourneyAnalyzer:
    def __init__(self):
        """Initialize the journey analyzer with loaded models"""
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.load_all_models()
        
    def load_all_models(self):
        """Load all trained models and their preprocessors"""
        model_dir = 'utils/saved_models'
        
        try:
            # Load Random Forest
            self.models['Random Forest'] = joblib.load(f'{model_dir}/random_forest_model.pkl')
            self.encoders['Random Forest'] = joblib.load(f'{model_dir}/travel_mode_encoder.pkl')
            
            # Load MLP
            self.models['MLP'] = joblib.load(f'{model_dir}/mlp_model.pkl')
            self.encoders['MLP'] = joblib.load(f'{model_dir}/mlp_travel_mode_encoder.pkl')
            self.scalers['MLP'] = joblib.load(f'{model_dir}/mlp_scaler.pkl')
            
            # Load Stacking
            self.models['Stacking'] = joblib.load(f'{model_dir}/stacking_model.pkl')
            self.encoders['Stacking'] = joblib.load(f'{model_dir}/stacking_travel_mode_encoder.pkl')
            
            print("✅ All models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Please train the models first by running:")
            print("  python models/random_forest_model.py")
            print("  python models/mlp_model.py") 
            print("  python models/stacking_model.py")
            sys.exit(1)
    
    def get_journey_data(self, journey_id, with_traffic=True):
        """Load GPS tracking and timestamp data for a specific journey"""
        traffic_suffix = "with_traffic" if with_traffic else "no_traffic"
        
        # Determine batch number
        batch_num = ((journey_id - 1) // 1000) + 1
        batch_dir = f"batch_{batch_num:04d}"
        
        # Load geolocation data
        geo_file = f"data/geolocations/{traffic_suffix}/{batch_dir}/journey_{journey_id:04d}.csv"
        timestamp_file = f"data/timestamps/{traffic_suffix}/{batch_dir}/journey_{journey_id:04d}.csv"
        
        try:
            geo_data = pd.read_csv(geo_file)
            timestamp_data = pd.read_csv(timestamp_file)
            
            # Get original journey parameters from travel_data.csv
            travel_data = pd.read_csv('data/travel_data.csv')
            original_journey = travel_data.iloc[journey_id - 1]  # 0-indexed
            
            return geo_data, timestamp_data, original_journey
            
        except FileNotFoundError:
            print(f"❌ Journey {journey_id} data not found. Please ensure journey ID is between 1 and 4414.")
            return None, None, None
    
    def predict_eta_for_journey(self, geo_data, timestamp_data, model_name):
        """Make ETA predictions for each point in the journey"""
        predictions = []
        
        for i, geo_row in geo_data.iterrows():
            # Get corresponding timestamp data
            timestamp_row = timestamp_data.iloc[i]
            
            # Prepare features: Distance, Speed, TravelMode
            distance = geo_row['distance_remaining_km']
            travel_mode = timestamp_row['travel_mode']
            
            # Calculate speed from remaining distance and ETA
            if geo_row['eta_remaining_minutes'] > 0:
                speed = (distance * 60) / geo_row['eta_remaining_minutes']  # km/h
            else:
                speed = 0  # At destination
            
            # Encode travel mode
            encoder = self.encoders[model_name]
            try:
                travel_mode_encoded = encoder.transform([travel_mode])[0]
            except ValueError:
                # Handle unseen travel mode
                travel_mode_encoded = 0
            
            # Prepare feature vector with proper column names
            feature_names = ['Distance', 'Speed', 'TravelMode_encoded']
            features = pd.DataFrame([[distance, speed, travel_mode_encoded]], columns=feature_names)
            
            # Make prediction based on model type
            if model_name == 'MLP':
                # Scale features for MLP
                features_scaled = self.scalers['MLP'].transform(features)
                prediction = self.models[model_name].predict(features_scaled)[0]
            else:
                # Random Forest and Stacking use raw features
                prediction = self.models[model_name].predict(features)[0]
            
            predictions.append(max(0, prediction))  # Ensure non-negative ETA
        
        return predictions
    
    def calculate_journey_metrics(self, actual_etas, predicted_etas):
        """Calculate performance metrics for a journey"""
        actual = np.array(actual_etas)
        predicted = np.array(predicted_etas)
        
        # Filter out zero values (destination points)
        mask = actual > 0
        if mask.sum() == 0:
            return {'mse': 0, 'mae': 0, 'mape': 0, 'r2': 0}
        
        actual_filtered = actual[mask]
        predicted_filtered = predicted[mask]
        
        mse = mean_squared_error(actual_filtered, predicted_filtered)
        mae = mean_absolute_error(actual_filtered, predicted_filtered)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
        
        # R² score
        r2 = r2_score(actual_filtered, predicted_filtered)
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'points': len(actual_filtered)
        }
    
    def plot_journey_analysis(self, journey_id, geo_data, timestamp_data, original_journey, with_traffic=True):
        """Create comprehensive plots for journey analysis"""
        # Get model predictions
        model_predictions = {}
        for model_name in self.models.keys():
            model_predictions[model_name] = self.predict_eta_for_journey(geo_data, timestamp_data, model_name)
        
        # Prepare data for plotting
        timestamps = geo_data['timestamp_minutes'].values
        actual_etas = geo_data['eta_remaining_minutes'].values
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Journey {journey_id} Analysis ({"With" if with_traffic else "No"} Traffic)\\n'
                    f'Mode: {original_journey["TravelMode"]}, Distance: {original_journey["Distance"]:.2f}km, '
                    f'Original ETA: {original_journey["ETA"]:.1f}min', fontsize=14, fontweight='bold')
        
        # Plot 1: ETA Predictions vs Actual
        ax1.plot(timestamps, actual_etas, 'k-', linewidth=3, label='Actual ETA', marker='o', markersize=4)
        
        colors = ['red', 'blue', 'green']
        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            ax1.plot(timestamps, predictions, '--', color=colors[i], linewidth=2, 
                    label=f'{model_name} Prediction', marker='s', markersize=3)
        
        ax1.set_xlabel('Elapsed Time (minutes)')
        ax1.set_ylabel('Remaining ETA (minutes)')
        ax1.set_title('ETA Predictions Over Journey')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors
        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            errors = np.array(predictions) - actual_etas
            ax2.plot(timestamps, errors, '--', color=colors[i], linewidth=2,
                    label=f'{model_name} Error', marker='s', markersize=3)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Elapsed Time (minutes)')
        ax2.set_ylabel('Prediction Error (minutes)')
        ax2.set_title('Prediction Errors Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GPS Path
        lats = geo_data['latitude'].values
        lons = geo_data['longitude'].values
        
        # Color points by time progression
        scatter = ax3.scatter(lons, lats, c=timestamps, cmap='viridis', s=50, alpha=0.7)
        ax3.plot(lons, lats, 'gray', alpha=0.5, linewidth=1)
        
        # Mark start and end
        ax3.scatter(lons[0], lats[0], color='green', s=100, marker='o', label='Start', edgecolors='black')
        ax3.scatter(lons[-1], lats[-1], color='red', s=100, marker='s', label='Hospital', edgecolors='black')
        
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.set_title('GPS Journey Path')
        ax3.legend()
        plt.colorbar(scatter, ax=ax3, label='Elapsed Time (min)')
        
        # Plot 4: Model Performance Metrics
        metrics_data = []
        for model_name, predictions in model_predictions.items():
            metrics = self.calculate_journey_metrics(actual_etas, predictions)
            metrics_data.append([model_name, metrics['mae'], metrics['mape'], metrics['r2']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'MAE', 'MAPE', 'R²'])
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax4.bar(x - width, metrics_df['MAE'], width, label='MAE (min)', alpha=0.8)
        ax4.bar(x, metrics_df['MAPE'], width, label='MAPE (%)', alpha=0.8)
        ax4.bar(x + width, metrics_df['R²'] * 10, width, label='R² (×10)', alpha=0.8)  # Scale R² for visibility
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Metric Values')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_df['Model'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = 'analysis_plots'
        os.makedirs(plot_dir, exist_ok=True)
        traffic_suffix = "with_traffic" if with_traffic else "no_traffic"
        plt.savefig(f'{plot_dir}/journey_{journey_id}_{traffic_suffix}.png', dpi=300, bbox_inches='tight')
        
        return metrics_df
    
    def print_detailed_metrics(self, journey_id, metrics_df, original_journey, timestamp_data):
        """Print detailed analysis results"""
        print("\\n" + "="*80)
        print(f"DETAILED ANALYSIS - JOURNEY {journey_id}")
        print("="*80)
        
        print(f"\\n📍 Journey Details:")
        print(f"   Travel Mode: {original_journey['TravelMode']}")
        print(f"   Total Distance: {original_journey['Distance']:.2f} km")
        print(f"   Average Speed: {original_journey['Speed']:.2f} km/h")
        print(f"   Original ETA: {original_journey['ETA']:.1f} minutes")
        print(f"   Actual Duration: {timestamp_data['actual_duration_minutes'].iloc[0]:.1f} minutes")
        print(f"   Time Variance: {timestamp_data['time_variance_percent'].iloc[0]:.1f}%")
        print(f"   GPS Updates: {len(timestamp_data)} points (every 2 minutes)")
        
        print(f"\\n📊 Model Performance Metrics:")
        print("-" * 60)
        print(f"{'Model':<15} {'MAE (min)':<12} {'MAPE (%)':<12} {'R² Score':<12}")
        print("-" * 60)
        
        for _, row in metrics_df.iterrows():
            print(f"{row['Model']:<15} {row['MAE']:<12.3f} {row['MAPE']:<12.2f} {row['R²']:<12.3f}")
        
        # Find best model
        best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Model']
        best_r2 = metrics_df.loc[metrics_df['R²'].idxmax(), 'R²']
        
        print(f"\\n🏆 Best Performing Model: {best_model} (R² = {best_r2:.3f})")
        
        # Additional insights
        print(f"\\n💡 Analysis Insights:")
        if timestamp_data['is_rush_hour'].sum() > 0:
            print(f"   • Journey occurred during rush hour")
        if abs(timestamp_data['time_variance_percent'].iloc[0]) > 20:
            print(f"   • Significant time variance from original estimate")
        if len(timestamp_data) > 10:
            print(f"   • Long journey with {len(timestamp_data)} tracking points")

def main():
    """Main interactive analysis function"""
    analyzer = JourneyAnalyzer()
    
    print("\\n" + "="*60)
    print("🚗 PATIENT QUEUE MANAGEMENT SYSTEM")
    print("   Individual Journey Analysis Tool")
    print("="*60)
    
    while True:
        try:
            # Get user input
            print(f"\\nEnter journey ID (1-4414) or 'quit' to exit:")
            user_input = input("Journey ID: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
                
            journey_id = int(user_input)
            
            if not (1 <= journey_id <= 4414):
                print("❌ Please enter a number between 1 and 4414")
                continue
            
            # Ask for traffic preference
            print("\\nAnalyze with traffic effects? (y/n):")
            traffic_choice = input("Traffic: ").strip().lower()
            with_traffic = traffic_choice in ['y', 'yes', '']
            
            print(f"\\n🔍 Analyzing Journey {journey_id}...")
            
            # Load journey data
            geo_data, timestamp_data, original_journey = analyzer.get_journey_data(journey_id, with_traffic)
            
            if geo_data is None:
                continue
            
            # Perform analysis and create plots
            metrics_df = analyzer.plot_journey_analysis(journey_id, geo_data, timestamp_data, original_journey, with_traffic)
            
            # Print detailed results
            analyzer.print_detailed_metrics(journey_id, metrics_df, original_journey, timestamp_data)
            
            traffic_suffix = "with_traffic" if with_traffic else "no_traffic"
            print(f"\\n📈 Plot saved: analysis_plots/journey_{journey_id}_{traffic_suffix}.png")
            
            print(f"\\n" + "-"*60)
            
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\\n\\n👋 Analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

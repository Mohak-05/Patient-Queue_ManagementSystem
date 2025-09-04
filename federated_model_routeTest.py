"""
Federated Learning Model Analysis Script
Analyzes federated model performance on individual journey GPS tracking data
Compares ETA predictions from federated models with actual journey progression
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

class FederatedJourneyAnalyzer:
    def __init__(self):
        """Initialize the federated journey analyzer with loaded models"""
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.load_federated_models()
        
    def load_federated_models(self):
        """Load individual hospital models for federated ensemble prediction"""
        hospitals = ['hospital_a', 'hospital_b', 'hospital_c']
        self.hospital_models = {}
        self.hospital_encoders = {}
        self.hospital_scalers = {}
        
        try:
            # Load models from each hospital
            for hospital in hospitals:
                model_dir = f'federated_learning/hospital_models/{hospital}'
                
                # Load Random Forest and MLP for each hospital
                rf_model = joblib.load(f'{model_dir}/random_forest_model.pkl')
                mlp_model = joblib.load(f'{model_dir}/mlp_model.pkl')
                
                # Load encoders and scalers
                rf_encoder = joblib.load(f'{model_dir}/random_forest_encoder.pkl')
                mlp_encoder = joblib.load(f'{model_dir}/mlp_encoder.pkl')
                mlp_scaler = joblib.load(f'{model_dir}/mlp_scaler.pkl')
                
                self.hospital_models[hospital] = {
                    'random_forest': rf_model,
                    'mlp': mlp_model
                }
                self.hospital_encoders[hospital] = {
                    'random_forest': rf_encoder,
                    'mlp': mlp_encoder
                }
                self.hospital_scalers[hospital] = {
                    'mlp': mlp_scaler
                }
            
            # Create federated ensemble models
            self.models['Federated Random Forest'] = self.create_federated_rf()
            self.models['Federated MLP'] = self.create_federated_mlp()
            
            print("✅ Federated models created from hospital ensembles!")
            print(f"   🏥 Hospitals: {hospitals}")
            print(f"   📊 Models available: {list(self.models.keys())}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading hospital models: {e}")
            print("Please run the federated learning pipeline first:")
            print("  cd federated_learning")
            print("  python run_federated_pipeline.py")
            sys.exit(1)
    
    def create_federated_rf(self):
        """Create federated Random Forest ensemble"""
        class FederatedRandomForestEnsemble:
            def __init__(self, hospital_models, hospital_encoders):
                self.hospital_models = hospital_models
                self.hospital_encoders = hospital_encoders
            
            def predict(self, X):
                predictions = []
                for hospital in self.hospital_models:
                    model = self.hospital_models[hospital]['random_forest']
                    pred = model.predict(X)
                    predictions.append(pred)
                
                # Average predictions from all hospitals
                return np.mean(predictions, axis=0)
        
        return FederatedRandomForestEnsemble(self.hospital_models, self.hospital_encoders)
    
    def create_federated_mlp(self):
        """Create federated MLP ensemble"""
        class FederatedMLPEnsemble:
            def __init__(self, hospital_models, hospital_encoders, hospital_scalers):
                self.hospital_models = hospital_models
                self.hospital_encoders = hospital_encoders
                self.hospital_scalers = hospital_scalers
            
            def predict(self, X):
                predictions = []
                for hospital in self.hospital_models:
                    model = self.hospital_models[hospital]['mlp']
                    scaler = self.hospital_scalers[hospital]['mlp']
                    
                    # Scale features using hospital-specific scaler
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                
                # Weighted average based on hospital performance (equal weights for now)
                return np.mean(predictions, axis=0)
        
        return FederatedMLPEnsemble(self.hospital_models, self.hospital_encoders, self.hospital_scalers)
    
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
    
    def predict_eta_for_journey(self, geo_data, timestamp_data, model_name, specific_hospital=None):
        """Make ETA predictions for each point in the journey using federated models or specific hospital models"""
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
            
            # If specific hospital is provided, use that hospital's encoder
            if specific_hospital:
                encoder = self.hospital_encoders[specific_hospital]['random_forest']
            else:
                # Use first hospital's encoder for federated models (they should be similar)
                encoder = list(self.hospital_encoders.values())[0]['random_forest']
            
            try:
                travel_mode_encoded = encoder.transform([travel_mode])[0]
            except ValueError:
                # Handle unseen travel mode
                travel_mode_encoded = 0
            
            # Prepare feature vector
            feature_names = ['Distance', 'Speed', 'TravelMode_encoded']
            features = pd.DataFrame([[distance, speed, travel_mode_encoded]], columns=feature_names)
            
            # Make prediction based on model type and whether it's hospital-specific
            if specific_hospital:
                # Use specific hospital's model
                if 'MLP' in model_name:
                    scaler = self.hospital_scalers[specific_hospital]['mlp']
                    X_scaled = scaler.transform(features)
                    prediction = self.hospital_models[specific_hospital]['mlp'].predict(X_scaled)[0]
                else:
                    prediction = self.hospital_models[specific_hospital]['random_forest'].predict(features)[0]
            else:
                # Use federated ensemble
                if 'MLP' in model_name:
                    prediction = self.models[model_name].predict(features)[0]
                else:
                    prediction = self.models[model_name].predict(features)[0]
            
            predictions.append(max(0, prediction))  # Ensure non-negative ETA
        
        return predictions
    
    def get_hospital_assignment(self, journey_id):
        """Determine which hospital this journey was assigned to during federated training"""
        try:
            with open('federated_learning/journey_hospital_mapping.json', 'r') as f:
                import json
                mapping = json.load(f)
                return mapping.get(str(journey_id), 'Unknown')
        except FileNotFoundError:
            return 'Unknown'
    
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
    
    def plot_federated_analysis(self, journey_id, geo_data, timestamp_data, original_journey, with_traffic=True):
        """Create comprehensive plots for federated journey analysis"""
        # Get hospital assignment
        hospital = self.get_hospital_assignment(journey_id)
        
        # Get federated model predictions
        model_predictions = {}
        for model_name in self.models.keys():
            model_predictions[model_name] = self.predict_eta_for_journey(geo_data, timestamp_data, model_name)
        
        # Get corresponding hospital model predictions
        hospital_predictions = {}
        if hospital != 'Unknown':
            hospital_predictions[f'{hospital.upper()} Random Forest'] = self.predict_eta_for_journey(
                geo_data, timestamp_data, 'Random Forest', specific_hospital=hospital)
            hospital_predictions[f'{hospital.upper()} MLP'] = self.predict_eta_for_journey(
                geo_data, timestamp_data, 'MLP', specific_hospital=hospital)
        
        # Combine all predictions
        all_predictions = {**model_predictions, **hospital_predictions}
        
        # Prepare data for plotting
        timestamps = geo_data['timestamp_minutes'].values
        actual_etas = geo_data['eta_remaining_minutes'].values
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Federated vs Hospital-Specific Journey {journey_id} Analysis ({"With" if with_traffic else "No"} Traffic)\\n'
                    f'Hospital: {hospital.upper()}, Mode: {original_journey["TravelMode"]}, '
                    f'Distance: {original_journey["Distance"]:.2f}km, Original ETA: {original_journey["ETA"]:.1f}min', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: ETA Predictions vs Actual
        ax1.plot(timestamps, actual_etas, 'k-', linewidth=3, label='Actual ETA', marker='o', markersize=4)
        
        colors = ['red', 'blue', 'green', 'orange']
        linestyles = ['--', '--', '-.', '-.']
        for i, (model_name, predictions) in enumerate(all_predictions.items()):
            ax1.plot(timestamps, predictions, linestyles[i % len(linestyles)], color=colors[i % len(colors)], 
                    linewidth=2, label=f'{model_name}', marker='s', markersize=3)
        
        ax1.set_xlabel('Elapsed Time (minutes)')
        ax1.set_ylabel('Remaining ETA (minutes)')
        ax1.set_title('Federated vs Hospital-Specific ETA Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors
        for i, (model_name, predictions) in enumerate(all_predictions.items()):
            errors = np.array(predictions) - actual_etas
            ax2.plot(timestamps, errors, linestyles[i % len(linestyles)], color=colors[i % len(colors)], 
                    linewidth=2, label=f'{model_name} Error', marker='s', markersize=3)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Elapsed Time (minutes)')
        ax2.set_ylabel('Prediction Error (minutes)')
        ax2.set_title('Prediction Errors: Federated vs Hospital-Specific')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GPS Path with Hospital Info
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
        ax3.set_title(f'GPS Journey Path - {hospital.upper()} Hospital Dataset')
        ax3.legend()
        plt.colorbar(scatter, ax=ax3, label='Elapsed Time (min)')
        
        # Plot 4: Model Performance Comparison
        metrics_data = []
        for model_name, predictions in all_predictions.items():
            metrics = self.calculate_journey_metrics(actual_etas, predictions)
            metrics_data.append([model_name, metrics['mae'], metrics['mape'], metrics['r2']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'MAE', 'MAPE', 'R²'])
        
        x = np.arange(len(metrics_df))
        width = 0.2
        
        ax4.bar(x - width, metrics_df['MAE'], width, label='MAE (min)', alpha=0.8, color='skyblue')
        ax4.bar(x, metrics_df['MAPE']/5, width, label='MAPE (%/5)', alpha=0.8, color='lightcoral')
        
        # Create second y-axis for R²
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + width, metrics_df['R²'], width, label='R² Score', alpha=0.6, color='lightgreen')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('MAE (min) / MAPE (%)', color='blue')
        ax4_twin.set_ylabel('R² Score', color='green')
        ax4.set_title('Federated vs Hospital-Specific Performance')
        ax4.set_xticks(x)
        ax4.set_xticklabels([name.replace('Federated ', 'Fed ').replace(' Random Forest', ' RF') 
                           for name in metrics_df['Model']], rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = 'analysis_plots'
        os.makedirs(plot_dir, exist_ok=True)
        traffic_suffix = "with_traffic" if with_traffic else "no_traffic"
        plt.savefig(f'{plot_dir}/federated_vs_hospital_journey_{journey_id}_{traffic_suffix}.png', dpi=300, bbox_inches='tight')
        
        return metrics_df, hospital
    
    def print_federated_metrics(self, journey_id, metrics_df, original_journey, timestamp_data, hospital):
        """Print detailed federated analysis results"""
        print("\\n" + "="*80)
        print(f"FEDERATED vs HOSPITAL-SPECIFIC ANALYSIS - JOURNEY {journey_id}")
        print("="*80)
        
        print(f"\\n🏥 Model Comparison Details:")
        print(f"   Assigned Hospital: {hospital.upper()}")
        print(f"   Federated Models: Multi-hospital knowledge aggregation")
        print(f"   Hospital Models: {hospital.upper()}-specific training only")
        print(f"   Training Strategy: Collaborative vs Individual learning")
        
        print(f"\\n📍 Journey Details:")
        print(f"   Travel Mode: {original_journey['TravelMode']}")
        print(f"   Total Distance: {original_journey['Distance']:.2f} km")
        print(f"   Average Speed: {original_journey['Speed']:.2f} km/h")
        print(f"   Original ETA: {original_journey['ETA']:.1f} minutes")
        print(f"   Actual Duration: {timestamp_data['actual_duration_minutes'].iloc[0]:.1f} minutes")
        print(f"   Time Variance: {timestamp_data['time_variance_percent'].iloc[0]:.1f}%")
        print(f"   GPS Updates: {len(timestamp_data)} points (every 2 minutes)")
        
        print(f"\\n📊 Model Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<35} {'MAE (min)':<12} {'MAPE (%)':<12} {'R² Score':<12}")
        print("-" * 80)
        
        # Separate federated and hospital models
        federated_models = []
        hospital_models = []
        
        for _, row in metrics_df.iterrows():
            if 'Federated' in row['Model']:
                federated_models.append(row)
            else:
                hospital_models.append(row)
        
        # Print federated models first
        for row in federated_models:
            print(f"{row['Model']:<35} {row['MAE']:<12.3f} {row['MAPE']:<12.2f} {row['R²']:<12.3f}")
        
        if hospital_models:
            print("-" * 80)
        
        # Print hospital-specific models
        for row in hospital_models:
            print(f"{row['Model']:<35} {row['MAE']:<12.3f} {row['MAPE']:<12.2f} {row['R²']:<12.3f}")
        
        # Find best models in each category
        best_overall = metrics_df.loc[metrics_df['R²'].idxmax()]
        best_federated = max(federated_models, key=lambda x: x['R²']) if federated_models else None
        best_hospital = max(hospital_models, key=lambda x: x['R²']) if hospital_models else None
        
        print(f"\\n🏆 Performance Analysis:")
        print(f"   Best Overall: {best_overall['Model']} (R² = {best_overall['R²']:.3f})")
        if best_federated:
            print(f"   Best Federated: {best_federated['Model']} (R² = {best_federated['R²']:.3f})")
        if best_hospital:
            print(f"   Best Hospital: {best_hospital['Model']} (R² = {best_hospital['R²']:.3f})")
        
        # Performance comparison insights
        print(f"\\n🔗 Federated vs Hospital-Specific Insights:")
        if best_federated and best_hospital:
            fed_r2 = best_federated['R²']
            hosp_r2 = best_hospital['R²']
            if fed_r2 > hosp_r2:
                improvement = ((fed_r2 - hosp_r2) / hosp_r2) * 100
                print(f"   • Federated learning shows {improvement:.1f}% improvement over hospital-specific")
                print(f"   • Multi-hospital collaboration enhances prediction accuracy")
            elif hosp_r2 > fed_r2:
                improvement = ((hosp_r2 - fed_r2) / fed_r2) * 100
                print(f"   • Hospital-specific model shows {improvement:.1f}% better performance")
                print(f"   • Local specialization outperforms global knowledge")
            else:
                print(f"   • Federated and hospital-specific models show similar performance")
        
        print(f"   • Journey trained on {hospital} hospital data")
        print(f"   • Federated models benefit from multi-hospital knowledge sharing")
        print(f"   • Hospital models leverage local traffic patterns and conditions")
        
        # Additional insights
        if timestamp_data['is_rush_hour'].sum() > 0:
            print(f"   • Journey occurred during rush hour - local knowledge advantage")
        if abs(timestamp_data['time_variance_percent'].iloc[0]) > 20:
            print(f"   • Significant time variance - challenging prediction scenario")
        if len(timestamp_data) > 10:
            print(f"   • Long journey with {len(timestamp_data)} tracking points - comprehensive analysis")

def main():
    """Main federated analysis function"""
    analyzer = FederatedJourneyAnalyzer()
    
    print("\\n" + "="*70)
    print("🏥 FEDERATED PATIENT QUEUE MANAGEMENT SYSTEM")
    print("   Federated Learning Journey Analysis Tool")
    print("="*70)
    
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
            
            print(f"\\n🔍 Analyzing Journey {journey_id} with Federated Models...")
            
            # Load journey data
            geo_data, timestamp_data, original_journey = analyzer.get_journey_data(journey_id, with_traffic)
            
            if geo_data is None:
                continue
            
            # Perform federated analysis and create plots
            metrics_df, hospital = analyzer.plot_federated_analysis(journey_id, geo_data, timestamp_data, original_journey, with_traffic)
            
            # Print detailed results
            analyzer.print_federated_metrics(journey_id, metrics_df, original_journey, timestamp_data, hospital)
            
            traffic_suffix = "with_traffic" if with_traffic else "no_traffic"
            print(f"\\n📈 Plot saved: analysis_plots/federated_vs_hospital_journey_{journey_id}_{traffic_suffix}.png")
            
            print(f"\\n" + "-"*70)
            
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\\n\\n👋 Analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

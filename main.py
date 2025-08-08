"""
Main simulation script for Intelligent Patient Queue Management System.

This script simulates real-time patient queue updates with location-based
ETA prediction using machine learning models.
"""

import os
import sys
import time
import random
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import GeoSimulator, haversine_distance
from models import RandomForestETAPredictor, StackingETAPredictor


class Patient:
    """Represents a patient in the queue system."""
    
    def __init__(self, patient_id: str, latitude: float, longitude: float,
                 speed: float, travel_mode: str, eta_minutes: float):
        """
        Initialize a patient.
        
        Args:
            patient_id (str): Unique patient identifier
            latitude (float): Patient's current latitude
            longitude (float): Patient's current longitude
            speed (float): Travel speed in km/h
            travel_mode (str): Travel mode ('Walk', 'Bike', 'Car')
            eta_minutes (float): Estimated time of arrival in minutes
        """
        self.patient_id = patient_id
        self.latitude = latitude
        self.longitude = longitude
        self.speed = speed
        self.travel_mode = travel_mode
        self.eta_minutes = eta_minutes
        self.arrival_time = datetime.now()
        self.distance_km = 0.0
    
    def update_eta(self, time_elapsed_minutes: float):
        """
        Update patient ETA based on elapsed time.
        
        Args:
            time_elapsed_minutes (float): Time elapsed since last update
        """
        self.eta_minutes = max(0, self.eta_minutes - time_elapsed_minutes)
    
    def calculate_distance_to_hospital(self, hospital_lat: float, hospital_lon: float):
        """
        Calculate and update distance to hospital.
        
        Args:
            hospital_lat (float): Hospital latitude
            hospital_lon (float): Hospital longitude
        """
        self.distance_km = haversine_distance(
            self.latitude, self.longitude, hospital_lat, hospital_lon
        )
    
    def __str__(self):
        """String representation of patient."""
        return f"Patient({self.patient_id}, ETA: {self.eta_minutes:.1f}min, Mode: {self.travel_mode})"
    
    def __repr__(self):
        """Representation of patient."""
        return self.__str__()


class PatientQueueManager:
    """Manages the patient queue and predictions."""
    
    def __init__(self, hospital_lat: float = 40.7128, hospital_lon: float = -74.0060):
        """
        Initialize the queue manager.
        
        Args:
            hospital_lat (float): Hospital latitude
            hospital_lon (float): Hospital longitude
        """
        self.hospital_lat = hospital_lat
        self.hospital_lon = hospital_lon
        self.patients: List[Patient] = []
        self.geo_simulator = GeoSimulator(hospital_lat, hospital_lon)
        self.predictor = None
        self.model_type = "random_forest"  # or "stacking"
        
    def load_model(self, model_path: str = None, model_type: str = "random_forest"):
        """
        Load a trained ML model for ETA prediction.
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of model ('random_forest' or 'stacking')
        """
        self.model_type = model_type
        
        try:
            if model_type == "stacking":
                self.predictor = StackingETAPredictor()
                if model_path and os.path.exists(model_path):
                    self.predictor.load_model(model_path)
                else:
                    # Train on the fly if no model exists
                    print("Training stacking model on the fly...")
                    data_path = "data/sample_travel_data.csv"
                    if os.path.exists(data_path):
                        self.predictor.train(data_path)
                    else:
                        print("Warning: No training data found. Using basic ETA calculation.")
                        self.predictor = None
            else:
                self.predictor = RandomForestETAPredictor()
                if model_path and os.path.exists(model_path):
                    self.predictor.load_model(model_path)
                else:
                    # Train on the fly if no model exists
                    print("Training random forest model on the fly...")
                    data_path = "data/sample_travel_data.csv"
                    if os.path.exists(data_path):
                        self.predictor.train(data_path)
                    else:
                        print("Warning: No training data found. Using basic ETA calculation.")
                        self.predictor = None
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using basic ETA calculation instead.")
            self.predictor = None
    
    def add_patient(self, patient: Patient):
        """
        Add a new patient to the queue.
        
        Args:
            patient (Patient): Patient to add
        """
        patient.calculate_distance_to_hospital(self.hospital_lat, self.hospital_lon)
        self.patients.append(patient)
        print(f"Added {patient.patient_id} to queue (Distance: {patient.distance_km:.2f} km)")
    
    def remove_arrived_patients(self) -> List[Patient]:
        """
        Remove patients who have arrived (ETA <= 0).
        
        Returns:
            List[Patient]: List of removed patients
        """
        arrived_patients = [p for p in self.patients if p.eta_minutes <= 0]
        self.patients = [p for p in self.patients if p.eta_minutes > 0]
        
        for patient in arrived_patients:
            print(f"Patient {patient.patient_id} has arrived and been removed from queue")
        
        return arrived_patients
    
    def update_patient_etas(self, time_elapsed_minutes: float = 2.0):
        """
        Update ETA for all patients in the queue.
        
        Args:
            time_elapsed_minutes (float): Time elapsed since last update
        """
        for patient in self.patients:
            if self.predictor:
                # Use ML model for prediction
                try:
                    new_eta = self.predictor.predict(
                        patient.distance_km, patient.speed, patient.travel_mode
                    )
                    # Apply time decay
                    patient.eta_minutes = max(0, new_eta - time_elapsed_minutes)
                except Exception as e:
                    print(f"Error in ML prediction for {patient.patient_id}: {e}")
                    # Fallback to basic calculation
                    patient.update_eta(time_elapsed_minutes)
            else:
                # Basic ETA calculation
                patient.update_eta(time_elapsed_minutes)
    
    def get_sorted_queue(self) -> List[Patient]:
        """
        Get patients sorted by ETA (ascending).
        
        Returns:
            List[Patient]: Sorted list of patients
        """
        return sorted(self.patients, key=lambda p: p.eta_minutes)
    
    def print_queue_status(self):
        """Print current queue status in a formatted table."""
        if not self.patients:
            print("Queue is empty.")
            return
        
        sorted_patients = self.get_sorted_queue()
        
        print("\n" + "="*80)
        print(f"PATIENT QUEUE STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print(f"Model: {self.model_type.upper()}")
        print("="*80)
        print(f"{'Rank':<6}{'Patient ID':<12}{'ETA (min)':<12}{'Travel Mode':<12}{'Distance (km)':<15}")
        print("-"*80)
        
        for i, patient in enumerate(sorted_patients, 1):
            print(f"{i:<6}{patient.patient_id:<12}{patient.eta_minutes:<12.1f}"
                  f"{patient.travel_mode:<12}{patient.distance_km:<15.2f}")
        
        print("-"*80)
        print(f"Total patients in queue: {len(sorted_patients)}")
        print("="*80)
    
    def generate_new_patients(self, num_patients: int = None) -> List[Patient]:
        """
        Generate new patients for the simulation.
        
        Args:
            num_patients (int): Number of patients to generate (random if None)
        
        Returns:
            List[Patient]: Generated patients
        """
        if num_patients is None:
            num_patients = random.randint(1, 3)  # Randomly add 1-3 patients
        
        new_patients = []
        for i in range(num_patients):
            patient_id = f"P{len(self.patients) + i + 1:03d}"
            patient_data = self.geo_simulator.generate_patient_data(patient_id)
            
            patient = Patient(
                patient_id=patient_id,
                latitude=patient_data['latitude'],
                longitude=patient_data['longitude'],
                speed=patient_data['speed_kmh'],
                travel_mode=patient_data['travel_mode'],
                eta_minutes=patient_data['eta_minutes']
            )
            
            new_patients.append(patient)
        
        return new_patients


def run_simulation(duration_minutes: int = 20, update_interval_minutes: float = 2.0, 
                  model_type: str = "random_forest"):
    """
    Run the patient queue management simulation.
    
    Args:
        duration_minutes (int): Total simulation duration
        update_interval_minutes (float): Update interval in minutes
        model_type (str): ML model type ('random_forest' or 'stacking')
    """
    print("Starting Patient Queue Management System Simulation")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Update Interval: {update_interval_minutes} minutes")
    print(f"Model Type: {model_type}")
    print("-" * 60)
    
    # Initialize queue manager
    queue_manager = PatientQueueManager()
    queue_manager.load_model(model_type=model_type)
    
    # Generate initial patients (5-10)
    initial_patient_count = random.randint(5, 10)
    print(f"Generating {initial_patient_count} initial patients...")
    
    initial_patients = queue_manager.generate_new_patients(initial_patient_count)
    for patient in initial_patients:
        queue_manager.add_patient(patient)
    
    # Simulation loop
    elapsed_time = 0.0
    update_count = 0
    
    try:
        while elapsed_time < duration_minutes:
            update_count += 1
            
            # Update patient ETAs
            queue_manager.update_patient_etas(update_interval_minutes)
            
            # Remove arrived patients
            queue_manager.remove_arrived_patients()
            
            # Occasionally add new patients
            if update_count % 3 == 0:  # Every 3 updates (6 minutes)
                new_patients = queue_manager.generate_new_patients()
                for patient in new_patients:
                    queue_manager.add_patient(patient)
            
            # Print queue status
            queue_manager.print_queue_status()
            
            # Wait for next update (simulate real-time)
            if elapsed_time + update_interval_minutes < duration_minutes:
                print(f"\nWaiting {update_interval_minutes} minutes for next update...")
                time.sleep(2)  # Sleep for 2 seconds to simulate time passing
            
            elapsed_time += update_interval_minutes
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    print(f"\nSimulation completed after {elapsed_time:.1f} minutes.")
    print(f"Final queue status:")
    queue_manager.print_queue_status()


def train_models():
    """Train and save ML models if training data exists."""
    data_path = "data/sample_travel_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Training data not found: {data_path}")
        return
    
    print("Training machine learning models...")
    
    # Train Random Forest
    try:
        from models import train_and_save_model
        rf_predictor = train_and_save_model(data_path, "random_forest_model.joblib")
        print("Random Forest model trained and saved.")
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Train Stacking Model
    try:
        from models import train_and_save_stacking_model
        stacking_predictor = train_and_save_stacking_model(data_path, "stacking_model.joblib")
        print("Stacking model trained and saved.")
    except Exception as e:
        print(f"Error training Stacking model: {e}")


def main():
    """Main function to run the patient queue management system."""
    print("Intelligent Patient Queue Management System")
    print("==========================================")
    
    # Check if we should train models first
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_models()
        return
    
    # Configuration
    duration = 20  # minutes
    interval = 2.0  # minutes
    model_type = "stacking"  # or "random_forest"
    
    # Override with command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["random_forest", "stacking"]:
            model_type = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
        except ValueError:
            print("Invalid duration. Using default 20 minutes.")
    
    # Run simulation
    run_simulation(duration, interval, model_type)


if __name__ == "__main__":
    main()

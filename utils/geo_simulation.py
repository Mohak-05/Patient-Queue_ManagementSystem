"""
Geographic simulation utility for generating patient locations and travel modes.

This module provides functions to generate simulated patient GPS coordinates,
travel modes, and related parameters for testing the queue management system.
"""

import random
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from .haversine import haversine_distance
except ImportError:
    from haversine import haversine_distance


@dataclass
class TravelModeConfig:
    """Configuration for different travel modes."""
    name: str
    min_speed: float
    max_speed: float
    probability: float


# Travel mode configurations (speeds in km/h)
TRAVEL_MODES = [
    TravelModeConfig("Walk", 3.0, 7.0, 0.3),
    TravelModeConfig("Bike", 10.0, 20.0, 0.3),
    TravelModeConfig("Car", 25.0, 60.0, 0.4)
]


class GeoSimulator:
    """Simulates geographic data for patient queue management."""
    
    def __init__(self, hospital_lat: float = 40.7128, hospital_lon: float = -74.0060):
        """
        Initialize the geo simulator with hospital location.
        
        Args:
            hospital_lat (float): Hospital latitude (default: NYC coordinates)
            hospital_lon (float): Hospital longitude (default: NYC coordinates)
        """
        self.hospital_lat = hospital_lat
        self.hospital_lon = hospital_lon
        random.seed(42)  # For reproducible results
    
    def generate_random_coordinate_nearby(self, max_radius_km: float = 10.0) -> Tuple[float, float]:
        """
        Generate a random coordinate within specified radius of the hospital.
        
        Args:
            max_radius_km (float): Maximum radius in kilometers from hospital
        
        Returns:
            Tuple[float, float]: Generated (latitude, longitude) coordinates
        """
        # Generate random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0.1, max_radius_km)
        
        # Convert distance to degrees (approximate)
        # 1 degree ≈ 111 km at equator
        lat_offset = (distance * math.cos(angle)) / 111.0
        lon_offset = (distance * math.sin(angle)) / (111.0 * math.cos(math.radians(self.hospital_lat)))
        
        new_lat = self.hospital_lat + lat_offset
        new_lon = self.hospital_lon + lon_offset
        
        return new_lat, new_lon
    
    def select_travel_mode(self) -> TravelModeConfig:
        """
        Select a travel mode based on probability weights.
        
        Returns:
            TravelModeConfig: Selected travel mode configuration
        """
        total_prob = sum(mode.probability for mode in TRAVEL_MODES)
        rand_val = random.uniform(0, total_prob)
        
        cumulative_prob = 0
        for mode in TRAVEL_MODES:
            cumulative_prob += mode.probability
            if rand_val <= cumulative_prob:
                return mode
        
        return TRAVEL_MODES[-1]  # Fallback to last mode
    
    def generate_patient_data(self, patient_id: str, max_radius_km: float = 10.0) -> Dict[str, Any]:
        """
        Generate complete patient travel data.
        
        Args:
            patient_id (str): Unique patient identifier
            max_radius_km (float): Maximum radius for patient location
        
        Returns:
            Dict[str, Any]: Patient data including location, travel mode, speed, distance, ETA
        """
        # Generate patient location
        patient_lat, patient_lon = self.generate_random_coordinate_nearby(max_radius_km)
        
        # Calculate distance to hospital
        distance_km = haversine_distance(patient_lat, patient_lon, 
                                       self.hospital_lat, self.hospital_lon)
        
        # Select travel mode and generate speed
        travel_mode = self.select_travel_mode()
        speed_kmh = random.uniform(travel_mode.min_speed, travel_mode.max_speed)
        
        # Calculate ETA in minutes
        eta_minutes = (distance_km / speed_kmh) * 60
        
        # Add some realistic variation (traffic, delays, etc.)
        eta_minutes *= random.uniform(0.8, 1.3)
        
        return {
            'patient_id': patient_id,
            'latitude': patient_lat,
            'longitude': patient_lon,
            'distance_km': distance_km,
            'travel_mode': travel_mode.name,
            'speed_kmh': speed_kmh,
            'eta_minutes': eta_minutes,
            'hospital_lat': self.hospital_lat,
            'hospital_lon': self.hospital_lon
        }
    
    def generate_multiple_patients(self, num_patients: int, 
                                 max_radius_km: float = 10.0) -> List[Dict[str, Any]]:
        """
        Generate data for multiple patients.
        
        Args:
            num_patients (int): Number of patients to generate
            max_radius_km (float): Maximum radius for patient locations
        
        Returns:
            List[Dict[str, Any]]: List of patient data dictionaries
        """
        patients = []
        for i in range(num_patients):
            patient_id = f"P{i+1:03d}"
            patient_data = self.generate_patient_data(patient_id, max_radius_km)
            patients.append(patient_data)
        
        return patients
    
    def update_patient_location(self, patient_data: Dict[str, Any], 
                              time_elapsed_minutes: float) -> Dict[str, Any]:
        """
        Update patient location based on elapsed time and travel speed.
        
        Args:
            patient_data (Dict[str, Any]): Current patient data
            time_elapsed_minutes (float): Time elapsed in minutes
        
        Returns:
            Dict[str, Any]: Updated patient data
        """
        # Calculate distance traveled
        speed_kmh = patient_data['speed_kmh']
        distance_traveled = (speed_kmh * time_elapsed_minutes) / 60.0  # Convert to km
        
        # Update ETA
        new_eta = patient_data['eta_minutes'] - time_elapsed_minutes
        patient_data['eta_minutes'] = max(0, new_eta)
        
        # Update distance (simple approximation)
        current_distance = patient_data['distance_km']
        new_distance = max(0, current_distance - distance_traveled)
        patient_data['distance_km'] = new_distance
        
        return patient_data


def create_sample_dataset(num_samples: int = 100, filename: str = None) -> List[Dict[str, Any]]:
    """
    Create a sample dataset for training ML models.
    
    Args:
        num_samples (int): Number of samples to generate
        filename (str): Optional filename to save the dataset
    
    Returns:
        List[Dict[str, Any]]: Generated dataset
    """
    simulator = GeoSimulator()
    dataset = simulator.generate_multiple_patients(num_samples, max_radius_km=15.0)
    
    if filename:
        import pandas as pd
        df = pd.DataFrame([{
            'Distance': data['distance_km'],
            'Speed': data['speed_kmh'],
            'TravelMode': data['travel_mode'],
            'ETA': data['eta_minutes']
        } for data in dataset])
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
    
    return dataset


if __name__ == "__main__":
    # Test the geo simulation
    simulator = GeoSimulator()
    
    # Generate sample patients
    patients = simulator.generate_multiple_patients(5)
    
    print("Generated Patient Data:")
    print("-" * 80)
    for patient in patients:
        print(f"Patient {patient['patient_id']}:")
        print(f"  Location: ({patient['latitude']:.4f}, {patient['longitude']:.4f})")
        print(f"  Distance: {patient['distance_km']:.2f} km")
        print(f"  Travel Mode: {patient['travel_mode']}")
        print(f"  Speed: {patient['speed_kmh']:.1f} km/h")
        print(f"  ETA: {patient['eta_minutes']:.1f} minutes")
        print()
    
    # Create sample dataset
    print("Creating sample dataset...")
    create_sample_dataset(50, "sample_geo_data.csv")

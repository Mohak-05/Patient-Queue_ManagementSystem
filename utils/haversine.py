"""
Haversine distance calculation utility.

This module provides functions to calculate the great-circle distance
between two points on Earth given their latitude and longitude coordinates.
"""

import math
from typing import Tuple


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the Haversine formula to calculate the distance between two points
    on the Earth's surface given their latitude and longitude coordinates.
    
    Args:
        lat1 (float): Latitude of the first point in decimal degrees
        lon1 (float): Longitude of the first point in decimal degrees
        lat2 (float): Latitude of the second point in decimal degrees
        lon2 (float): Longitude of the second point in decimal degrees
    
    Returns:
        float: Distance between the two points in kilometers
    
    Example:
        >>> distance = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        >>> print(f"Distance: {distance:.2f} km")
        Distance: 3944.42 km
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    earth_radius = 6371.0
    
    # Calculate the distance
    distance = earth_radius * c
    
    return distance


def calculate_distance_from_coordinates(coord1: Tuple[float, float], 
                                      coord2: Tuple[float, float]) -> float:
    """
    Calculate distance between two coordinate tuples.
    
    Args:
        coord1 (Tuple[float, float]): First coordinate as (latitude, longitude)
        coord2 (Tuple[float, float]): Second coordinate as (latitude, longitude)
    
    Returns:
        float: Distance in kilometers
    
    Example:
        >>> hospital_coord = (40.7128, -74.0060)
        >>> patient_coord = (40.7589, -73.9851)
        >>> distance = calculate_distance_from_coordinates(patient_coord, hospital_coord)
        >>> print(f"Distance to hospital: {distance:.2f} km")
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return haversine_distance(lat1, lon1, lat2, lon2)


if __name__ == "__main__":
    # Test the haversine distance calculation
    # Example: Distance between New York City and Los Angeles
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437
    
    distance = haversine_distance(nyc_lat, nyc_lon, la_lat, la_lon)
    print(f"Distance between NYC and LA: {distance:.2f} km")
    
    # Test with coordinate tuples
    hospital = (40.7128, -74.0060)
    patient = (40.7589, -73.9851)
    distance = calculate_distance_from_coordinates(patient, hospital)
    print(f"Distance from patient to hospital: {distance:.2f} km")

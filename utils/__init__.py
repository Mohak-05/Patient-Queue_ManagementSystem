"""
Utility modules for Patient Queue Management System.

This package contains utility functions for geographic calculations.
"""

from .haversine import haversine_distance, calculate_distance_from_coordinates

__all__ = [
    'haversine_distance',
    'calculate_distance_from_coordinates'
]

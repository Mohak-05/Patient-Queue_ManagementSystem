# Patient Queue Management System

## Dataset Generation

### Original Travel Dataset (4,414 records)

**Method**: Realistic multi-factor modeling

- **Distance**: Exponential/Gamma/Normal distributions by travel mode
- **Speed**: Normal distribution with traffic/weather factors
- **ETA**: Distance/speed + mode-specific delays + random variation

**Travel Mode Parameters:**
| Mode | Speed Range (km/h) | Max Distance (km) | Distribution |
|------|-------------------|-------------------|--------------|
| Walk | 3.0 - 7.0 | 3.0 | 25% |
| Bike | 8.0 - 25.0 | 8.0 | 20% |
| Car | 15.0 - 80.0 | 25.0 | 45% |
| PublicTransport | 12.0 - 40.0 | 20.0 | 10% |

### GPS Tracking Dataset (17,656 files)

**Method**: Realistic path simulation with 2-minute GPS updates

**Geolocation Generation:**

- **Starting Points**: Trigonometric distance/angle conversion from hospital (40.7128, -74.0060)
  - Walk/Bike: Uniform random angles (0-360°)
  - Car: Major road bias (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° ± 15°)
  - PublicTransport: Transit corridor bias (15°, 75°, 105°, 165°, 195°, 255°, 285°, 345° ± 10°)
- **Coordinate Formula**: `lat_offset = (distance × cos(angle)) / 111.0`, `lon_offset = (distance × sin(angle)) / (111.0 × cos(hospital_lat))`
- **Path Simulation**: Linear interpolation + perpendicular road curves
  - Curve patterns: Walk (random offsets), Bike (gentle curves), Car (multiple road curves), PublicTransport (stop-based deviations)

**Structure:**

- `data/geolocations/with_traffic/` - GPS tracking with traffic delays
- `data/geolocations/no_traffic/` - GPS tracking baseline
- `data/timestamps/with_traffic/` - Time analysis with traffic
- `data/timestamps/no_traffic/` - Time analysis baseline

**Parameters:**

- **Hospital Location**: 40.7128, -74.0060 (fixed)
- **Update Interval**: 2 minutes
- **Road Variation**: Walk 15%, Bike 20%, Car 35%, PublicTransport 40%
- **Distance Accuracy**: ±10-20% variation from original dataset
- **Traffic Delays**: Rush hours 2.0-2.2×, Lunch 1.3×, Normal 1.0×
- **File Organization**: 1,000 files per batch directory

**Output Format:**

- **Geolocations**: journey_id, timestamp_minutes, latitude, longitude, distance_remaining_km, eta_remaining_minutes
- **Timestamps**: journey_id, update_sequence, elapsed_minutes, time_of_day_hour, is_rush_hour, original_eta_minutes, actual_duration_minutes, time_variance_percent, travel_mode, original_distance_km, original_speed_kmh

# Intelligent Patient Queue Management System

A Machine Learning-based patient queue management system that predicts patient arrival times using location-based ETA prediction with travel mode consideration.

## 🏥 System Overview

This system simulates real-time patient queue updates by predicting ETAs based on:

- **Patient travel mode**: Walk, Bike, Car
- **Speed**: Variable speed based on travel mode (km/h)
- **Distance**: Calculated using Haversine formula
- **ETA**: Predicted using machine learning models (target variable in minutes)

## 🗂️ Project Structure

```
Patient-Queue_ManagementSystem/
├── data/
│   └── sample_travel_data.csv      # Sample training data
├── utils/
│   ├── __init__.py                 # Package initialization
│   ├── haversine.py               # Distance calculation using Haversine formula
│   └── geo_simulation.py          # GPS coordinate and travel mode simulation
├── models/
│   ├── __init__.py                # Package initialization
│   ├── train_random_forest.py    # Random Forest regressor
│   └── stacking_model.py          # Stacking ensemble model
├── main.py                        # Main simulation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Features

### Core Components

1. **Data Generation**

   - Simulated patient GPS coordinates
   - Travel mode selection (Walk/Bike/Car with realistic speeds)
   - Distance calculation using Haversine formula

2. **Machine Learning Models**

   - **Random Forest Regressor**: Primary ETA prediction model
   - **Stacking Ensemble**: Combines RandomForest, KNN, and LinearRegression
   - Feature engineering with travel mode encoding and scaling

3. **Real-time Simulation**

   - Queue updates every 2 minutes
   - Dynamic patient arrival and departure
   - Sorted queue display by ETA

4. **Queue Management**
   - Auto-removal of patients when ETA ≤ 0
   - CLI printout: [PatientID, ETA, TravelMode, Distance]
   - Real-time queue ranking

## 📋 Installation & Setup

### 1. Clone/Download the Project

```bash
cd Patient-Queue_ManagementSystem
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models (Optional)

```bash
python main.py --train
```

## 🎮 Usage

### Basic Simulation (20 minutes, Stacking model)

```bash
python main.py
```

### Using Random Forest Model

```bash
python main.py random_forest
```

### Custom Duration (30 minutes)

```bash
python main.py stacking 30
```

### Train Models Only

```bash
python main.py --train
```

## 🔧 Configuration

### Travel Mode Configurations

- **Walk**: 3-7 km/h (30% probability)
- **Bike**: 10-20 km/h (30% probability)
- **Car**: 25-60 km/h (40% probability)

### Default Settings

- **Hospital Location**: NYC coordinates (40.7128, -74.0060)
- **Update Interval**: 2 minutes
- **Patient Radius**: 10 km from hospital
- **Initial Patients**: 5-10 random patients

## 🧠 Machine Learning Models

### 1. Random Forest Regressor

- **Features**: Distance, Speed, Travel Mode (encoded)
- **Hyperparameters**: 100 estimators, max_depth=10
- **Performance Metrics**: MAE, RMSE, R²

### 2. Stacking Ensemble

- **Base Models**:
  - Random Forest (50 estimators)
  - K-Nearest Neighbors (k=5)
  - Linear Regression
- **Meta-learner**: Linear Regression
- **Cross-validation**: 5-fold CV

## 📊 Sample Output

```
================================================================================
PATIENT QUEUE STATUS - 14:32:15
Model: STACKING
================================================================================
Rank  Patient ID  ETA (min)   Travel Mode Distance (km)
--------------------------------------------------------------------------------
1     P003        2.1         Car         1.8
2     P007        3.4         Bike        0.9
3     P001        4.2         Walk        0.3
4     P005        5.8         Car         2.4
5     P002        7.1         Bike        1.5
--------------------------------------------------------------------------------
Total patients in queue: 5
================================================================================
```

## 🔍 Key Functions

### Utils Package

- `haversine_distance()`: Calculate great-circle distance
- `GeoSimulator.generate_patient_data()`: Create realistic patient scenarios
- `create_sample_dataset()`: Generate training data

### Models Package

- `RandomForestETAPredictor.train()`: Train Random Forest model
- `StackingETAPredictor.predict()`: Ensemble ETA prediction
- `save_model()` / `load_model()`: Model persistence

### Main Simulation

- `PatientQueueManager`: Core queue management logic
- `Patient`: Individual patient representation
- `run_simulation()`: Main simulation loop

## 🎯 Simulation Flow

1. **Initialize**: Load/train ML model, generate initial patients (5-10)
2. **Update Loop** (every 2 minutes):
   - Update patient ETAs using ML prediction
   - Remove arrived patients (ETA ≤ 0)
   - Add new patients periodically
   - Display sorted queue
3. **Real-time Display**: Show current queue with rankings

## 📈 Model Performance

The system tracks and displays:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) score
- Cross-validation scores
- Individual base model performance (for Stacking)

## 🛠️ Customization

### Adding New Travel Modes

Edit `utils/geo_simulation.py`:

```python
TRAVEL_MODES.append(
    TravelModeConfig("Scooter", 15.0, 25.0, 0.1)
)
```

### Changing Hospital Location

Modify coordinates in `main.py`:

```python
queue_manager = PatientQueueManager(lat=your_lat, lon=your_lon)
```

### Model Hyperparameters

Adjust parameters in model files:

```python
RandomForestRegressor(n_estimators=200, max_depth=15)
```

## 🔬 Testing Individual Components

### Test Haversine Distance

```bash
cd utils
python haversine.py
```

### Test Geo Simulation

```bash
cd utils
python geo_simulation.py
```

### Test Random Forest Model

```bash
cd models
python train_random_forest.py
```

### Test Stacking Model

```bash
cd models
python stacking_model.py
```

## 📊 Data Format

The system uses CSV data with columns:

- `Distance`: Distance in kilometers (float)
- `Speed`: Speed in km/h (float)
- `TravelMode`: Walk/Bike/Car (string)
- `ETA`: Target variable in minutes (float)

## 🚦 Error Handling

- **Missing Models**: Trains on-the-fly or uses basic ETA calculation
- **Invalid Predictions**: Falls back to time-based ETA updates
- **Data Issues**: Handles unseen travel modes gracefully
- **Interruption**: Graceful shutdown with Ctrl+C

## 🤝 Contributing

To extend the system:

1. Add new travel modes in `geo_simulation.py`
2. Implement additional ML models in `models/`
3. Enhance visualization in `main.py`
4. Add new distance calculation methods in `haversine.py`

## 📝 License

This project is created for educational and research purposes.

---

**Note**: This system simulates a patient queue management scenario and should be adapted with real-world considerations for production use, including privacy, security, and regulatory compliance.

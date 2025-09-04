"""
Federated Learning Data Splitter
Splits the main dataset into three hospital datasets while maintaining geo-location consistency
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_dataset_by_hospitals():
    """Split main dataset into 3 hospital datasets"""
    print("Hospital Data Splitting - Federated Learning...")
    
    # Load main dataset
    df = pd.read_csv('../data/travel_data.csv')
    print(f"Total dataset size: {len(df)} records")
    
    # Create stratified split to maintain travel mode distribution
    # First split: Hospital A gets 40%, remaining 60% for B and C
    hospital_a, temp_bc = train_test_split(
        df, test_size=0.6, random_state=42, 
        stratify=df['TravelMode']
    )
    
    # Second split: Hospital B gets 30%, Hospital C gets 30%
    hospital_b, hospital_c = train_test_split(
        temp_bc, test_size=0.5, random_state=42,
        stratify=temp_bc['TravelMode']
    )
    
    # Create hospital_data directory
    os.makedirs('hospital_data', exist_ok=True)
    
    # Save hospital datasets with journey IDs preserved
    hospitals = {
        'hospital_a': hospital_a,
        'hospital_b': hospital_b, 
        'hospital_c': hospital_c
    }
    
    print("\nHospital Dataset Distribution:")
    print("="*50)
    
    # Create journey ID mapping
    journey_mapping = {}
    
    for name, data in hospitals.items():
        # Add journey_id column (original index + 1)
        data_with_ids = data.copy()
        data_with_ids['journey_id'] = data.index + 1
        
        # Save hospital dataset with journey IDs
        data_with_ids.to_csv(f'hospital_data/{name}_data.csv', index=False)
        
        # Store mapping
        for journey_id in data_with_ids['journey_id']:
            journey_mapping[int(journey_id)] = name
        
        # Print statistics
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(data)} ({len(data)/len(df)*100:.1f}%)")
        print(f"  Travel Mode Distribution:")
        mode_dist = data['TravelMode'].value_counts()
        for mode, count in mode_dist.items():
            print(f"    {mode}: {count} ({count/len(data)*100:.1f}%)")
        print(f"  Distance Range: {data['Distance'].min():.2f} - {data['Distance'].max():.2f} km")
        print(f"  Speed Range: {data['Speed'].min():.1f} - {data['Speed'].max():.1f} km/h")
        print(f"  ETA Range: {data['ETA'].min():.1f} - {data['ETA'].max():.1f} min")
    
    # Save journey mapping
    import json
    with open('journey_hospital_mapping.json', 'w') as f:
        json.dump(journey_mapping, f, indent=2)
    
    print(f"\n[SUCCESS] Hospital datasets saved to hospital_data/")
    print(f"[SUCCESS] Journey mapping saved ({len(journey_mapping)} journeys)")
    return hospitals

def create_hospital_gps_splits():
    """Create corresponding GPS data splits for each hospital"""
    print("\n[GPS] Creating GPS data splits for hospitals...")
    
    # Load journey mapping
    import json
    with open('journey_hospital_mapping.json', 'r') as f:
        journey_mapping = json.load(f)
    
    # Convert string keys to int
    journey_mapping = {int(k): v for k, v in journey_mapping.items()}
    
    # Group journey IDs by hospital
    hospitals = {}
    for journey_id, hospital in journey_mapping.items():
        if hospital not in hospitals:
            hospitals[hospital] = set()
        hospitals[hospital].add(journey_id)
    
    # Create GPS directories for each hospital
    for hospital in hospitals.keys():
        os.makedirs(f'hospital_data/{hospital}_gps/with_traffic', exist_ok=True)
        os.makedirs(f'hospital_data/{hospital}_gps/no_traffic', exist_ok=True)
        os.makedirs(f'hospital_data/{hospital}_timestamps/with_traffic', exist_ok=True)
        os.makedirs(f'hospital_data/{hospital}_timestamps/no_traffic', exist_ok=True)
    
    # Copy GPS files for each hospital's journey IDs
    import shutil
    import glob
    
    total_files_copied = 0
    
    for traffic_type in ['with_traffic', 'no_traffic']:
        # Get all GPS files
        gps_files = glob.glob(f'../data/geolocations/{traffic_type}/**/*.csv', recursive=True)
        timestamp_files = glob.glob(f'../data/timestamps/{traffic_type}/**/*.csv', recursive=True)
        
        for hospital, journey_ids in hospitals.items():
            hospital_gps_copied = 0
            hospital_timestamp_copied = 0
            
            # Copy GPS files
            for gps_file in gps_files:
                journey_id = int(os.path.basename(gps_file).replace('journey_', '').replace('.csv', ''))
                if journey_id in journey_ids:
                    dest_dir = f'hospital_data/{hospital}_gps/{traffic_type}'
                    # Maintain batch structure
                    batch_dir = os.path.basename(os.path.dirname(gps_file))
                    os.makedirs(f'{dest_dir}/{batch_dir}', exist_ok=True)
                    shutil.copy2(gps_file, f'{dest_dir}/{batch_dir}/')
                    hospital_gps_copied += 1
            
            # Copy timestamp files  
            for ts_file in timestamp_files:
                journey_id = int(os.path.basename(ts_file).replace('journey_', '').replace('.csv', ''))
                if journey_id in journey_ids:
                    dest_dir = f'hospital_data/{hospital}_timestamps/{traffic_type}'
                    batch_dir = os.path.basename(os.path.dirname(ts_file))
                    os.makedirs(f'{dest_dir}/{batch_dir}', exist_ok=True)
                    shutil.copy2(ts_file, f'{dest_dir}/{batch_dir}/')
                    hospital_timestamp_copied += 1
            
            total_files_copied += hospital_gps_copied
            print(f"  {hospital}: {hospital_gps_copied} GPS files, {hospital_timestamp_copied} timestamp files ({traffic_type})")
    
    print(f"\n[SUCCESS] Total GPS files distributed: {total_files_copied}")

if __name__ == "__main__":
    # Split main dataset
    hospitals = split_dataset_by_hospitals()
    
    # Create GPS splits
    create_hospital_gps_splits()
    
    print("\n[COMPLETE] Federated learning data preparation complete!")

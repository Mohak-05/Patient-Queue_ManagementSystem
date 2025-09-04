"""
Create journey ID mapping for federated learning
Maps each journey ID to its corresponding hospital
"""

import pandas as pd
import json

def create_journey_mapping():
    """Create mapping of journey IDs to hospitals"""
    print("🗺️ Creating journey ID to hospital mapping...")
    
    mapping = {}
    
    # Load each hospital's data and map journey IDs
    for hospital in ['hospital_a', 'hospital_b', 'hospital_c']:
        df = pd.read_csv(f'federated_learning/hospital_data/{hospital}_data.csv')
        
        # Journey IDs are the original index + 1 (since they were 0-indexed in splitting)
        for idx in df.index:
            journey_id = idx + 1  # Convert to 1-indexed journey ID
            mapping[journey_id] = hospital
    
    # Save mapping
    with open('federated_learning/journey_hospital_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Mapping created for {len(mapping)} journeys")
    print("📊 Hospital distribution:")
    hospital_counts = {}
    for hospital in mapping.values():
        hospital_counts[hospital] = hospital_counts.get(hospital, 0) + 1
    
    for hospital, count in hospital_counts.items():
        print(f"  {hospital}: {count} journeys")
    
    return mapping

if __name__ == "__main__":
    create_journey_mapping()

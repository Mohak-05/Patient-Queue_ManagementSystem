#!/usr/bin/env python3
"""
Federated Learning Pipeline Runner
=================================

This script runs the complete federated learning pipeline:
1. Split data into hospital datasets
2. Train individual hospital models
3. Aggregate models into federated versions

Usage: python run_federated_pipeline.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """Print pipeline banner"""
    print("=" * 60)
    print("FEDERATED LEARNING PIPELINE")
    print("   Patient Queue Management System")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def run_step(step_name, script_path, description):
    """Run a pipeline step"""
    print(f"Step: {step_name}")
    print(f"Description: {description}")
    print(f"Running: {script_path}")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              check=True, capture_output=True, text=True)
        
        # Print key outputs (last few lines)
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) > 5:
            print("...")
            for line in output_lines[-5:]:
                print(line)
        else:
            print(result.stdout.strip())
        
        elapsed = time.time() - start_time
        print(f"SUCCESS: {step_name} completed in {elapsed:.1f}s")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {step_name} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    
    print()
    return True

def check_outputs():
    """Check if pipeline outputs exist"""
    print("Checking Pipeline Outputs:")
    print("-" * 40)
    
    outputs = {
        'Hospital A Data': 'hospital_data/hospital_a_data.csv',
        'Hospital B Data': 'hospital_data/hospital_b_data.csv', 
        'Hospital C Data': 'hospital_data/hospital_c_data.csv',
        'Journey Mapping': 'journey_hospital_mapping.json',
        'Hospital A Models': 'hospital_models/hospital_a/random_forest_model.pkl',
        'Hospital B Models': 'hospital_models/hospital_b/random_forest_model.pkl',
        'Hospital C Models': 'hospital_models/hospital_c/random_forest_model.pkl',
        'Federated RF': 'global_models/federated_random_forest.pkl',
        'Federated MLP': 'global_models/federated_mlp.pkl',
        'Federated Encoder': 'global_models/federated_encoder.pkl'
    }
    
    all_good = True
    for name, path in outputs.items():
        if os.path.exists(path):
            print(f"SUCCESS: {name}")
        else:
            print(f"MISSING: {name} - Missing: {path}")
            all_good = False
    
    print()
    return all_good

def main():
    """Run the complete federated learning pipeline"""
    print_banner()
    
    # Pipeline steps
    steps = [
        ("Data Splitting", "data_splitter.py", "Split main dataset into hospital-specific subsets"),
        ("Hospital Training", "hospital_trainer.py", "Train individual models for each hospital"),
        ("Model Aggregation", "aggregator.py", "Aggregate hospital models into federated versions")
    ]
    
    # Run each step
    for step_name, script, description in steps:
        success = run_step(step_name, script, description)
        if not success:
            print("PIPELINE FAILED! Pipeline failed at step:", step_name)
            return False
    
    # Check outputs
    success = check_outputs()
    
    if success:
        print("SUCCESS! FEDERATED LEARNING PIPELINE COMPLETED!")
        print()
        print("Next Steps:")
        print("1. Analyze journeys: python ../federated_model_analyzer.py")
        print("2. Quick test: python ../test_federated_analyzer.py")
        print("3. View models: ls global_models/")
        print()
        print("Ready for federated patient queue management!")
    else:
        print("INCOMPLETE: Pipeline completed with some missing outputs.")
        print("   Check the error messages above.")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

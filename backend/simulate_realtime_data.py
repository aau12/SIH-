"""
Real-Time Data Simulator
=========================
Simulates new GNSS data arriving every 15 minutes by appending to the cleaned dataset.

Usage:
    python simulate_realtime_data.py --satellite MEO --duration 60
"""

import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = Path("data/processed")

def simulate_new_sample(last_row, timestamp):
    """
    Simulate a new GNSS sample based on the last row.
    
    Args:
        last_row: Previous data row
        timestamp: New timestamp
        
    Returns:
        New row with simulated data
    """
    new_row = last_row.copy()
    
    # Add small random variations (realistic drift)
    error_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
    
    for col in error_cols:
        if col in new_row:
            # Add random walk with small drift
            drift = np.random.normal(0, 0.05)  # Small random change
            new_row[col] = new_row[col] + drift
    
    return new_row


def append_new_sample(satellite_type):
    """
    Append a new simulated sample to the cleaned dataset.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
    """
    data_file = PROCESSED_DATA_DIR / f"{satellite_type}_clean_15min.csv"
    
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        return False
    
    # Read existing data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Get last timestamp and row
    last_timestamp = df.index[-1]
    last_row = df.iloc[-1]
    
    # Generate new timestamp (15 minutes later)
    new_timestamp = last_timestamp + timedelta(minutes=15)
    
    # Simulate new sample
    new_row = simulate_new_sample(last_row, new_timestamp)
    
    # Create new dataframe row
    new_df = pd.DataFrame([new_row], index=[new_timestamp])
    
    # Append to existing data
    df_updated = pd.concat([df, new_df])
    
    # Save back to file
    df_updated.to_csv(data_file)
    
    print(f"✓ [{datetime.now().strftime('%H:%M:%S')}] Appended new sample for {satellite_type}")
    print(f"  Timestamp: {new_timestamp}")
    print(f"  x_error: {new_row['x_error (m)']:.4f}m")
    print(f"  y_error: {new_row['y_error (m)']:.4f}m")
    print(f"  z_error: {new_row['z_error (m)']:.4f}m")
    print(f"  satclockerror: {new_row['satclockerror (m)']:.4f}m")
    
    return True


def run_simulation(satellite_type, duration_minutes, interval_minutes=15):
    """
    Run continuous data simulation.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        duration_minutes: How long to run (minutes)
        interval_minutes: Interval between samples (default 15)
    """
    print(f"\n{'='*60}")
    print(f"REAL-TIME DATA SIMULATOR - {satellite_type}")
    print(f"{'='*60}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Samples to generate: {duration_minutes // interval_minutes}")
    print(f"Press Ctrl+C to stop early")
    print(f"{'='*60}\n")
    
    interval_seconds = interval_minutes * 60
    num_samples = duration_minutes // interval_minutes
    
    try:
        for i in range(num_samples):
            print(f"\n→ Sample {i+1}/{num_samples}")
            
            success = append_new_sample(satellite_type)
            
            if not success:
                print("✗ Failed to append sample")
                break
            
            if i < num_samples - 1:
                print(f"\n  Waiting {interval_minutes} minutes...")
                time.sleep(interval_seconds)
        
        print(f"\n{'='*60}")
        print(f"✓ Simulation completed")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("Simulation stopped by user")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Simulate real-time GNSS data')
    parser.add_argument('--satellite', type=str, choices=['MEO', 'GEO'], required=True,
                       help='Satellite type')
    parser.add_argument('--duration', type=int, default=60,
                       help='Simulation duration in minutes (default: 60)')
    parser.add_argument('--interval', type=int, default=15,
                       help='Interval between samples in minutes (default: 15)')
    
    args = parser.parse_args()
    
    run_simulation(args.satellite, args.duration, args.interval)


if __name__ == "__main__":
    main()

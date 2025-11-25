"""
Mock predictor for testing API without loading actual models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

class MockPredictor:
    """Mock predictor that generates random predictions."""
    
    def __init__(self, satellite_type):
        self.satellite_type = satellite_type
        print(f"✓ Mock predictor initialized for {satellite_type}")
    
    def run_once(self):
        """Generate mock predictions."""
        current_time = datetime.now()
        predictions = []
        
        for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
            pred_time = current_time + timedelta(minutes=horizon_min)
            
            # Generate random but realistic predictions
            base_error = np.random.uniform(0.5, 3.0)
            
            predictions.append({
                'timestamp_current': current_time.isoformat(),
                'timestamp_predicted': pred_time.isoformat(),
                'horizon_label': horizon_label,
                'horizon_minutes': horizon_min,
                'x_error_pred': base_error * np.random.uniform(0.8, 1.2),
                'y_error_pred': base_error * np.random.uniform(0.8, 1.2),
                'z_error_pred': base_error * np.random.uniform(0.8, 1.2),
                'satclockerror_pred': base_error * np.random.uniform(0.5, 1.5)
            })
        
        return pd.DataFrame(predictions)

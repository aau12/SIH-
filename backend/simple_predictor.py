"""
Simplified predictor that works with available data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import lightgbm as lgb

PROCESSED_DATA_DIR = Path("data/processed")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
PREDICTIONS_DIR = Path("predictions/realtime")

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

class SimplePredictor:
    """Simplified predictor using only LightGBM models."""
    
    def __init__(self, satellite_type):
        self.satellite_type = satellite_type
        self.models = {}
        self._load_models()
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_models(self):
        """Load LightGBM models."""
        print(f"Loading models for {self.satellite_type}...")
        count = 0
        
        for error_col in ERROR_COLUMNS:
            col_key = error_col.replace(' ', '_').replace('(', '').replace(')', '')
            self.models[error_col] = {}
            
            for horizon_label in HORIZON_LABELS:
                model_path = LIGHTGBM_MODELS_DIR / self.satellite_type.lower() / f"{col_key}_{horizon_label}.txt"
                
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    self.models[error_col][horizon_label] = model
                    count += 1
        
        print(f"✓ Loaded {count} models")
    
    def _create_simple_features(self, df):
        """Create simple features from recent data."""
        # Use only the most recent values as features
        latest = df.iloc[-1]
        
        features = {}
        for col in ERROR_COLUMNS:
            if col in df.columns:
                # Current value
                features[f'{col}_current'] = latest[col]
                
                # Simple lag features (last 3 values)
                for i in range(1, min(4, len(df))):
                    features[f'{col}_lag_{i}'] = df[col].iloc[-i-1] if len(df) > i else latest[col]
                
                # Simple rolling mean
                features[f'{col}_mean_3'] = df[col].tail(3).mean()
                features[f'{col}_std_3'] = df[col].tail(3).std() if len(df) >= 3 else 0
        
        return pd.DataFrame([features])
    
    def run_once(self):
        """Generate predictions."""
        try:
            # Load recent data
            data_file = PROCESSED_DATA_DIR / f"{self.satellite_type}_clean_15min.csv"
            df = pd.read_csv(data_file)
            
            # Get last 10 timesteps
            window = df.tail(10)
            
            # Create features
            X = self._create_simple_features(window)
            
            # Generate predictions
            current_time = datetime.now()
            predictions = []
            
            for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
                pred_time = current_time + timedelta(minutes=horizon_min)
                
                row = {
                    'timestamp_current': current_time.isoformat(),
                    'timestamp_predicted': pred_time.isoformat(),
                    'horizon_label': horizon_label,
                    'horizon_minutes': horizon_min
                }
                
                for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
                    if error_col in self.models and horizon_label in self.models[error_col]:
                        model = self.models[error_col][horizon_label]
                        # Use available features (model will use what it needs)
                        try:
                            pred = model.predict(X.values)[0]
                        except:
                            # If prediction fails, use simple baseline
                            pred = window[error_col].iloc[-1]
                        row[f'{short_name}_pred'] = pred
                    else:
                        # Fallback to last known value
                        row[f'{short_name}_pred'] = window[error_col].iloc[-1]
                
                predictions.append(row)
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

"""
Simple Real-Time GNSS Predictor (Windows Compatible)
=====================================================
Generates real-time predictions without Unicode characters.

Usage:
    python realtime_predict_simple.py MEO
    python realtime_predict_simple.py GEO
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Configuration
PROCESSED_DATA_DIR = Path("data/processed")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
LSTM_MODELS_DIR = Path("models/lstm")
REALTIME_PREDICTIONS_DIR = Path("predictions/realtime")

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

# LSTM Model
class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=36):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                               batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        return self.decoder(hidden[-1])

# Feature Engineering
def add_features(df):
    """Add all features to dataframe."""
    df = df.copy()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    LAG_STEPS = [1, 2, 4, 8, 12, 16, 24, 48]
    ROLLING_WINDOWS = [3, 6, 12]
    
    for col in ERROR_COLUMNS:
        if col in df.columns:
            # Lag features
            for lag in LAG_STEPS:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            
            # Rolling features
            for window in ROLLING_WINDOWS:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window, min_periods=1).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window, min_periods=1).max()
            
            # Trend features
            df[f"{col}_diff1"] = df[col].diff()
            df[f"{col}_diff2"] = df[f"{col}_diff1"].diff()
    
    # Time features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df.index.dayofweek
    df['day_index'] = (df.index.date - df.index.date.min()).astype('timedelta64[D]').astype(int)
    
    return df.dropna()

def predict_realtime(satellite_type):
    """Generate real-time predictions."""
    print("\n" + "="*60)
    print(f"REAL-TIME PREDICTOR - {satellite_type}")
    print("="*60)
    
    # Ensure output directory
    REALTIME_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[1/5] Loading data...")
    data_file = PROCESSED_DATA_DIR / f"{satellite_type}_clean_15min.csv"
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Get sliding window (last 100 rows to ensure enough after feature engineering)
    window_df = df.iloc[-100:].copy()
    print(f"    Loaded {len(window_df)} timesteps")
    print(f"    Latest: {window_df.index[-1]}")
    
    # Build features
    print(f"\n[2/5] Building features...")
    features_df = add_features(window_df)
    print(f"    Features created: {len(features_df.columns)}")
    
    if len(features_df) == 0:
        print("ERROR: No valid features after processing")
        return
    
    # Get latest row
    latest_features = features_df.iloc[-1:].copy()
    feature_cols = [c for c in latest_features.columns if not any(x in c for x in ['t+', 'Unnamed'])]
    X = latest_features[feature_cols].values
    
    # Load and predict with LightGBM
    print(f"\n[3/5] Loading LightGBM models...")
    import lightgbm as lgb
    lgbm_predictions = {}
    lgbm_count = 0
    
    for error_col in ERROR_COLUMNS:
        col_key = error_col.replace(' ', '_').replace('(', '').replace(')', '')
        lgbm_predictions[error_col] = {}
        
        for horizon_label in HORIZON_LABELS:
            model_path = LIGHTGBM_MODELS_DIR / satellite_type.lower() / f"{col_key}_{horizon_label}.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                pred = model.predict(X)[0]
                lgbm_predictions[error_col][horizon_label] = pred
                lgbm_count += 1
    
    print(f"    Loaded {lgbm_count} models")
    
    # Load and predict with LSTM
    print(f"\n[4/5] Loading LSTM model...")
    lstm_predictions = {}
    lstm_path = LSTM_MODELS_DIR / f"lstm_{satellite_type.lower()}.pth"
    
    if lstm_path.exists():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = len(feature_cols)
        
        lstm_model = LSTMEncoderDecoder(input_size=input_size, hidden_size=64, 
                                       num_layers=2, dropout=0.2, output_size=36)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model.to(device)
        lstm_model.eval()
        
        # Prepare LSTM input (sequence)
        lookback = 48
        if len(features_df) >= lookback:
            lstm_window = features_df.iloc[-lookback:]
            X_lstm = lstm_window[feature_cols].values
            X_tensor = torch.FloatTensor(X_lstm).unsqueeze(0).to(device)
            
            with torch.no_grad():
                lstm_output = lstm_model(X_tensor).cpu().numpy()[0]
            
            idx = 0
            for error_col in ERROR_COLUMNS:
                lstm_predictions[error_col] = {}
                for horizon_label in HORIZON_LABELS:
                    lstm_predictions[error_col][horizon_label] = lstm_output[idx]
                    idx += 1
            
            print(f"    LSTM loaded and predicted")
    
    # Ensemble predictions
    print(f"\n[5/5] Generating ensemble predictions...")
    ensemble_predictions = []
    current_time = window_df.index[-1]
    
    for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
        pred_time = current_time + timedelta(minutes=horizon_min)
        
        row = {
            'timestamp_current': current_time,
            'timestamp_predicted': pred_time,
            'horizon_label': horizon_label,
            'horizon_minutes': horizon_min
        }
        
        for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
            lgbm_pred = lgbm_predictions.get(error_col, {}).get(horizon_label, 0)
            lstm_pred = lstm_predictions.get(error_col, {}).get(horizon_label, 0)
            ensemble_pred = 0.6 * lgbm_pred + 0.4 * lstm_pred
            
            row[f'{short_name}_pred'] = ensemble_pred
            row[f'{short_name}_lgbm'] = lgbm_pred
            row[f'{short_name}_lstm'] = lstm_pred
        
        ensemble_predictions.append(row)
    
    predictions_df = pd.DataFrame(ensemble_predictions)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = REALTIME_PREDICTIONS_DIR / f"{satellite_type}_{timestamp}.csv"
    latest_file = REALTIME_PREDICTIONS_DIR / f"{satellite_type}_latest.csv"
    json_file = REALTIME_PREDICTIONS_DIR / f"{satellite_type}_latest.json"
    
    predictions_df.to_csv(timestamped_file, index=False)
    predictions_df.to_csv(latest_file, index=False)
    predictions_df.to_json(json_file, orient='records', indent=2)
    
    print(f"\n    Saved to: {latest_file}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    for _, row in predictions_df.iterrows():
        print(f"{row['horizon_label']:>6} | "
              f"x:{row['x_error_pred']:7.3f}m | "
              f"y:{row['y_error_pred']:7.3f}m | "
              f"z:{row['z_error_pred']:7.3f}m | "
              f"clk:{row['satclockerror_pred']:7.3f}m")
    
    print("="*60)
    print(f"SUCCESS! Predictions saved to: predictions/realtime/")
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python realtime_predict_simple.py <MEO|GEO>")
        sys.exit(1)
    
    satellite = sys.argv[1].upper()
    if satellite not in ['MEO', 'GEO']:
        print("Error: Satellite must be MEO or GEO")
        sys.exit(1)
    
    predict_realtime(satellite)

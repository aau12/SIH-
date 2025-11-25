"""
Real-Time GNSS Forecasting Service
===================================
Continuously monitors for new GNSS data and generates predictions every 15 minutes.

Usage:
    python realtime_predictor.py --satellite MEO --mode loop
    python realtime_predictor.py --satellite GEO --mode api
"""

import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_DATA_DIR = Path("data/features")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
LSTM_MODELS_DIR = Path("models/lstm")
REALTIME_PREDICTIONS_DIR = Path("predictions/realtime")

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

LOOKBACK_WINDOW = 48  # Number of historical steps needed
UPDATE_INTERVAL = 900  # 15 minutes in seconds

# ============================================================================
# LSTM MODEL DEFINITION (same as training)
# ============================================================================

class LSTMEncoderDecoder(nn.Module):
    """LSTM Encoder-Decoder for multi-horizon forecasting."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=36):
        super(LSTMEncoderDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        context = hidden[-1]
        output = self.decoder(context)
        return output


# ============================================================================
# FEATURE ENGINEERING (from feature_engineering.py)
# ============================================================================

def add_lag_features(df, column_name, lags):
    """Add lag features for a column."""
    df_new = df.copy()
    for lag in lags:
        feature_name = f"{column_name}_lag_{lag}"
        df_new[feature_name] = df_new[column_name].shift(lag)
    return df_new


def add_rolling_features(df, column_name, windows):
    """Add rolling window statistics."""
    df_new = df.copy()
    for window in windows:
        df_new[f"{column_name}_rolling_mean_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).mean()
        df_new[f"{column_name}_rolling_std_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).std()
        df_new[f"{column_name}_rolling_min_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).min()
        df_new[f"{column_name}_rolling_max_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).max()
    return df_new


def add_trend_features(df, column_name):
    """Add trend features (derivatives)."""
    df_new = df.copy()
    df_new[f"{column_name}_diff1"] = df_new[column_name].diff()
    df_new[f"{column_name}_diff2"] = df_new[f"{column_name}_diff1"].diff()
    return df_new


def add_time_features(df):
    """Add time-based features."""
    df_new = df.copy()
    df_new['hour'] = df_new.index.hour
    df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
    df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
    df_new['day_of_week'] = df_new.index.dayofweek
    df_new['day_index'] = (df_new.index.date - df_new.index.date.min()).astype('timedelta64[D]').astype(int)
    return df_new


def build_features_from_window(window_df):
    """Build all features from a sliding window of data."""
    df = window_df.copy()
    
    # Normalize column names
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Add features for each error column
    LAG_STEPS = [1, 2, 4, 8, 12, 16, 24, 48]
    ROLLING_WINDOWS = [3, 6, 12]
    
    for col in ERROR_COLUMNS:
        if col in df.columns:
            df = add_lag_features(df, col, LAG_STEPS)
            df = add_rolling_features(df, col, ROLLING_WINDOWS)
            df = add_trend_features(df, col)
    
    # Add time features
    df = add_time_features(df)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


# ============================================================================
# MODEL LOADING
# ============================================================================

class RealtimePredictor:
    """Real-time GNSS forecasting predictor."""
    
    def __init__(self, satellite_type):
        """
        Initialize predictor with trained models.
        
        Args:
            satellite_type: 'MEO' or 'GEO'
        """
        self.satellite_type = satellite_type
        self.lightgbm_models = {}
        self.lstm_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING REAL-TIME PREDICTOR - {satellite_type}")
        print(f"{'='*60}")
        
        self._load_models()
        
        # Ensure output directory exists
        REALTIME_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        
    def _load_models(self):
        """Load all trained models."""
        print(f"\n-> Loading models...")
        
        # Load LightGBM models
        lgbm_count = 0
        for error_col in ERROR_COLUMNS:
            col_key = error_col.replace(' ', '_').replace('(', '').replace(')', '')
            self.lightgbm_models[error_col] = {}
            
            for horizon_label in HORIZON_LABELS:
                model_path = LIGHTGBM_MODELS_DIR / self.satellite_type.lower() / f"{col_key}_{horizon_label}.txt"
                
                if model_path.exists():
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=str(model_path))
                    self.lightgbm_models[error_col][horizon_label] = model
                    lgbm_count += 1
        
        print(f"  ✓ Loaded {lgbm_count} LightGBM models")
        
        # Load LSTM model
        lstm_path = LSTM_MODELS_DIR / f"lstm_{self.satellite_type.lower()}.pth"
        
        if lstm_path.exists():
            # Determine input size from a sample feature file
            sample_features = FEATURES_DATA_DIR / f"{self.satellite_type}_features.csv"
            if sample_features.exists():
                sample_df = pd.read_csv(sample_features, nrows=1)
                feature_cols = [c for c in sample_df.columns 
                               if not any(x in c for x in ['t+', 'Unnamed'])]
                input_size = len(feature_cols)
                
                self.lstm_model = LSTMEncoderDecoder(
                    input_size=input_size,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    output_size=36
                )
                
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                self.lstm_model.to(self.device)
                self.lstm_model.eval()
                
                print(f"  ✓ Loaded LSTM model")
        
        print(f"  ✓ All models loaded successfully")
    
    def get_latest_window(self):
        """
        Get the latest sliding window of data.
        
        Returns:
            DataFrame with last LOOKBACK_WINDOW timesteps
        """
        # Read the cleaned data file
        data_file = PROCESSED_DATA_DIR / f"{self.satellite_type}_clean_15min.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Get last N rows
        window_df = df.iloc[-LOOKBACK_WINDOW:].copy()
        
        return window_df
    
    def predict_from_window(self, window_df):
        """
        Generate predictions from a sliding window.
        
        Args:
            window_df: DataFrame with historical data
            
        Returns:
            DataFrame with predictions for all horizons
        """
        # Build features
        features_df = build_features_from_window(window_df)
        
        if len(features_df) == 0:
            raise ValueError("No valid features after processing window")
        
        # Get the latest feature row
        latest_features = features_df.iloc[-1:].copy()
        
        # Get feature columns (exclude targets and index)
        feature_cols = [c for c in latest_features.columns 
                       if not any(x in c for x in ['t+', 'Unnamed'])]
        
        X = latest_features[feature_cols].values
        
        # Predict with LightGBM
        lgbm_predictions = {}
        
        for error_col in ERROR_COLUMNS:
            lgbm_predictions[error_col] = {}
            
            for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
                if error_col in self.lightgbm_models and horizon_label in self.lightgbm_models[error_col]:
                    model = self.lightgbm_models[error_col][horizon_label]
                    pred = model.predict(X)[0]
                    lgbm_predictions[error_col][horizon_label] = pred
        
        # Predict with LSTM
        lstm_predictions = {}
        
        if self.lstm_model is not None:
            # Prepare LSTM input (sequence of last LOOKBACK_WINDOW steps)
            lstm_features_df = build_features_from_window(window_df)
            
            if len(lstm_features_df) >= LOOKBACK_WINDOW:
                lstm_window = lstm_features_df.iloc[-LOOKBACK_WINDOW:]
                X_lstm = lstm_window[feature_cols].values
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_lstm).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    lstm_output = self.lstm_model(X_tensor).cpu().numpy()[0]
                
                # Parse LSTM output (36 values = 9 horizons × 4 error columns)
                idx = 0
                for error_col in ERROR_COLUMNS:
                    lstm_predictions[error_col] = {}
                    for horizon_label in HORIZON_LABELS:
                        lstm_predictions[error_col][horizon_label] = lstm_output[idx]
                        idx += 1
        
        # Ensemble predictions (60% LightGBM, 40% LSTM)
        ensemble_predictions = []
        
        current_time = window_df.index[-1]
        
        for i, (horizon_label, horizon_min) in enumerate(zip(HORIZON_LABELS, HORIZON_MINUTES)):
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
                
                # Ensemble
                ensemble_pred = 0.6 * lgbm_pred + 0.4 * lstm_pred
                
                row[f'{short_name}_pred'] = ensemble_pred
                row[f'{short_name}_lgbm'] = lgbm_pred
                row[f'{short_name}_lstm'] = lstm_pred
            
            ensemble_predictions.append(row)
        
        return pd.DataFrame(ensemble_predictions)
    
    def save_predictions(self, predictions_df):
        """Save predictions to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save with timestamp
        timestamped_file = REALTIME_PREDICTIONS_DIR / f"{self.satellite_type}_{timestamp}.csv"
        predictions_df.to_csv(timestamped_file, index=False)
        
        # Save as "latest" for dashboard
        latest_file = REALTIME_PREDICTIONS_DIR / f"{self.satellite_type}_latest.csv"
        predictions_df.to_csv(latest_file, index=False)
        
        # Save JSON version
        json_file = REALTIME_PREDICTIONS_DIR / f"{self.satellite_type}_latest.json"
        predictions_df.to_json(json_file, orient='records', indent=2)
        
        print(f"  ✓ Saved predictions to: {latest_file}")
        
        return latest_file
    
    def run_once(self):
        """Run prediction once."""
        print(f"\n{'='*60}")
        print(f"GENERATING PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        try:
            # Get latest window
            window_df = self.get_latest_window()
            print(f"  ✓ Retrieved window: {len(window_df)} timesteps")
            print(f"  ✓ Latest timestamp: {window_df.index[-1]}")
            
            # Generate predictions
            predictions_df = self.predict_from_window(window_df)
            print(f"  ✓ Generated {len(predictions_df)} horizon predictions")
            
            # Save predictions
            output_file = self.save_predictions(predictions_df)
            
            # Print summary
            print(f"\n→ Prediction Summary:")
            for _, row in predictions_df.iterrows():
                print(f"  {row['horizon_label']:>6} | "
                      f"x:{row['x_error_pred']:7.3f}m | "
                      f"y:{row['y_error_pred']:7.3f}m | "
                      f"z:{row['z_error_pred']:7.3f}m | "
                      f"clk:{row['satclockerror_pred']:7.3f}m")
            
            return predictions_df
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_loop(self, interval=UPDATE_INTERVAL):
        """Run continuous prediction loop."""
        print(f"\n{'='*60}")
        print(f"STARTING REAL-TIME PREDICTION LOOP")
        print(f"Update interval: {interval} seconds ({interval/60:.1f} minutes)")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}")
        
        try:
            while True:
                self.run_once()
                
                print(f"\n→ Waiting {interval/60:.1f} minutes until next update...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("Real-time prediction loop stopped by user")
            print(f"{'='*60}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Real-time GNSS Forecasting')
    parser.add_argument('--satellite', type=str, choices=['MEO', 'GEO'], required=True,
                       help='Satellite type')
    parser.add_argument('--mode', type=str, choices=['once', 'loop'], default='once',
                       help='Run mode: once or continuous loop')
    parser.add_argument('--interval', type=int, default=900,
                       help='Update interval in seconds (default: 900 = 15 min)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealtimePredictor(args.satellite)
    
    # Run based on mode
    if args.mode == 'once':
        predictor.run_once()
    else:
        predictor.run_loop(interval=args.interval)


if __name__ == "__main__":
    main()

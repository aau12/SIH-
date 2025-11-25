"""
Day 8 Multi-Horizon GNSS Prediction
====================================
This script generates full multi-horizon predictions for the 8th day using
trained LightGBM and LSTM models. Supports ensemble predictions and generates
comprehensive forecast outputs.

Forecast Horizons: 15min, 30min, 45min, 1h, 2h, 3h, 6h, 12h, 24h
"""

import os
import json
import warnings
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURES_DATA_DIR = Path("data/features")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
LSTM_MODELS_DIR = Path("models/lstm")
PREDICTIONS_DIR = Path("predictions")
PLOTS_DIR = Path("predictions/plots")

# Error columns
ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]

# Forecast horizons (in 15-minute steps)
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

# Lookback window for predictions
LOOKBACK = 48  # Last 12 hours (48 * 15 minutes)

# Ensemble weights
ENSEMBLE_WEIGHT_LGBM = 0.6
ENSEMBLE_WEIGHT_LSTM = 0.4


# ============================================================================
# LSTM MODEL ARCHITECTURE (must match training)
# ============================================================================

class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Encoder-Decoder for multi-horizon forecasting.
    Must match the architecture used in train_models.py
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=36):
        super(LSTMEncoderDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder (fully connected)
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
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # Use last hidden state
        context = hidden[-1]
        # Decode to multi-horizon predictions
        output = self.decoder(context)
        return output


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create necessary directories for predictions and plots.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured: {PREDICTIONS_DIR}, {PLOTS_DIR}")


def load_feature_data(file_path):
    """
    Load feature-engineered dataset.
    
    Args:
        file_path: Path to feature CSV file
        
    Returns:
        pandas DataFrame with datetime index
    """
    try:
        print(f"\n→ Loading feature data from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Normalize column names
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        print(f"  ✓ Loaded {len(df)} rows")
        print(f"  ✓ Columns: {len(df.columns)}")
        print(f"  ✓ Time range: {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        raise


def get_feature_columns(df):
    """
    Extract feature column names (exclude target columns with t+N pattern).
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of feature column names
    """
    # Features are all columns that don't end with t+N pattern
    target_suffixes = tuple(f't+{h}' for h in FORECAST_HORIZONS)
    feature_cols = [col for col in df.columns if not col.endswith(target_suffixes)]
    return feature_cols


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_lightgbm_models(satellite_type):
    """
    Load all LightGBM models for a satellite type.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Dictionary of models: {error_col: {horizon_label: model}}
    """
    print(f"\n→ Loading LightGBM models for {satellite_type}...")
    
    models = {}
    model_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
    
    if not model_dir.exists():
        print(f"  ⚠ Model directory not found: {model_dir}")
        return models
    
    loaded_count = 0
    
    for error_col in ERROR_COLUMNS:
        models[error_col] = {}
        
        for horizon, horizon_label in zip(FORECAST_HORIZONS, HORIZON_LABELS):
            # Construct model filename
            model_name = f"{error_col.replace(' ', '_').replace('(', '').replace(')', '')}_{horizon_label}.txt"
            model_path = model_dir / model_name
            
            if model_path.exists():
                try:
                    model = lgb.Booster(model_file=str(model_path))
                    models[error_col][horizon_label] = model
                    loaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to load {model_name}: {e}")
            else:
                print(f"  ⚠ Model not found: {model_name}")
    
    print(f"  ✓ Loaded {loaded_count} LightGBM models")
    return models


def load_lstm_model(satellite_type):
    """
    Load LSTM model for a satellite type.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Loaded LSTM model, or None if not found
    """
    print(f"\n→ Loading LSTM model for {satellite_type}...")
    
    model_path = LSTM_MODELS_DIR / f"{satellite_type.lower()}_model.pth"
    
    if not model_path.exists():
        print(f"  ⚠ LSTM model not found: {model_path}")
        return None
    
    try:
        # Load checkpoint (allow unsafe globals for numpy compatibility)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get model configuration
        config = checkpoint['model_config']
        
        # Initialize model
        model = LSTMEncoderDecoder(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  ✓ LSTM model loaded successfully")
        print(f"  ✓ Input size: {config['input_size']}, Output size: {config['output_size']}")
        
        return model
        
    except Exception as e:
        print(f"  ✗ Error loading LSTM model: {e}")
        return None


# ============================================================================
# INPUT PREPARATION
# ============================================================================

def prepare_input_window(df, lookback=48):
    """
    Extract the last lookback window for prediction.
    
    Args:
        df: pandas DataFrame with cleaned data
        lookback: Number of timesteps to use
        
    Returns:
        DataFrame window, last timestamp
    """
    if len(df) < lookback:
        print(f"  ⚠ Warning: Dataset has only {len(df)} rows, less than lookback {lookback}")
        window = df
    else:
        window = df.iloc[-lookback:]
    
    last_timestamp = window.index[-1]
    
    print(f"\n→ Input window prepared:")
    print(f"  • Window size: {len(window)} timesteps")
    print(f"  • Time range: {window.index[0]} to {last_timestamp}")
    print(f"  • Last timestamp: {last_timestamp}")
    
    return window, last_timestamp


def prepare_lightgbm_input(window):
    """
    Prepare input features for LightGBM prediction.
    Uses the last timestep with all engineered features.
    
    Args:
        window: DataFrame window
        
    Returns:
        Feature array for prediction
    """
    feature_cols = get_feature_columns(window)
    
    # Use last row as features (includes lag, rolling, trend, time features)
    X = window[feature_cols].iloc[-1:].values
    
    print(f"\n→ LightGBM input prepared:")
    print(f"  • Features shape: {X.shape}")
    print(f"  • Feature columns: {len(feature_cols)}")
    
    return X


def prepare_lstm_input(window):
    """
    Prepare input sequence for LSTM prediction.
    
    Args:
        window: DataFrame window
        
    Returns:
        Tensor of shape (1, lookback, num_features)
    """
    feature_cols = get_feature_columns(window)
    
    # Get sequence (includes all engineered features)
    X = window[feature_cols].values
    
    # Convert to tensor and add batch dimension
    X_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, lookback, features)
    
    print(f"\n→ LSTM input prepared:")
    print(f"  • Input shape: {X_tensor.shape}")
    
    return X_tensor


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_lightgbm(models, X, satellite_type):
    """
    Generate predictions using LightGBM models.
    
    Args:
        models: Dictionary of LightGBM models
        X: Input features
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Dictionary of predictions: {horizon_label: {error_col: value}}
    """
    print(f"\n→ Generating LightGBM predictions for {satellite_type}...")
    
    predictions = {}
    
    for horizon_label in HORIZON_LABELS:
        predictions[horizon_label] = {}
        
        for error_col in ERROR_COLUMNS:
            if error_col in models and horizon_label in models[error_col]:
                model = models[error_col][horizon_label]
                pred = model.predict(X)[0]
                predictions[horizon_label][error_col] = pred
            else:
                predictions[horizon_label][error_col] = np.nan
    
    print(f"  ✓ Generated predictions for {len(HORIZON_LABELS)} horizons")
    
    return predictions


def predict_lstm(model, X_tensor, satellite_type):
    """
    Generate predictions using LSTM model.
    
    Args:
        model: LSTM model
        X_tensor: Input tensor
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Dictionary of predictions: {horizon_label: {error_col: value}}
    """
    print(f"\n→ Generating LSTM predictions for {satellite_type}...")
    
    if model is None:
        print(f"  ⚠ No LSTM model available")
        return None
    
    try:
        with torch.no_grad():
            output = model(X_tensor)  # (1, output_size)
        
        # Convert to numpy
        predictions_array = output.squeeze(0).numpy()  # (output_size,)
        
        # Reshape to (num_horizons, num_error_cols)
        predictions_array = predictions_array.reshape(len(FORECAST_HORIZONS), len(ERROR_COLUMNS))
        
        # Organize into dictionary
        predictions = {}
        for i, horizon_label in enumerate(HORIZON_LABELS):
            predictions[horizon_label] = {}
            for j, error_col in enumerate(ERROR_COLUMNS):
                predictions[horizon_label][error_col] = predictions_array[i, j]
        
        print(f"  ✓ Generated predictions for {len(HORIZON_LABELS)} horizons")
        
        return predictions
        
    except Exception as e:
        print(f"  ✗ Error during LSTM prediction: {e}")
        return None


# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================

def ensemble_predictions(lgbm_preds, lstm_preds, weight_lgbm=0.6, weight_lstm=0.4):
    """
    Ensemble LightGBM and LSTM predictions.
    
    Args:
        lgbm_preds: LightGBM predictions dictionary
        lstm_preds: LSTM predictions dictionary
        weight_lgbm: Weight for LightGBM (default: 0.6)
        weight_lstm: Weight for LSTM (default: 0.4)
        
    Returns:
        Ensembled predictions dictionary
    """
    print(f"\n→ Ensembling predictions (LightGBM: {weight_lgbm}, LSTM: {weight_lstm})...")
    
    if lstm_preds is None:
        print(f"  ⚠ LSTM predictions not available, using LightGBM only")
        return lgbm_preds
    
    ensemble_preds = {}
    
    for horizon_label in HORIZON_LABELS:
        ensemble_preds[horizon_label] = {}
        
        for error_col in ERROR_COLUMNS:
            lgbm_val = lgbm_preds[horizon_label].get(error_col, np.nan)
            lstm_val = lstm_preds[horizon_label].get(error_col, np.nan)
            
            # Ensemble if both available
            if not np.isnan(lgbm_val) and not np.isnan(lstm_val):
                ensemble_val = weight_lgbm * lgbm_val + weight_lstm * lstm_val
            elif not np.isnan(lgbm_val):
                ensemble_val = lgbm_val
            elif not np.isnan(lstm_val):
                ensemble_val = lstm_val
            else:
                ensemble_val = np.nan
            
            ensemble_preds[horizon_label][error_col] = ensemble_val
    
    print(f"  ✓ Ensemble predictions generated")
    
    return ensemble_preds


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def build_prediction_table(predictions, last_timestamp, satellite_type):
    """
    Build final prediction DataFrame.
    
    Args:
        predictions: Predictions dictionary
        last_timestamp: Last known timestamp
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        pandas DataFrame with predictions
    """
    print(f"\n→ Building prediction table for {satellite_type}...")
    
    rows = []
    
    for horizon_label, horizon_minutes in zip(HORIZON_LABELS, HORIZON_MINUTES):
        # Calculate prediction timestamp
        pred_timestamp = last_timestamp + timedelta(minutes=horizon_minutes)
        
        row = {
            'horizon_label': horizon_label,
            'horizon_minutes': horizon_minutes,
            'timestamp_predicted': pred_timestamp
        }
        
        # Add predictions for each error column
        for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
            pred_value = predictions[horizon_label].get(error_col, np.nan)
            row[f'{short_name}_pred'] = pred_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"  ✓ Prediction table created with {len(df)} rows")
    
    return df


def save_predictions(df, satellite_type):
    """
    Save predictions to CSV and JSON.
    
    Args:
        df: Prediction DataFrame
        satellite_type: 'MEO' or 'GEO'
    """
    # Save CSV
    csv_path = PREDICTIONS_DIR / f"{satellite_type}_Day8_Predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Predictions saved to: {csv_path}")
    
    # Save JSON summary
    json_data = {
        'satellite_type': satellite_type,
        'prediction_count': len(df),
        'horizons': HORIZON_LABELS,
        'predictions': df.to_dict(orient='records')
    }
    
    json_path = PREDICTIONS_DIR / f"{satellite_type}_Day8_Predictions.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"✓ JSON summary saved to: {json_path}")


def plot_predictions(df, satellite_type):
    """
    Plot prediction forecasts.
    
    Args:
        df: Prediction DataFrame
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n→ Generating prediction plots for {satellite_type}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{satellite_type} Day 8 Multi-Horizon Predictions', 
                 fontsize=16, fontweight='bold')
    
    error_cols_plot = ['x_error_pred', 'y_error_pred', 'z_error_pred', 'satclockerror_pred']
    titles = ['X Error', 'Y Error', 'Z Error', 'Satellite Clock Error']
    
    for ax, col, title in zip(axes.flat, error_cols_plot, titles):
        ax.plot(df['horizon_minutes'], df[col], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Forecast Horizon (minutes)', fontsize=11)
        ax.set_ylabel('Predicted Error (m)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"{satellite_type}_Day8_Predictions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plot saved to: {plot_path}")


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================

def predict_for_satellite(satellite_type):
    """
    Complete prediction pipeline for one satellite type.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  DAY 8 PREDICTION PIPELINE FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    try:
        # 1. Load feature-engineered data
        data_file = FEATURES_DATA_DIR / f"{satellite_type}_features.csv"
        df = load_feature_data(data_file)
        
        # 2. Prepare input window
        window, last_timestamp = prepare_input_window(df, LOOKBACK)
        
        # 3. Load models
        lgbm_models = load_lightgbm_models(satellite_type)
        lstm_model = load_lstm_model(satellite_type)
        
        # 4. Prepare inputs
        X_lgbm = prepare_lightgbm_input(window)
        X_lstm = prepare_lstm_input(window)
        
        # 5. Generate predictions
        lgbm_preds = predict_lightgbm(lgbm_models, X_lgbm, satellite_type)
        lstm_preds = predict_lstm(lstm_model, X_lstm, satellite_type)
        
        # 6. Ensemble predictions
        final_preds = ensemble_predictions(
            lgbm_preds, 
            lstm_preds, 
            ENSEMBLE_WEIGHT_LGBM, 
            ENSEMBLE_WEIGHT_LSTM
        )
        
        # 7. Build prediction table
        pred_df = build_prediction_table(final_preds, last_timestamp, satellite_type)
        
        # 8. Save outputs
        save_predictions(pred_df, satellite_type)
        plot_predictions(pred_df, satellite_type)
        
        print(f"\n{'='*70}")
        print(f"✓ {satellite_type} PREDICTION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
        return pred_df
        
    except Exception as e:
        print(f"\n✗ ERROR during {satellite_type} prediction: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_final_summary():
    """
    Print final summary of predictions.
    """
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print(f"{'='*70}")
    
    # Check output files
    meo_csv = PREDICTIONS_DIR / "MEO_Day8_Predictions.csv"
    geo_csv = PREDICTIONS_DIR / "GEO_Day8_Predictions.csv"
    
    print(f"\n📊 PREDICTIONS GENERATED:")
    if meo_csv.exists():
        meo_df = pd.read_csv(meo_csv)
        print(f"  • MEO: {len(meo_df)} horizons")
    if geo_csv.exists():
        geo_df = pd.read_csv(geo_csv)
        print(f"  • GEO: {len(geo_df)} horizons")
    
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"  • Predictions: {PREDICTIONS_DIR}")
    print(f"  • Plots: {PLOTS_DIR}")
    
    print(f"\n🔮 FORECAST HORIZONS:")
    for label, minutes in zip(HORIZON_LABELS, HORIZON_MINUTES):
        print(f"  • {label:6s} → {minutes:4d} minutes ahead")
    
    print(f"\n{'='*70}")
    print("✓ ALL PREDICTIONS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print(" "*15 + "GNSS DAY 8 MULTI-HORIZON PREDICTION")
    print(" "*25 + "FORECAST PIPELINE")
    print("="*70)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Predict for MEO
        meo_predictions = predict_for_satellite("MEO")
        
        # Predict for GEO
        geo_predictions = predict_for_satellite("GEO")
        
        # Print final summary
        print_final_summary()
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

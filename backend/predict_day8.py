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
LOOKBACK = 56  # Last 14 hours (56 * 15 minutes) - matches v2 training

# Ensemble weights (V2: includes correction network)
ENSEMBLE_WEIGHT_LSTM = 0.5
ENSEMBLE_WEIGHT_LGBM = 0.3
ENSEMBLE_WEIGHT_CORRECTION = 0.2

# Correction models directory
CORRECTION_MODELS_DIR = Path("models/correction")


# ============================================================================
# LSTM MODEL ARCHITECTURE (must match training)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for long-term dependencies."""
    
    def __init__(self, hidden_size, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.fc_out(out)
        
        return out


class CNN_LSTM_Attention(nn.Module):
    """
    Enhanced CNN-LSTM-Attention architecture (V2).
    Supports both old and new model formats.
    """
    
    def __init__(self, input_size, hidden_size=192, num_layers=3, dropout=0.3,
                 output_size=36, cnn_filters=64, cnn_kernel=3, attention_heads=4):
        super(CNN_LSTM_Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN for short-term feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_filters * 2),
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_size, attention_heads)
        self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Decoder
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
        # CNN expects (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x_cnn)
        
        # Attention
        attn_out = self.attention(lstm_out)
        attn_out = self.attention_norm(lstm_out + attn_out)  # Residual connection
        
        # Use last timestep
        context = attn_out[:, -1, :]
        
        # Decode
        output = self.decoder(context)
        
        return output


# Legacy model for backward compatibility
class LSTMEncoderDecoder(nn.Module):
    """Legacy LSTM Encoder-Decoder for backward compatibility."""
    
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
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create necessary directories for predictions and plots.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories ensured: {PREDICTIONS_DIR}, {PLOTS_DIR}")


def get_features_file(satellite_type):
    """Get the correct feature file (prefer v2 if available)."""
    v2_path = FEATURES_DATA_DIR / f"{satellite_type}_features_v2.csv"
    v1_path = FEATURES_DATA_DIR / f"{satellite_type}_features.csv"
    
    if v2_path.exists():
        return v2_path
    return v1_path


def load_feature_data(file_path):
    """
    Load feature-engineered dataset.
    
    Args:
        file_path: Path to feature CSV file
        
    Returns:
        pandas DataFrame with datetime index
    """
    try:
        print(f"\n-> Loading feature data from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Normalize column names
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        print(f"  [OK] Loaded {len(df)} rows")
        print(f"  [OK] Columns: {len(df.columns)}")
        print(f"  [OK] Time range: {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"  [ERROR] loading {file_path}: {e}")
        raise


def get_feature_columns(df):
    """
    Extract feature column names (exclude target columns with t+N pattern).
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of feature column names
    """
    target_suffixes = tuple(f't+{h}' for h in FORECAST_HORIZONS)
    feature_cols = [col for col in df.columns if not col.endswith(target_suffixes)]
    feature_cols = [col for col in feature_cols if col not in ERROR_COLUMNS]
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
    print(f"\n-> Loading LightGBM models for {satellite_type}...")
    
    models = {}
    model_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
    
    if not model_dir.exists():
        print(f"  [WARN] Model directory not found: {model_dir}")
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
                    print(f"  [WARN] Failed to load {model_name}: {e}")
            else:
                print(f"  [WARN] Model not found: {model_name}")
    
    print(f"  [OK] Loaded {loaded_count} LightGBM models")
    return models


def load_lstm_model(satellite_type):
    """
    Load LSTM model for a satellite type.
    Supports both V2 (CNN-LSTM-Attention) and legacy (LSTMEncoderDecoder) models.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Loaded LSTM model, or None if not found
    """
    print(f"\n-> Loading LSTM model for {satellite_type}...")
    
    model_path = LSTM_MODELS_DIR / f"{satellite_type.lower()}_model.pth"
    
    if not model_path.exists():
        print(f"  [WARN] LSTM model not found: {model_path}")
        return None
    
    try:
        # Load checkpoint (allow unsafe globals for numpy compatibility)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get model configuration
        config = checkpoint['model_config']
        
        # Check if this is a V2 model (has cnn_filters)
        if 'cnn_filters' in config:
            # V2: CNN-LSTM-Attention model
            model = CNN_LSTM_Attention(**config)
            print(f"  [OK] Loaded V2 CNN-LSTM-Attention model")
        else:
            # Legacy: LSTMEncoderDecoder model
            model = LSTMEncoderDecoder(**config)
            print(f"  [OK] Loaded legacy LSTM model")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  [OK] Input size: {config['input_size']}, Output size: {config['output_size']}")
        
        return model
        
    except Exception as e:
        print(f"  [ERROR] loading LSTM model: {e}")
        import traceback
        traceback.print_exc()
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
        print(f"  [WARN] Dataset has only {len(df)} rows, less than lookback {lookback}")
        window = df
    else:
        window = df.iloc[-lookback:]
    
    last_timestamp = window.index[-1]
    
    print(f"\n-> Input window prepared:")
    print(f"  - Window size: {len(window)} timesteps")
    print(f"  - Time range: {window.index[0]} to {last_timestamp}")
    print(f"  - Last timestamp: {last_timestamp}")
    
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
    
    print(f"\n-> LightGBM input prepared:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Feature columns: {len(feature_cols)}")
    
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
    
    print(f"\n-> LSTM input prepared:")
    print(f"  - Input shape: {X_tensor.shape}")
    
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
    print(f"\n-> Generating LightGBM predictions for {satellite_type}...")
    
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
    
    print(f"  [OK] Generated predictions for {len(HORIZON_LABELS)} horizons")
    
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
    print(f"\n-> Generating LSTM predictions for {satellite_type}...")
    
    if model is None:
        print(f"  [WARN] No LSTM model available")
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
        
        print(f"  [OK] Generated predictions for {len(HORIZON_LABELS)} horizons")
        
        return predictions
        
    except Exception as e:
        print(f"  [ERROR] during LSTM prediction: {e}")
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
    print(f"\n-> Ensembling predictions (LightGBM: {weight_lgbm}, LSTM: {weight_lstm})...")
    
    if lstm_preds is None:
        print(f"  [WARN] LSTM predictions not available, using LightGBM only")
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
    
    print(f"  [OK] Ensemble predictions generated")
    
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
    print(f"\n-> Building prediction table for {satellite_type}...")
    
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
    
    print(f"  [OK] Prediction table created with {len(df)} rows")
    
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
    print(f"\n[OK] Predictions saved to: {csv_path}")
    
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
    print(f"[OK] JSON summary saved to: {json_path}")


def plot_predictions(df, satellite_type):
    """
    Plot prediction forecasts.
    
    Args:
        df: Prediction DataFrame
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n-> Generating prediction plots for {satellite_type}...")
    
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
    
    print(f"  [OK] Plot saved to: {plot_path}")


# ============================================================================
# ROLLING MULTI-SAMPLE PREDICTIONS
# ============================================================================

def predict_rolling(satellite_type, num_samples=None, step_size=1):
    """
    Generate rolling predictions from multiple starting points.
    This creates multiple samples per horizon for statistical tests like Shapiro-Wilk.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        num_samples: Number of prediction samples to generate. If None, use all available data.
        step_size: Number of timesteps between each starting point (default: 1 = every 15 min)
        
    Returns:
        DataFrame with multiple predictions per horizon
    """
    # 1. Load feature-engineered data (use v2 features if available)
    data_file = get_features_file(satellite_type)
    df = load_feature_data(data_file)
    
    # 2. Load models once
    lgbm_models = load_lightgbm_models(satellite_type)
    lstm_model = load_lstm_model(satellite_type)
    
    # 3. Calculate number of samples if not specified (use all available data)
    if num_samples is None:
        num_samples = max(1, (len(df) - LOOKBACK) // step_size + 1)
    
    print(f"\n{'='*70}")
    print(f"  ROLLING PREDICTIONS FOR {satellite_type} ({num_samples} samples)")
    print(f"{'='*70}")
    
    # Verify we have enough data
    total_needed = LOOKBACK + (num_samples - 1) * step_size
    
    if len(df) < total_needed:
        print(f"  [WARN] Dataset has {len(df)} rows, adjusting num_samples")
        num_samples = max(1, (len(df) - LOOKBACK) // step_size + 1)
        print(f"  [INFO] Using {num_samples} samples instead")
    
    all_predictions = []
    feature_cols = get_feature_columns(df)
    
    print(f"\n-> Generating {num_samples} prediction sets...")
    
    for i in range(num_samples):
        # Calculate window end position (going backwards from end of data)
        end_idx = len(df) - i * step_size
        start_idx = end_idx - LOOKBACK
        
        if start_idx < 0:
            break
            
        # Extract window
        window = df.iloc[start_idx:end_idx]
        last_timestamp = window.index[-1]
        
        # Prepare inputs
        X_lgbm = window[feature_cols].iloc[-1:].values
        X_lstm = torch.FloatTensor(window[feature_cols].values).unsqueeze(0)
        
        # Generate LightGBM predictions
        lgbm_preds = {}
        for horizon_label in HORIZON_LABELS:
            lgbm_preds[horizon_label] = {}
            for error_col in ERROR_COLUMNS:
                if error_col in lgbm_models and horizon_label in lgbm_models[error_col]:
                    model = lgbm_models[error_col][horizon_label]
                    pred = model.predict(X_lgbm)[0]
                    lgbm_preds[horizon_label][error_col] = pred
        
        # Generate LSTM predictions (if available)
        lstm_preds = None
        if lstm_model is not None:
            with torch.no_grad():
                lstm_output = lstm_model(X_lstm).numpy().flatten()
            lstm_preds = {}
            idx = 0
            for error_col in ERROR_COLUMNS:
                lstm_preds_col = {}
                for horizon_label in HORIZON_LABELS:
                    if idx < len(lstm_output):
                        lstm_preds_col[horizon_label] = lstm_output[idx]
                        idx += 1
                lstm_preds[horizon_label] = lstm_preds.get(horizon_label, {})
                for hl in HORIZON_LABELS:
                    if hl not in lstm_preds:
                        lstm_preds[hl] = {}
                    lstm_preds[hl][error_col] = lstm_preds_col.get(hl, np.nan)
        
        # Ensemble or use LightGBM only
        final_preds = lgbm_preds
        if lstm_preds is not None:
            for horizon_label in HORIZON_LABELS:
                for error_col in ERROR_COLUMNS:
                    lgbm_val = lgbm_preds[horizon_label].get(error_col, np.nan)
                    lstm_val = lstm_preds[horizon_label].get(error_col, np.nan)
                    if not np.isnan(lgbm_val) and not np.isnan(lstm_val):
                        final_preds[horizon_label][error_col] = ENSEMBLE_WEIGHT_LGBM * lgbm_val + ENSEMBLE_WEIGHT_LSTM * lstm_val
        
        # Build prediction rows
        for horizon_label, horizon_minutes in zip(HORIZON_LABELS, HORIZON_MINUTES):
            pred_timestamp = last_timestamp + timedelta(minutes=horizon_minutes)
            
            row = {
                'sample_id': i,
                'horizon_label': horizon_label,
                'horizon_minutes': horizon_minutes,
                'timestamp_current': last_timestamp,
                'timestamp_predicted': pred_timestamp
            }
            
            for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
                row[f'{short_name}_pred'] = final_preds[horizon_label].get(error_col, np.nan)
            
            all_predictions.append(row)
        
        # Print progress every 50 samples or at key points
        if (i + 1) % 50 == 0 or i == 0 or i == num_samples - 1:
            print(f"  - Sample {i+1}/{num_samples}: window ending at {last_timestamp}")
    
    # Create DataFrame
    pred_df = pd.DataFrame(all_predictions)
    
    # Save predictions
    csv_path = PREDICTIONS_DIR / f"{satellite_type}_Day8_Predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Rolling predictions saved to: {csv_path}")
    print(f"[OK] Total predictions: {len(pred_df)} ({num_samples} samples x {len(HORIZON_LABELS)} horizons)")
    
    return pred_df


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
        # 1. Load feature-engineered data (use v2 features if available)
        data_file = get_features_file(satellite_type)
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
        print(f"[OK] {satellite_type} PREDICTION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
        return pred_df
        
    except Exception as e:
        print(f"\n[ERROR] during {satellite_type} prediction: {e}")
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
    
    print(f"\nPREDICTIONS GENERATED:")
    if meo_csv.exists():
        meo_df = pd.read_csv(meo_csv)
        print(f"  - MEO: {len(meo_df)} predictions")
    if geo_csv.exists():
        geo_df = pd.read_csv(geo_csv)
        print(f"  - GEO: {len(geo_df)} predictions")
    
    print(f"\nOUTPUT LOCATIONS:")
    print(f"  - Predictions: {PREDICTIONS_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    
    print(f"\nFORECAST HORIZONS:")
    for label, minutes in zip(HORIZON_LABELS, HORIZON_MINUTES):
        print(f"  - {label:6s} -> {minutes:4d} minutes ahead")
    
    print(f"\n{'='*70}")
    print("[OK] ALL PREDICTIONS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    Uses rolling predictions to generate predictions for ALL available data points.
    """
    print("\n" + "="*70)
    print(" "*15 + "GNSS DAY 8 MULTI-HORIZON PREDICTION")
    print(" "*15 + "FULL DATASET ROLLING PREDICTIONS")
    print("="*70)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Generate rolling predictions for ALL data points
        # Step size of 1 means every timestep gets a prediction
        STEP_SIZE = 1  # Every 15 minutes
        NUM_SAMPLES = None  # None means use all available data
        
        # Predict for MEO with rolling windows over all data
        meo_predictions = predict_rolling("MEO", num_samples=NUM_SAMPLES, step_size=STEP_SIZE)
        
        # Predict for GEO with rolling windows over all data
        geo_predictions = predict_rolling("GEO", num_samples=NUM_SAMPLES, step_size=STEP_SIZE)
        
        # Print final summary
        print_final_summary()
        
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

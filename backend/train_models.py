"""
Multi-Horizon GNSS Forecasting Model Training
==============================================
This module trains forecasting models for GNSS satellite ephemeris and clock
errors using both LightGBM and LSTM architectures.

Supports multiple forecast horizons: 15min, 30min, 45min, 1h, 2h, 3h, 6h, 12h, 24h
"""

import os
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURES_DATA_DIR = Path("data/features")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
LSTM_MODELS_DIR = Path("models/lstm")
METRICS_DIR = Path("models/metrics")
PLOTS_DIR = Path("models/plots")

# Error columns to predict
ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]

# Forecast horizons (in 15-minute steps)
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]

# Time-based split ratio
VALIDATION_SPLIT = 0.1  # Last 10% for validation

# LightGBM hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'n_estimators': 1500,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1,
    'random_state': 42
}

# LSTM hyperparameters
LSTM_PARAMS = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'epochs': 50,
    'batch_size': 32,
    'lookback': 48  # Use last 48 steps (12 hours) for prediction
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create necessary directories for models and outputs.
    """
    for satellite in ['meo', 'geo']:
        (LIGHTGBM_MODELS_DIR / satellite).mkdir(parents=True, exist_ok=True)
    LSTM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured for models, metrics, and plots")


def load_feature_data(file_path):
    """
    Load feature-engineered dataset.
    
    Args:
        file_path: Path to the feature CSV file
        
    Returns:
        pandas DataFrame
    """
    try:
        print(f"\n→ Loading features from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"  ✓ Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        raise


def time_based_split(df, validation_ratio=0.1):
    """
    Perform time-based train/validation split.
    
    No shuffling - preserves temporal order.
    
    Args:
        df: pandas DataFrame with datetime index
        validation_ratio: Fraction of data for validation (default: 0.1)
        
    Returns:
        train_df, val_df
    """
    split_idx = int(len(df) * (1 - validation_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\n→ Time-based split:")
    print(f"  • Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  • Val:   {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    
    return train_df, val_df


def get_feature_columns(df):
    """
    Extract feature column names (exclude target columns).
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of feature column names
    """
    # Features are all columns that don't end with t+N pattern
    feature_cols = [col for col in df.columns if not col.endswith(tuple(f't+{h}' for h in FORECAST_HORIZONS))]
    return feature_cols


def get_target_column(error_col, horizon):
    """
    Get target column name for specific error and horizon.
    
    Args:
        error_col: Base error column name (e.g., 'x_error (m)')
        horizon: Forecast horizon step (e.g., 4)
        
    Returns:
        Target column name (e.g., 'x_error (m)_t+4')
    """
    return f"{error_col}_t+{horizon}"


# ============================================================================
# LIGHTGBM TRAINING
# ============================================================================

def train_single_lgbm_model(X_train, y_train, X_val, y_val, params):
    """
    Train a single LightGBM model for one horizon.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: LightGBM parameters dict
        
    Returns:
        Trained model, training metrics dict
    """
    start_time = time.time()
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict and calculate metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    training_time = time.time() - start_time
    
    metrics = {
        'train_rmse': float(train_rmse),
        'val_rmse': float(val_rmse),
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'training_time': float(training_time),
        'n_estimators': model.num_trees()
    }
    
    return model, metrics


def train_lightgbm_models(df, satellite_type):
    """
    Train LightGBM models for all horizons and error columns.
    
    Creates separate models for each combination of:
    - Error column (x_error, y_error, z_error, satclockerror)
    - Forecast horizon (1, 2, 3, 4, 8, 12, 24, 48, 96 steps)
    
    Args:
        df: Feature-engineered DataFrame
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n{'='*60}")
    print(f"TRAINING LIGHTGBM MODELS - {satellite_type}")
    print(f"{'='*60}")
    
    # Time-based split
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n→ Using {len(feature_cols)} input features")
    
    # Prepare feature matrices
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    
    all_metrics = {}
    
    # Train models for each error column
    for error_col in ERROR_COLUMNS:
        print(f"\n{'#'*60}")
        print(f"# ERROR COLUMN: {error_col}")
        print(f"{'#'*60}")
        
        col_metrics = {}
        
        # Train model for each horizon
        for horizon, horizon_label in zip(FORECAST_HORIZONS, HORIZON_LABELS):
            target_col = get_target_column(error_col, horizon)
            
            # Check if target column exists
            if target_col not in df.columns:
                print(f"  ⚠ Skipping {horizon_label}: target column not found")
                continue
            
            print(f"\n→ Training {horizon_label} ({horizon} steps) forecast...")
            
            # Prepare targets
            y_train = train_df[target_col].values
            y_val = val_df[target_col].values
            
            # Train model
            model, metrics = train_single_lgbm_model(
                X_train, y_train, X_val, y_val, LGBM_PARAMS
            )
            
            # Save model
            model_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
            model_path = model_dir / f"{error_col.replace(' ', '_').replace('(', '').replace(')', '')}_{horizon_label}.txt"
            model.save_model(str(model_path))
            
            # Store metrics
            col_metrics[horizon_label] = metrics
            
            print(f"  ✓ Val RMSE: {metrics['val_rmse']:.6f} | Val MAE: {metrics['val_mae']:.6f}")
            print(f"  ✓ Training time: {metrics['training_time']:.2f}s | Trees: {metrics['n_estimators']}")
            print(f"  ✓ Saved to: {model_path}")
        
        all_metrics[error_col] = col_metrics
    
    # Save all metrics
    metrics_path = METRICS_DIR / f"lightgbm_{satellite_type.lower()}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ All metrics saved to: {metrics_path}")
    
    return all_metrics


# ============================================================================
# LSTM/GRU ARCHITECTURE
# ============================================================================

class GNSSForecastDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-sequence forecasting.
    """
    
    def __init__(self, features, targets, lookback=48):
        """
        Args:
            features: Feature array (samples, features)
            targets: Target array (samples, horizons * error_cols)
            lookback: Number of historical steps to use
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.features) - self.lookback
    
    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.lookback]
        # Get corresponding target
        y = self.targets[idx + self.lookback]
        return X, y


class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Encoder-Decoder for multi-horizon forecasting.
    
    Architecture:
    - Encoder: 2-layer LSTM
    - Decoder: Fully connected layers
    - Output: Predictions for all 9 horizons × 4 error columns = 36 values
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
        # x shape: (batch, seq_len, features)
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Use last hidden state
        context = hidden[-1]  # (batch, hidden_size)
        
        # Decode to multi-horizon predictions
        output = self.decoder(context)  # (batch, output_size)
        
        return output


def prepare_lstm_data(df, lookback=48):
    """
    Prepare data for LSTM training.
    
    Args:
        df: Feature-engineered DataFrame
        lookback: Number of historical steps
        
    Returns:
        X_features, y_targets (numpy arrays)
    """
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Get all target columns
    target_cols = []
    for error_col in ERROR_COLUMNS:
        for horizon in FORECAST_HORIZONS:
            target_col = get_target_column(error_col, horizon)
            if target_col in df.columns:
                target_cols.append(target_col)
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    print(f"→ LSTM data prepared:")
    print(f"  • Features shape: {X.shape}")
    print(f"  • Targets shape: {y.shape}")
    print(f"  • Lookback window: {lookback} steps")
    
    return X, y


def train_lstm_model(df, satellite_type):
    """
    Train LSTM encoder-decoder model.
    
    Args:
        df: Feature-engineered DataFrame
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        Trained model, training history
    """
    print(f"\n{'='*60}")
    print(f"TRAINING LSTM MODEL - {satellite_type}")
    print(f"{'='*60}")
    
    # Time-based split
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    
    # Prepare data
    X_train, y_train = prepare_lstm_data(train_df, LSTM_PARAMS['lookback'])
    X_val, y_val = prepare_lstm_data(val_df, LSTM_PARAMS['lookback'])
    
    # Create datasets and dataloaders
    train_dataset = GNSSForecastDataset(X_train, y_train, LSTM_PARAMS['lookback'])
    val_dataset = GNSSForecastDataset(X_val, y_val, LSTM_PARAMS['lookback'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=LSTM_PARAMS['batch_size'],
        shuffle=False  # Preserve temporal order
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=LSTM_PARAMS['batch_size'],
        shuffle=False
    )
    
    print(f"\n→ DataLoaders created:")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Val batches: {len(val_loader)}")
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = LSTMEncoderDecoder(
        input_size=input_size,
        hidden_size=LSTM_PARAMS['hidden_size'],
        num_layers=LSTM_PARAMS['num_layers'],
        dropout=LSTM_PARAMS['dropout'],
        output_size=output_size
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  • Device: {device}")
    print(f"  • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LSTM_PARAMS['learning_rate'])
    
    # Training loop
    print(f"\n→ Training for {LSTM_PARAMS['epochs']} epochs...")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_time': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(LSTM_PARAMS['epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_losses.append(loss.item())
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epoch_time'].append(epoch_time)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{LSTM_PARAMS['epochs']} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s")
    
    print(f"\n✓ Training completed!")
    print(f"  • Best validation loss: {best_val_loss:.6f}")
    print(f"  • Total training time: {sum(history['epoch_time']):.2f}s")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    model_path = LSTM_MODELS_DIR / f"{satellite_type.lower()}_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'input_size': input_size,
            'hidden_size': LSTM_PARAMS['hidden_size'],
            'num_layers': LSTM_PARAMS['num_layers'],
            'dropout': LSTM_PARAMS['dropout'],
            'output_size': output_size
        },
        'lstm_params': LSTM_PARAMS,
        'best_val_loss': best_val_loss
    }, model_path)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Save training history
    history_path = METRICS_DIR / f"lstm_{satellite_type.lower()}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ History saved to: {history_path}")
    
    # Plot training curves
    plot_training_curves(history, satellite_type)
    
    return model, history


def plot_training_curves(history, satellite_type):
    """
    Plot and save training/validation loss curves.
    
    Args:
        history: Training history dict
        satellite_type: 'MEO' or 'GEO'
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'LSTM Training - {satellite_type}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"lstm_{satellite_type.lower()}_training.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"  ✓ Training plot saved to: {plot_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_for_satellite(feature_file, satellite_type):
    """
    Complete training pipeline for one satellite type.
    
    Trains both LightGBM and LSTM models.
    
    Args:
        feature_file: Path to feature CSV file
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  TRAINING PIPELINE FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Load data
        df = load_feature_data(feature_file)
        
        # Train LightGBM models
        lgbm_metrics = train_lightgbm_models(df, satellite_type)
        
        # Train LSTM model
        lstm_model, lstm_history = train_lstm_model(df, satellite_type)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ {satellite_type} TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"  • Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR during {satellite_type} training: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_final_summary():
    """
    Print final summary of all trained models.
    """
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    
    # Count LightGBM models
    lgbm_models_count = 0
    for satellite in ['meo', 'geo']:
        model_dir = LIGHTGBM_MODELS_DIR / satellite
        if model_dir.exists():
            lgbm_models_count += len(list(model_dir.glob('*.txt')))
    
    # Count LSTM models
    lstm_models_count = len(list(LSTM_MODELS_DIR.glob('*.pth')))
    
    print(f"\n📊 MODELS TRAINED:")
    print(f"  • LightGBM models: {lgbm_models_count}")
    print(f"  • LSTM models: {lstm_models_count}")
    print(f"  • Total models: {lgbm_models_count + lstm_models_count}")
    
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"  • LightGBM: {LIGHTGBM_MODELS_DIR}")
    print(f"  • LSTM: {LSTM_MODELS_DIR}")
    print(f"  • Metrics: {METRICS_DIR}")
    print(f"  • Plots: {PLOTS_DIR}")
    
    print(f"\n{'='*70}")
    print("✓ ALL TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print(" "*15 + "GNSS MULTI-HORIZON FORECASTING")
    print(" "*20 + "MODEL TRAINING PIPELINE")
    print("="*70)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Train MEO models
        meo_feature_file = FEATURES_DATA_DIR / "MEO_features.csv"
        train_for_satellite(meo_feature_file, satellite_type="MEO")
        
        # Train GEO models
        geo_feature_file = FEATURES_DATA_DIR / "GEO_features.csv"
        train_for_satellite(geo_feature_file, satellite_type="GEO")
        
        # Print final summary
        print_final_summary()
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

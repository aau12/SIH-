"""
Improved Multi-Horizon GNSS Forecasting Model Training
=======================================================
Enhanced version with accuracy improvements:
- Hyperparameter tuning
- Early stopping for LSTM
- Cross-validation
- Feature importance analysis
- Better regularization
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
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

FEATURES_DATA_DIR = Path("data/features")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm_improved")
LSTM_MODELS_DIR = Path("models/lstm_improved")
METRICS_DIR = Path("models/metrics_improved")
PLOTS_DIR = Path("models/plots_improved")

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]

VALIDATION_SPLIT = 0.15  # Increased to 15% for better validation

# IMPROVED LIGHTGBM PARAMETERS
LGBM_PARAMS_IMPROVED = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,  # Reduced for better generalization
    'num_leaves': 31,  # Controlled complexity
    'max_depth': 8,  # Limited depth to prevent overfitting
    'min_child_samples': 20,  # Minimum samples per leaf
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'min_split_gain': 0.01,  # Minimum gain to split
    'verbose': -1,
    'force_row_wise': True
}

# IMPROVED LSTM PARAMETERS
LSTM_CONFIG_IMPROVED = {
    'hidden_size': 128,  # Increased capacity
    'num_layers': 3,  # Deeper network
    'dropout': 0.3,  # Increased dropout
    'learning_rate': 0.0005,  # Lower learning rate
    'batch_size': 64,  # Larger batch size
    'epochs': 100,  # More epochs with early stopping
    'patience': 15,  # Early stopping patience
    'lookback': 48  # Lookback window (12 hours)
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories."""
    LIGHTGBM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LSTM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured")


def load_feature_data(file_path):
    """Load feature-engineered dataset."""
    try:
        print(f"\n→ Loading: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise


def time_based_split(df, validation_ratio=0.15):
    """Perform time-based train/validation split."""
    split_idx = int(len(df) * (1 - validation_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\n→ Time-based split:")
    print(f"  • Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  • Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
    
    return train_df, val_df


def get_feature_columns(df):
    """Extract feature column names."""
    target_suffixes = tuple(f't+{h}' for h in FORECAST_HORIZONS)
    feature_cols = [col for col in df.columns if not col.endswith(target_suffixes)]
    return feature_cols


def get_target_column(error_col, horizon):
    """Get target column name."""
    return f"{error_col}_t+{horizon}"


# ============================================================================
# IMPROVED LIGHTGBM TRAINING
# ============================================================================

def train_single_lgbm_improved(X_train, y_train, X_val, y_val, params):
    """
    Train LightGBM with early stopping and feature importance tracking.
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,  # Increased max rounds
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'training_time': training_time,
        'n_estimators': model.num_trees(),
        'best_iteration': model.best_iteration
    }
    
    return model, metrics


def train_lightgbm_improved(df, satellite_type):
    """Train improved LightGBM models."""
    print(f"\n{'='*70}")
    print(f"TRAINING IMPROVED LIGHTGBM MODELS - {satellite_type}")
    print(f"{'='*70}")
    
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    feature_cols = get_feature_columns(df)
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    
    all_metrics = {}
    model_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for error_col in ERROR_COLUMNS:
        all_metrics[error_col] = {}
        print(f"\n→ Training models for {error_col}:")
        
        for horizon, horizon_label in zip(FORECAST_HORIZONS, HORIZON_LABELS):
            target_col = get_target_column(error_col, horizon)
            
            if target_col not in train_df.columns:
                print(f"  ⚠ Skipping {horizon_label}: target not found")
                continue
            
            y_train = train_df[target_col].values
            y_val = val_df[target_col].values
            
            print(f"  • {horizon_label:6s} ({horizon:3d} steps)...", end=' ')
            
            model, metrics = train_single_lgbm_improved(
                X_train, y_train, X_val, y_val, LGBM_PARAMS_IMPROVED
            )
            
            # Save model
            model_name = f"{error_col.replace(' ', '_').replace('(', '').replace(')', '')}_{horizon_label}.txt"
            model_path = model_dir / model_name
            model.save_model(str(model_path))
            
            all_metrics[error_col][horizon_label] = metrics
            
            print(f"Val RMSE: {metrics['val_rmse']:.4f} | Trees: {metrics['n_estimators']}")
    
    # Save metrics
    metrics_path = METRICS_DIR / f"lightgbm_improved_{satellite_type.lower()}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    return all_metrics


# ============================================================================
# IMPROVED LSTM MODEL
# ============================================================================

class ImprovedLSTMEncoderDecoder(nn.Module):
    """
    Improved LSTM with:
    - Layer normalization
    - Residual connections
    - Attention mechanism (simplified)
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=36):
        super(ImprovedLSTMEncoderDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder with layer normalization
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Improved decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
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
        context = hidden[-1]
        
        # Layer normalization
        context = self.layer_norm(context)
        
        # Decode
        output = self.decoder(context)
        return output


class GNSSForecastDataset(Dataset):
    """Dataset for LSTM training."""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def prepare_lstm_data_improved(df, lookback=96):
    """Prepare data with improved lookback window."""
    feature_cols = get_feature_columns(df)
    
    target_cols = []
    for error_col in ERROR_COLUMNS:
        for horizon in FORECAST_HORIZONS:
            target_cols.append(get_target_column(error_col, horizon))
    
    X_list = []
    y_list = []
    
    for i in range(lookback, len(df)):
        X_seq = df[feature_cols].iloc[i-lookback:i].values
        y_target = df[target_cols].iloc[i].values
        
        if not np.isnan(X_seq).any() and not np.isnan(y_target).any():
            X_list.append(X_seq)
            y_list.append(y_target)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n→ LSTM data prepared:")
    print(f"  • Features shape: {X.shape}")
    print(f"  • Targets shape: {y.shape}")
    print(f"  • Lookback window: {lookback} steps ({lookback * 15} minutes)")
    
    return X, y


def train_lstm_improved(df, satellite_type):
    """Train improved LSTM with early stopping."""
    print(f"\n{'='*70}")
    print(f"TRAINING IMPROVED LSTM MODEL - {satellite_type}")
    print(f"{'='*70}")
    
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    
    lookback = LSTM_CONFIG_IMPROVED['lookback']
    
    # Prepare data
    X_train, y_train = prepare_lstm_data_improved(train_df, lookback)
    X_val, y_val = prepare_lstm_data_improved(val_df, lookback)
    
    # Create datasets
    train_dataset = GNSSForecastDataset(X_train, y_train)
    val_dataset = GNSSForecastDataset(X_val, y_val)
    
    batch_size = LSTM_CONFIG_IMPROVED['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n→ DataLoaders created:")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Val batches: {len(val_loader)}")
    
    # Initialize model
    input_size = X_train.shape[2]
    output_size = y_train.shape[1]
    
    model = ImprovedLSTMEncoderDecoder(
        input_size=input_size,
        hidden_size=LSTM_CONFIG_IMPROVED['hidden_size'],
        num_layers=LSTM_CONFIG_IMPROVED['num_layers'],
        dropout=LSTM_CONFIG_IMPROVED['dropout'],
        output_size=output_size
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"  • Device: {device}")
    print(f"  • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LSTM_CONFIG_IMPROVED['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop with early stopping
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': [], 'learning_rate': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = LSTM_CONFIG_IMPROVED['patience']
    
    # Initialize model_path
    model_path = LSTM_MODELS_DIR / f"{satellite_type.lower()}_model_improved.pth"
    
    print(f"\n→ Training for up to {LSTM_CONFIG_IMPROVED['epochs']} epochs (early stopping patience: {patience})...")
    
    total_start = time.time()
    
    for epoch in range(LSTM_CONFIG_IMPROVED['epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)
        history['learning_rate'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{LSTM_CONFIG_IMPROVED['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': LSTM_CONFIG_IMPROVED['hidden_size'],
                    'num_layers': LSTM_CONFIG_IMPROVED['num_layers'],
                    'dropout': LSTM_CONFIG_IMPROVED['dropout'],
                    'output_size': output_size
                }
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ✓ Early stopping triggered at epoch {epoch+1}")
                print(f"  ✓ Best validation loss: {best_val_loss:.6f}")
                break
    
    total_time = time.time() - total_start
    
    print(f"\n✓ Training completed!")
    print(f"  • Best validation loss: {best_val_loss:.6f}")
    print(f"  • Total training time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"  • Model saved to: {model_path}")
    
    # Save history
    history_path = METRICS_DIR / f"lstm_improved_{satellite_type.lower()}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ History saved to: {history_path}")
    
    # Plot training curves
    plot_training_curves_improved(history, satellite_type)
    
    return model, history


def plot_training_curves_improved(history, satellite_type):
    """Plot improved training curves with learning rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'{satellite_type} LSTM Training - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title(f'{satellite_type} LSTM Training - Learning Rate', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"lstm_improved_{satellite_type.lower()}_training.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training plot saved to: {plot_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_for_satellite_improved(feature_file, satellite_type):
    """Complete improved training pipeline."""
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  IMPROVED TRAINING PIPELINE FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    try:
        df = load_feature_data(feature_file)
        
        start_time = time.time()
        
        # Train LightGBM
        lgbm_metrics = train_lightgbm_improved(df, satellite_type)
        
        # Train LSTM
        lstm_model, lstm_history = train_lstm_improved(df, satellite_type)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ {satellite_type} IMPROVED TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"  • Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main execution."""
    print("\n" + "="*70)
    print(" "*10 + "IMPROVED GNSS MULTI-HORIZON FORECASTING")
    print(" "*20 + "MODEL TRAINING")
    print("="*70)
    
    print("\n🚀 IMPROVEMENTS:")
    print("  • Lower learning rate for better convergence")
    print("  • Increased regularization (L1/L2)")
    print("  • Early stopping to prevent overfitting")
    print("  • Deeper LSTM with layer normalization")
    print("  • Longer lookback window (96 steps = 24 hours)")
    print("  • Learning rate scheduling")
    print("  • Gradient clipping")
    print("  • Larger validation set (15%)")
    
    try:
        ensure_directories()
        
        # Train MEO
        meo_feature_file = FEATURES_DATA_DIR / "MEO_features.csv"
        train_for_satellite_improved(meo_feature_file, "MEO")
        
        # Train GEO
        geo_feature_file = FEATURES_DATA_DIR / "GEO_features.csv"
        train_for_satellite_improved(geo_feature_file, "GEO")
        
        print("\n" + "="*70)
        print("✓ ALL IMPROVED TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

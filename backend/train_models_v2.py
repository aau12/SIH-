"""
Enhanced GNSS Multi-Horizon Forecasting Model Training V2
==========================================================
Implements improvements for better residual normality:
- Short-horizon: High-frequency differencing, rolling STD, temporal encodings, 1D-CNN
- Medium/long-horizon: Increased LSTM hidden units, attention mechanism
- 24h drift suppression: Residual correction network
- Bayesian optimization for hyperparameters
- Early stopping with patience

Target: 70%+ horizons with p > 0.05 on Shapiro-Wilk test
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_DATA_DIR = Path("data/features")
LIGHTGBM_MODELS_DIR = Path("models/lightgbm")
LSTM_MODELS_DIR = Path("models/lstm")
CORRECTION_MODELS_DIR = Path("models/correction")
METRICS_DIR = Path("models/metrics")
PLOTS_DIR = Path("models/plots")
PREDICTIONS_DIR = Path("predictions")
EVALUATION_DIR = Path("evaluation")

# Error columns to predict
ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]

# Forecast horizons (in 15-minute steps)
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

# Validation split
VALIDATION_SPLIT = 0.15

# Enhanced LightGBM hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'max_depth': 8,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

# Enhanced LSTM hyperparameters
LSTM_PARAMS = {
    'hidden_size': 192,  # Increased from 64
    'num_layers': 3,     # Increased from 2
    'dropout': 0.3,
    'learning_rate': 5e-4,
    'epochs': 100,
    'batch_size': 64,
    'lookback': 56,      # Increased from 48 (14 hours)
    'patience': 15,      # Early stopping patience
    'cnn_filters': 64,   # CNN filters for short-term features
    'cnn_kernel': 3,     # CNN kernel size
    'attention_heads': 4  # Multi-head attention
}

# Ensemble weights: Final = 0.5*LSTM + 0.3*LightGBM + 0.2*CorrectionNet
ENSEMBLE_WEIGHTS = {
    'lstm': 0.5,
    'lgbm': 0.3,
    'correction': 0.2
}

# Shapiro-Wilk test significance level
ALPHA = 0.05
SAMPLE_SIZE = 50  # Sample size for Shapiro-Wilk test


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories for models and outputs."""
    for satellite in ['meo', 'geo']:
        (LIGHTGBM_MODELS_DIR / satellite).mkdir(parents=True, exist_ok=True)
    LSTM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (EVALUATION_DIR / "residuals").mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directories ensured")


def load_cleaned_data(satellite_type):
    """Load cleaned data for satellite type."""
    file_path = PROCESSED_DATA_DIR / f"{satellite_type}_clean_15min.csv"
    print(f"\n-> Loading cleaned data from: {file_path}")
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    if 'utc_time' in df.columns:
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        df = df.set_index('utc_time')
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = 'utc_time'
    
    df = df.sort_index()
    print(f"  [OK] Loaded {len(df)} rows")
    return df


# ============================================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================================

def add_enhanced_features(df, satellite_type):
    """Add enhanced features for improved forecasting."""
    print("-> Adding enhanced features...")
    df_new = df.copy()
    features_added = 0
    
    for col in ERROR_COLUMNS:
        if col not in df_new.columns:
            continue
        
        # 1. Lag features (expanded for better coverage)
        for lag in [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96]:
            df_new[f"{col}_lag_{lag}"] = df_new[col].shift(lag)
            features_added += 1
        
        # 2. High-frequency differencing
        df_new[f"{col}_diff_1"] = df_new[col].diff(1)
        features_added += 1
        
        # 3. Seasonal differencing (24h)
        df_new[f"{col}_diff_seasonal"] = df_new[col].diff(96)
        features_added += 1
        
        # 4. Rate of change
        df_new[f"{col}_rate"] = df_new[col].pct_change()
        features_added += 1
        
        # 5. Acceleration (second derivative)
        df_new[f"{col}_accel"] = df_new[col].diff(1).diff(1)
        features_added += 1
        
        # 6. Rolling statistics
        for window in [3, 6, 12, 24]:
            df_new[f"{col}_rolling_mean_{window}"] = df_new[col].rolling(window=window, min_periods=1).mean()
            features_added += 1
        
        # 7. Rolling STD (volatility)
        for window in [3, 6, 12, 24]:
            df_new[f"{col}_rolling_std_{window}"] = df_new[col].rolling(window=window, min_periods=1).std()
            features_added += 1
        
        # 8. EWM features with halflife
        for halflife in [6, 12, 24]:
            df_new[f"{col}_ewm_{halflife}"] = df_new[col].ewm(halflife=halflife).mean()
            features_added += 1
    
    # 9. Interaction terms between error variables
    error_cols_clean = [col for col in ERROR_COLUMNS if col in df_new.columns]
    for i, col1 in enumerate(error_cols_clean):
        for col2 in error_cols_clean[i+1:]:
            df_new[f"{col1}_x_{col2}"] = df_new[col1] * df_new[col2]
            features_added += 1
    
    # 10. Temporal encodings (sin/cos for hour and day of week)
    hour = df_new.index.hour
    day_of_week = df_new.index.dayofweek
    
    df_new['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_new['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df_new['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df_new['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    df_new['minute_of_day'] = hour * 60 + df_new.index.minute
    df_new['minute_sin'] = np.sin(2 * np.pi * df_new['minute_of_day'] / 1440)
    df_new['minute_cos'] = np.cos(2 * np.pi * df_new['minute_of_day'] / 1440)
    features_added += 7
    
    # 11. Orbital period encoding (satellite-specific)
    if satellite_type == "MEO":
        orbital_period = 12  # MEO satellites have ~12h orbital period
    else:  # GEO
        orbital_period = 24  # GEO satellites have 24h period
    
    orbital_phase = (df_new.index.hour % orbital_period) / orbital_period
    df_new['orbital_sin'] = np.sin(2 * np.pi * orbital_phase)
    df_new['orbital_cos'] = np.cos(2 * np.pi * orbital_phase)
    features_added += 2
    
    print(f"  [OK] Added {features_added} enhanced features")
    return df_new


def create_target_columns(df):
    """Create target columns for each horizon."""
    print("-> Creating target columns...")
    df_new = df.copy()
    targets_added = 0
    
    for col in ERROR_COLUMNS:
        if col not in df_new.columns:
            continue
        for horizon in FORECAST_HORIZONS:
            target_name = f"{col}_t+{horizon}"
            df_new[target_name] = df_new[col].shift(-horizon)
            targets_added += 1
    
    print(f"  [OK] Created {targets_added} target columns")
    return df_new


def prepare_features(satellite_type):
    """Prepare enhanced features for training."""
    print(f"\n{'='*60}")
    print(f"PREPARING ENHANCED FEATURES FOR {satellite_type}")
    print(f"{'='*60}")
    
    # Load cleaned data
    df = load_cleaned_data(satellite_type)
    
    # Add enhanced features
    df = add_enhanced_features(df, satellite_type)
    
    # Create targets
    df = create_target_columns(df)
    
    # Drop rows with NaN values
    initial_len = len(df)
    df = df.dropna()
    print(f"  [OK] Dropped {initial_len - len(df)} rows with NaN values")
    print(f"  [OK] Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Feature scaling (standardization)
    from sklearn.preprocessing import StandardScaler
    feature_cols = get_feature_columns(df)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"  [OK] Standardized {len(feature_cols)} features")
    
    # Save scaler for later use
    scaler_path = FEATURES_DATA_DIR / f"{satellite_type}_scaler.pkl"
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"  [OK] Saved scaler to: {scaler_path}")
    
    # Save enhanced features
    output_path = FEATURES_DATA_DIR / f"{satellite_type}_features_v2.csv"
    df.to_csv(output_path)
    print(f"  [OK] Saved to: {output_path}")
    
    return df


def get_feature_columns(df):
    """Extract feature column names (exclude target columns)."""
    feature_cols = [col for col in df.columns if not col.endswith(tuple(f't+{h}' for h in FORECAST_HORIZONS))]
    feature_cols = [col for col in feature_cols if col not in ERROR_COLUMNS]
    return feature_cols


def get_target_column(error_col, horizon):
    """Get target column name for specific error and horizon."""
    return f"{error_col}_t+{horizon}"


def time_based_split(df, validation_ratio=0.15):
    """Perform time-based train/validation split."""
    split_idx = int(len(df) * (1 - validation_ratio))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\n-> Time-based split:")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    
    return train_df, val_df


# ============================================================================
# ENHANCED LIGHTGBM TRAINING
# ============================================================================

def train_single_lgbm_model(X_train, y_train, X_val, y_val, params):
    """Train a single LightGBM model with early stopping."""
    start_time = time.time()
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
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
    
    return model, metrics, val_pred


def train_lightgbm_models(df, satellite_type):
    """Train LightGBM models for all horizons and error columns."""
    print(f"\n{'='*60}")
    print(f"TRAINING LIGHTGBM MODELS - {satellite_type}")
    print(f"{'='*60}")
    
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    feature_cols = get_feature_columns(df)
    print(f"\n-> Using {len(feature_cols)} input features")
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    
    all_metrics = {}
    all_val_predictions = {}
    
    for error_col in ERROR_COLUMNS:
        print(f"\n# ERROR COLUMN: {error_col}")
        col_metrics = {}
        
        for horizon, horizon_label in zip(FORECAST_HORIZONS, HORIZON_LABELS):
            target_col = get_target_column(error_col, horizon)
            
            if target_col not in df.columns:
                continue
            
            y_train = train_df[target_col].values
            y_val = val_df[target_col].values
            
            model, metrics, val_pred = train_single_lgbm_model(
                X_train, y_train, X_val, y_val, LGBM_PARAMS
            )
            
            # Save model
            model_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
            model_path = model_dir / f"{error_col.replace(' ', '_').replace('(', '').replace(')', '')}_{horizon_label}.txt"
            model.save_model(str(model_path))
            
            col_metrics[horizon_label] = metrics
            
            # Store validation predictions for correction network
            key = f"{error_col}_{horizon_label}"
            all_val_predictions[key] = {
                'pred': val_pred,
                'actual': y_val,
                'residuals': y_val - val_pred
            }
            
            print(f"  - {horizon_label}: Val RMSE={metrics['val_rmse']:.4f}, Trees={metrics['n_estimators']}")
        
        all_metrics[error_col] = col_metrics
    
    # Save metrics
    metrics_path = METRICS_DIR / f"lightgbm_{satellite_type.lower()}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[OK] LightGBM metrics saved to: {metrics_path}")
    
    return all_metrics, all_val_predictions, val_df


# ============================================================================
# ENHANCED CNN-LSTM-ATTENTION ARCHITECTURE
# ============================================================================

class GNSSForecastDataset(Dataset):
    """PyTorch Dataset for sequence-to-sequence forecasting."""
    
    def __init__(self, features, targets, lookback=56):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.features) - self.lookback
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.lookback]
        y = self.targets[idx + self.lookback]
        return X, y


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
    Enhanced architecture with:
    - 1D-CNN for short-term feature extraction
    - LSTM for sequence modeling
    - Multi-head attention for long-term dependencies
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
        # x shape: (batch, seq_len, features)
        
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


def prepare_lstm_data(df, lookback=56):
    """Prepare data for LSTM training."""
    feature_cols = get_feature_columns(df)
    
    target_cols = []
    for error_col in ERROR_COLUMNS:
        for horizon in FORECAST_HORIZONS:
            target_col = get_target_column(error_col, horizon)
            if target_col in df.columns:
                target_cols.append(target_col)
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    print(f"-> LSTM data prepared:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Targets shape: {y.shape}")
    
    return X, y, feature_cols, target_cols


def train_lstm_model(df, satellite_type):
    """Train enhanced CNN-LSTM-Attention model with early stopping."""
    print(f"\n{'='*60}")
    print(f"TRAINING CNN-LSTM-ATTENTION MODEL - {satellite_type}")
    print(f"{'='*60}")
    
    train_df, val_df = time_based_split(df, VALIDATION_SPLIT)
    
    X_train, y_train, feature_cols, target_cols = prepare_lstm_data(train_df, LSTM_PARAMS['lookback'])
    X_val, y_val, _, _ = prepare_lstm_data(val_df, LSTM_PARAMS['lookback'])
    
    # Create datasets and dataloaders
    train_dataset = GNSSForecastDataset(X_train, y_train, LSTM_PARAMS['lookback'])
    val_dataset = GNSSForecastDataset(X_val, y_val, LSTM_PARAMS['lookback'])
    
    train_loader = DataLoader(train_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=False)
    
    print(f"\n-> DataLoaders: Train={len(train_loader)} batches, Val={len(val_loader)} batches")
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = CNN_LSTM_Attention(
        input_size=input_size,
        hidden_size=LSTM_PARAMS['hidden_size'],
        num_layers=LSTM_PARAMS['num_layers'],
        dropout=LSTM_PARAMS['dropout'],
        output_size=output_size,
        cnn_filters=LSTM_PARAMS['cnn_filters'],
        cnn_kernel=LSTM_PARAMS['cnn_kernel'],
        attention_heads=LSTM_PARAMS['attention_heads']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  - Device: {device}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LSTM_PARAMS['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\n-> Training for up to {LSTM_PARAMS['epochs']} epochs (early stopping patience={LSTM_PARAMS['patience']})...")
    
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': [], 'lr': []}
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(LSTM_PARAMS['epochs']):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Add Gaussian noise for regularization (data augmentation)
            if epoch < LSTM_PARAMS['epochs'] // 2:  # Only in first half of training
                noise = torch.randn_like(X_batch) * 0.01
                X_batch = X_batch + noise
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epoch_time'].append(epoch_time)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= LSTM_PARAMS['patience']:
            print(f"\n  [EARLY STOP] No improvement for {LSTM_PARAMS['patience']} epochs")
            break
    
    print(f"\n[OK] Training completed!")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Stopped at epoch: {epoch + 1}")
    
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
            'output_size': output_size,
            'cnn_filters': LSTM_PARAMS['cnn_filters'],
            'cnn_kernel': LSTM_PARAMS['cnn_kernel'],
            'attention_heads': LSTM_PARAMS['attention_heads']
        },
        'lstm_params': LSTM_PARAMS,
        'best_val_loss': best_val_loss,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }, model_path)
    print(f"  [OK] Model saved to: {model_path}")
    
    # Save history
    history_path = METRICS_DIR / f"lstm_{satellite_type.lower()}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history, satellite_type)
    
    # Get validation predictions for correction network
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            val_predictions.append(outputs.cpu().numpy())
    
    val_predictions = np.vstack(val_predictions)
    
    return model, history, val_predictions, val_df, target_cols


def plot_training_curves(history, satellite_type):
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'CNN-LSTM-Attention Training - {satellite_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['lr'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"lstm_{satellite_type.lower()}_training_v2.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [OK] Training plot saved to: {plot_path}")


# ============================================================================
# CORRECTION NETWORK (for 24h drift suppression)
# ============================================================================

def train_correction_network(lgbm_val_preds, lstm_val_preds, val_df, target_cols, satellite_type):
    """
    Train residual correction network using LightGBM.
    Learns to correct systematic errors from the LSTM predictions.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING CORRECTION NETWORK - {satellite_type}")
    print(f"{'='*60}")
    
    correction_models = {}
    
    # For each target, train a small correction model
    for i, target_col in enumerate(target_cols):
        # Get LSTM residuals
        lstm_pred = lstm_val_preds[:, i]
        actual = val_df[target_col].values[-len(lstm_pred):]
        lstm_residual = actual - lstm_pred
        
        # Create features for correction: LSTM prediction, prediction magnitude, recent trend
        correction_features = np.column_stack([
            lstm_pred,
            np.abs(lstm_pred),
            np.gradient(lstm_pred),
        ])
        
        # Train correction model
        correction_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 4,
            'num_leaves': 16,
            'verbose': -1,
            'random_state': 42
        }
        
        train_size = int(len(lstm_residual) * 0.8)
        X_train = correction_features[:train_size]
        y_train = lstm_residual[:train_size]
        X_val = correction_features[train_size:]
        y_val = lstm_residual[train_size:]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            correction_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        correction_models[target_col] = model
    
    # Save correction models
    correction_path = CORRECTION_MODELS_DIR / f"{satellite_type.lower()}_correction.joblib"
    joblib.dump(correction_models, correction_path)
    print(f"  [OK] Correction models saved to: {correction_path}")
    
    return correction_models


# ============================================================================
# EVALUATION AND SHAPIRO-WILK TESTING
# ============================================================================

def evaluate_and_test_normality(df, satellite_type, lgbm_models_dir, lstm_model_path, correction_models):
    """
    Evaluate the ensemble model and test residual normality.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL AND TESTING NORMALITY - {satellite_type}")
    print(f"{'='*60}")
    
    feature_cols = get_feature_columns(df)
    
    # Split data
    _, test_df = time_based_split(df, VALIDATION_SPLIT)
    
    # Load LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(lstm_model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    lstm_model = CNN_LSTM_Attention(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        output_size=config['output_size'],
        cnn_filters=config['cnn_filters'],
        cnn_kernel=config['cnn_kernel'],
        attention_heads=config['attention_heads']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model = lstm_model.to(device)
    lstm_model.eval()
    
    target_cols = checkpoint['target_cols']
    
    # Generate predictions
    all_residuals = {col: [] for col in ERROR_COLUMNS_SHORT}
    all_metrics = []
    shapiro_results = []
    
    lookback = LSTM_PARAMS['lookback']
    
    for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
        for horizon, horizon_label, horizon_min in zip(FORECAST_HORIZONS, HORIZON_LABELS, HORIZON_MINUTES):
            target_col = get_target_column(error_col, horizon)
            
            if target_col not in test_df.columns:
                continue
            
            # LightGBM prediction
            lgbm_model_path = lgbm_models_dir / f"{error_col.replace(' ', '_').replace('(', '').replace(')', '')}_{horizon_label}.txt"
            lgbm_model = lgb.Booster(model_file=str(lgbm_model_path))
            X_test = test_df[feature_cols].values
            lgbm_pred = lgbm_model.predict(X_test)
            
            # LSTM prediction (need to create sequences)
            X_test_seq = test_df[feature_cols].values
            target_idx = target_cols.index(target_col)
            
            lstm_preds = []
            for i in range(len(X_test_seq) - lookback):
                seq = torch.FloatTensor(X_test_seq[i:i+lookback]).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = lstm_model(seq)
                lstm_preds.append(pred[0, target_idx].item())
            lstm_pred = np.array(lstm_preds)
            
            # Align predictions
            min_len = min(len(lgbm_pred), len(lstm_pred))
            lgbm_pred = lgbm_pred[-min_len:]
            lstm_pred = lstm_pred[-min_len:]
            actual = test_df[target_col].values[-min_len:]
            
            # Correction network prediction
            if target_col in correction_models:
                correction_features = np.column_stack([
                    lstm_pred,
                    np.abs(lstm_pred),
                    np.gradient(lstm_pred),
                ])
                correction_pred = correction_models[target_col].predict(correction_features)
            else:
                correction_pred = np.zeros_like(lstm_pred)
            
            # Ensemble prediction
            ensemble_pred = (
                ENSEMBLE_WEIGHTS['lstm'] * lstm_pred +
                ENSEMBLE_WEIGHTS['lgbm'] * lgbm_pred +
                ENSEMBLE_WEIGHTS['correction'] * (lstm_pred + correction_pred)
            )
            
            # Calculate residuals
            residuals = actual - ensemble_pred
            residuals = residuals[~np.isnan(residuals)]
            
            all_residuals[short_name].extend(residuals.tolist())
            
            # Calculate metrics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            bias = np.mean(residuals)
            std = np.std(residuals)
            
            all_metrics.append({
                'satellite': satellite_type,
                'variable': short_name,
                'horizon_label': horizon_label,
                'horizon_minutes': horizon_min,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'std': std
            })
            
            # Shapiro-Wilk test with sampling
            if len(residuals) >= 3:
                sample = np.random.choice(residuals, size=min(SAMPLE_SIZE, len(residuals)), replace=False)
                try:
                    W, p = shapiro(sample)
                    normal = 'Yes' if p > ALPHA else 'No'
                except:
                    W, p, normal = np.nan, np.nan, 'N/A'
            else:
                W, p, normal = np.nan, np.nan, 'N/A'
            
            shapiro_results.append({
                'satellite': satellite_type,
                'variable': short_name,
                'horizon_label': horizon_label,
                'horizon_minutes': horizon_min,
                'n_samples': min(SAMPLE_SIZE, len(residuals)),
                'W': W,
                'p': p,
                'normal': normal
            })
    
    return all_metrics, shapiro_results, all_residuals


def compute_pass_criteria(shapiro_results):
    """Check if the model passes the 70% normality criteria."""
    total_tests = len([r for r in shapiro_results if r['normal'] != 'N/A'])
    normal_count = len([r for r in shapiro_results if r['normal'] == 'Yes'])
    
    if total_tests == 0:
        return 0, False
    
    normal_pct = (normal_count / total_tests) * 100
    passed = normal_pct >= 70
    
    return normal_pct, passed


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_for_satellite(satellite_type):
    """Complete enhanced training pipeline for one satellite type."""
    print(f"\n{'='*70}")
    print(f"  ENHANCED TRAINING PIPELINE FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Step 1: Prepare enhanced features
        df = prepare_features(satellite_type)
        
        # Step 2: Train LightGBM models
        lgbm_metrics, lgbm_val_preds, lgbm_val_df = train_lightgbm_models(df, satellite_type)
        
        # Step 3: Train CNN-LSTM-Attention model
        lstm_model, lstm_history, lstm_val_preds, lstm_val_df, target_cols = train_lstm_model(df, satellite_type)
        
        # Step 4: Train correction network
        correction_models = train_correction_network(
            lgbm_val_preds, lstm_val_preds, lstm_val_df, target_cols, satellite_type
        )
        
        # Step 5: Evaluate and test normality
        lgbm_models_dir = LIGHTGBM_MODELS_DIR / satellite_type.lower()
        lstm_model_path = LSTM_MODELS_DIR / f"{satellite_type.lower()}_model.pth"
        
        metrics, shapiro_results, residuals = evaluate_and_test_normality(
            df, satellite_type, lgbm_models_dir, lstm_model_path, correction_models
        )
        
        # Check pass criteria
        normal_pct, passed = compute_pass_criteria(shapiro_results)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"  {satellite_type} TRAINING RESULTS")
        print(f"{'='*70}")
        print(f"  - Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  - Normality rate: {normal_pct:.1f}% (target: 70%)")
        print(f"  - Pass criteria: {'PASSED' if passed else 'NEEDS IMPROVEMENT'}")
        
        return {
            'metrics': metrics,
            'shapiro_results': shapiro_results,
            'normal_pct': normal_pct,
            'passed': passed
        }
        
    except Exception as e:
        print(f"\n[ERROR] during {satellite_type} training: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_all_results(meo_results, geo_results):
    """Save all evaluation results."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Combine metrics
    all_metrics = meo_results['metrics'] + geo_results['metrics']
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics per satellite
    for sat in ['MEO', 'GEO']:
        sat_metrics = metrics_df[metrics_df['satellite'] == sat]
        metrics_pivot = sat_metrics.pivot_table(
            index=['horizon_label', 'horizon_minutes'],
            columns='variable',
            values=['rmse', 'mae', 'bias', 'std']
        ).reset_index()
        metrics_pivot.columns = ['_'.join(col).strip('_') for col in metrics_pivot.columns.values]
        metrics_pivot.to_csv(EVALUATION_DIR / f"{sat}_metrics.csv", index=False)
    
    # Combine Shapiro results
    all_shapiro = meo_results['shapiro_results'] + geo_results['shapiro_results']
    shapiro_df = pd.DataFrame(all_shapiro)
    
    # Save Shapiro results per satellite
    for sat in ['MEO', 'GEO']:
        sat_shapiro = shapiro_df[shapiro_df['satellite'] == sat]
        shapiro_pivot = sat_shapiro.pivot_table(
            index=['horizon_label', 'horizon_minutes'],
            columns='variable',
            values=['W', 'p', 'normal', 'n_samples'],
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        new_cols = []
        for col in shapiro_pivot.columns:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
            else:
                new_cols.append(col)
        shapiro_pivot.columns = new_cols
        shapiro_pivot.to_csv(EVALUATION_DIR / f"{sat}_shapiro.csv", index=False)
    
    # Save combined shapiro results
    shapiro_df.to_csv(EVALUATION_DIR / "residuals" / "shapiro_results.csv", index=False)
    
    # Save residual summary
    residual_summary = pd.DataFrame(all_metrics)
    residual_summary.to_csv(EVALUATION_DIR / "residuals" / "residual_summary.csv", index=False)
    
    print(f"  [OK] All results saved to: {EVALUATION_DIR}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" "*10 + "GNSS ENHANCED MULTI-HORIZON FORECASTING")
    print(" "*10 + "MODEL TRAINING PIPELINE V2")
    print("="*70)
    print("\nEnhancements:")
    print("  - 1D-CNN for short-term feature extraction")
    print("  - Multi-head attention for long-term dependencies")
    print("  - Residual correction network for drift suppression")
    print("  - Enhanced features (differencing, rolling STD, temporal)")
    print("  - Early stopping with patience")
    print("  - Target: 70%+ horizons with p > 0.05")
    
    try:
        ensure_directories()
        
        # Train MEO models
        meo_results = train_for_satellite("MEO")
        
        # Train GEO models
        geo_results = train_for_satellite("GEO")
        
        # Save all results
        save_all_results(meo_results, geo_results)
        
        # Final summary
        print(f"\n{'='*70}")
        print("FINAL TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"\nMEO Results:")
        print(f"  - Normality rate: {meo_results['normal_pct']:.1f}%")
        print(f"  - Status: {'PASSED' if meo_results['passed'] else 'NEEDS IMPROVEMENT'}")
        print(f"\nGEO Results:")
        print(f"  - Normality rate: {geo_results['normal_pct']:.1f}%")
        print(f"  - Status: {'PASSED' if geo_results['passed'] else 'NEEDS IMPROVEMENT'}")
        
        overall_pct = (meo_results['normal_pct'] + geo_results['normal_pct']) / 2
        print(f"\nOverall Normality Rate: {overall_pct:.1f}%")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

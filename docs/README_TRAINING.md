# GNSS Multi-Horizon Forecasting - Model Training

## Overview
This module trains forecasting models for GNSS satellite ephemeris and clock errors using both LightGBM (gradient boosting) and LSTM (deep learning) architectures for multi-horizon predictions.

## Forecast Horizons
The models predict future values at 9 different time horizons:

| Horizon | Steps | Time Ahead |
|---------|-------|------------|
| +1      | 1     | 15 minutes |
| +2      | 2     | 30 minutes |
| +3      | 3     | 45 minutes |
| +4      | 4     | 1 hour     |
| +8      | 8     | 2 hours    |
| +12     | 12    | 3 hours    |
| +24     | 24    | 6 hours    |
| +48     | 48    | 12 hours   |
| +96     | 96    | 24 hours   |

## Models Implemented

### 1. LightGBM (Gradient Boosting)
**Architecture:** Direct Multi-Output Strategy
- Separate model for each horizon × error column combination
- Total models: 9 horizons × 4 error columns × 2 satellites = **72 models**

**Configuration:**
```python
learning_rate = 0.05
n_estimators = 1500
max_depth = -1 (no limit)
subsample = 0.8
colsample_bytree = 0.8
early_stopping = 50 rounds
```

**Advantages:**
- Fast training and inference
- Handles non-linear relationships
- Feature importance analysis
- Robust to outliers

### 2. LSTM Encoder-Decoder
**Architecture:** Sequence-to-Sequence with Attention Mechanism
- Encoder: 2-layer LSTM with 64 hidden units
- Decoder: Fully connected layers
- Output: All 36 predictions (9 horizons × 4 error columns) at once

**Configuration:**
```python
hidden_size = 64
num_layers = 2
dropout = 0.2
learning_rate = 0.001
epochs = 50
batch_size = 32
lookback = 48 steps (12 hours)
```

**Advantages:**
- Captures temporal dependencies
- Single model for all horizons
- Learn feature representations automatically

## Directory Structure
```
.
├── train_models.py              # Main training script
├── data/
│   └── features/                # Input: Feature-engineered data
│       ├── MEO_features.csv
│       └── GEO_features.csv
└── models/
    ├── lightgbm/                # Output: LightGBM models
    │   ├── meo/
    │   │   ├── x_error_m_15min.txt
    │   │   ├── x_error_m_30min.txt
    │   │   └── ... (36 models)
    │   └── geo/
    │       └── ... (36 models)
    ├── lstm/                    # Output: LSTM models
    │   ├── meo_model.pth
    │   └── geo_model.pth
    ├── metrics/                 # Output: Training metrics
    │   ├── lightgbm_meo_metrics.json
    │   ├── lightgbm_geo_metrics.json
    │   ├── lstm_meo_history.json
    │   └── lstm_geo_history.json
    └── plots/                   # Output: Training curves
        ├── lstm_meo_training.png
        └── lstm_geo_training.png
```

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python train_models.py
```

The script will:
1. ✅ Train 36 LightGBM models for MEO
2. ✅ Train 1 LSTM model for MEO
3. ✅ Train 36 LightGBM models for GEO
4. ✅ Train 1 LSTM model for GEO
5. ✅ Save all models, metrics, and plots

### Training Time Estimates
- **LightGBM:** ~2-5 minutes per satellite (all horizons)
- **LSTM:** ~10-20 minutes per satellite (50 epochs)
- **Total:** ~25-50 minutes for both satellites

## Data Split Strategy

**Time-based split** (no shuffling):
- Training: First 90% of temporal data
- Validation: Last 10% of temporal data

This preserves temporal ordering and tests the model's ability to forecast into the future.

## Metrics Tracked

### For Each Model:
- **RMSE** (Root Mean Squared Error) - penalizes large errors
- **MAE** (Mean Absolute Error) - average magnitude of errors
- **Training time** - computational efficiency
- **Number of estimators** (LightGBM) - model complexity

### Example Metrics JSON:
```json
{
  "x_error (m)": {
    "15min": {
      "train_rmse": 0.123456,
      "val_rmse": 0.145678,
      "train_mae": 0.098765,
      "val_mae": 0.112345,
      "training_time": 2.34,
      "n_estimators": 234
    },
    ...
  }
}
```

## Model Loading (For Inference)

### Load LightGBM Model:
```python
import lightgbm as lgb

model = lgb.Booster(model_file='models/lightgbm/meo/x_error_m_15min.txt')
predictions = model.predict(X_test)
```

### Load LSTM Model:
```python
import torch
from train_models import LSTMEncoderDecoder

checkpoint = torch.load('models/lstm/meo_model.pth')
config = checkpoint['model_config']

model = LSTMEncoderDecoder(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predictions
with torch.no_grad():
    predictions = model(X_test_tensor)
```

## Key Features

### 1. Time-Based Splitting
✅ No data leakage - validation is strictly future data

### 2. Early Stopping
✅ Prevents overfitting - stops when validation performance degrades

### 3. Model Checkpointing
✅ Saves best model based on validation loss

### 4. Comprehensive Logging
✅ Tracks all metrics and training progress

### 5. GPU Support
✅ Automatically uses CUDA if available for LSTM training

## Hyperparameter Tuning

To modify hyperparameters, edit the configuration section in `train_models.py`:

```python
# LightGBM
LGBM_PARAMS = {
    'learning_rate': 0.05,      # Lower = slower but more accurate
    'n_estimators': 1500,       # More trees = more complex model
    'max_depth': -1,            # Tree depth (-1 = no limit)
    'subsample': 0.8,           # Row sampling ratio
    'colsample_bytree': 0.8     # Feature sampling ratio
}

# LSTM
LSTM_PARAMS = {
    'hidden_size': 64,          # LSTM hidden units
    'num_layers': 2,            # Number of LSTM layers
    'dropout': 0.2,             # Dropout rate
    'learning_rate': 1e-3,      # Adam optimizer learning rate
    'epochs': 50,               # Training epochs
    'batch_size': 32,           # Batch size
    'lookback': 48              # Historical window size
}
```

## Performance Tips

### For Faster Training:
- Reduce `n_estimators` for LightGBM (e.g., 500-1000)
- Reduce `epochs` for LSTM (e.g., 20-30)
- Increase `batch_size` (if memory allows)

### For Better Accuracy:
- Increase `n_estimators` (e.g., 2000-3000)
- Increase `hidden_size` for LSTM (e.g., 128, 256)
- Add more LSTM layers
- Increase `lookback` window

## Output Files

### Models:
- **LightGBM:** `.txt` format (can be loaded with lightgbm)
- **LSTM:** `.pth` format (PyTorch checkpoint)

### Metrics:
- **JSON files:** Easy to parse and analyze
- **Training plots:** Visual inspection of learning curves

### Plots:
- **Loss curves:** Track training/validation loss over epochs
- **PNG format:** High resolution (150 DPI)

## Troubleshooting

### Out of Memory (LSTM):
- Reduce `batch_size` (e.g., 16 or 8)
- Reduce `hidden_size` or `num_layers`
- Use CPU instead of GPU

### Slow Training:
- Reduce `n_estimators` (LightGBM)
- Reduce `epochs` (LSTM)
- Use GPU for LSTM

### Poor Validation Performance:
- Check for data leakage in features
- Increase model complexity
- Try different hyperparameters
- Add more training data

## Next Steps

After training:
1. **Evaluate** models on test set
2. **Compare** LightGBM vs LSTM performance
3. **Analyze** feature importance (LightGBM)
4. **Deploy** best models to production
5. **Monitor** real-time predictions

## References

- **LightGBM:** https://lightgbm.readthedocs.io/
- **PyTorch:** https://pytorch.org/docs/
- **Time Series Forecasting:** Best practices for temporal data

# GNSS Day 8 Multi-Horizon Prediction

## Overview
This script generates comprehensive multi-horizon forecasts for the 8th day using trained LightGBM and LSTM models. It supports ensemble predictions combining both model types for improved accuracy.

## Forecast Horizons
Predicts future values at 9 different time horizons:

| Horizon | Steps | Minutes Ahead | Time Ahead |
|---------|-------|---------------|------------|
| +1      | 1     | 15            | 15 minutes |
| +2      | 2     | 30            | 30 minutes |
| +3      | 3     | 45            | 45 minutes |
| +4      | 4     | 60            | 1 hour     |
| +8      | 8     | 120           | 2 hours    |
| +12     | 12    | 180           | 3 hours    |
| +24     | 24    | 360           | 6 hours    |
| +48     | 48    | 720           | 12 hours   |
| +96     | 96    | 1440          | 24 hours   |

## Prediction Strategy

### 1. LightGBM (Direct Method)
- Uses **last timestep** as input features
- Separate model for each horizon
- Fast inference
- Good for short-term predictions

### 2. LSTM (Sequence-to-Sequence)
- Uses **last 48 timesteps** (12 hours) as input sequence
- Single model predicts all horizons at once
- Captures temporal patterns
- Good for long-term predictions

### 3. Ensemble Method
- Combines LightGBM and LSTM predictions
- Default weights: **60% LightGBM + 40% LSTM**
- Balances short-term accuracy with long-term stability

## Directory Structure
```
.
├── predict_day8.py              # Main prediction script
├── data/
│   └── processed/               # Input: Cleaned data
│       ├── MEO_clean_15min.csv
│       └── GEO_clean_15min.csv
├── models/
│   ├── lightgbm/                # Input: Trained LightGBM models
│   │   ├── meo/
│   │   └── geo/
│   └── lstm/                    # Input: Trained LSTM models
│       ├── meo_model.pth
│       └── geo_model.pth
└── predictions/                 # Output: Predictions
    ├── MEO_Day8_Predictions.csv
    ├── MEO_Day8_Predictions.json
    ├── GEO_Day8_Predictions.csv
    ├── GEO_Day8_Predictions.json
    └── plots/
        ├── MEO_Day8_Predictions.png
        └── GEO_Day8_Predictions.png
```

## Usage

### Prerequisites
Ensure you have:
1. ✅ Cleaned data in `data/processed/`
2. ✅ Trained models in `models/lightgbm/` and `models/lstm/`

### Run Prediction
```bash
python predict_day8.py
```

The script will:
1. ✅ Load cleaned MEO and GEO data
2. ✅ Extract last 48 timesteps (lookback window)
3. ✅ Load all trained models
4. ✅ Generate LightGBM predictions
5. ✅ Generate LSTM predictions
6. ✅ Ensemble predictions
7. ✅ Save CSV, JSON, and plots

### Execution Time
- **Total runtime:** ~10-30 seconds
- Model loading: ~5 seconds
- Prediction: ~5-10 seconds
- Plotting: ~5 seconds

## Output Files

### 1. CSV Files
**Columns:**
- `horizon_label` - Human-readable horizon (e.g., "15min", "1h")
- `horizon_minutes` - Minutes ahead (15, 30, 45, ...)
- `timestamp_predicted` - Predicted timestamp
- `x_error_pred` - Predicted X error (meters)
- `y_error_pred` - Predicted Y error (meters)
- `z_error_pred` - Predicted Z error (meters)
- `satclockerror_pred` - Predicted satellite clock error (meters)

**Example:**
```csv
horizon_label,horizon_minutes,timestamp_predicted,x_error_pred,y_error_pred,z_error_pred,satclockerror_pred
15min,15,2025-09-09 11:45:00,-0.234,0.156,-0.089,0.012
30min,30,2025-09-09 12:00:00,-0.245,0.167,-0.098,0.015
...
```

### 2. JSON Files
Contains:
- Satellite type
- Prediction count
- All horizon labels
- Complete predictions in structured format

### 3. Plot Files
- **4-panel visualization** showing predictions for all error components
- X-axis: Forecast horizon (log scale)
- Y-axis: Predicted error (meters)
- Saved as high-resolution PNG (150 DPI)

## Configuration

### Modify Ensemble Weights
Edit in `predict_day8.py`:
```python
ENSEMBLE_WEIGHT_LGBM = 0.6  # LightGBM weight
ENSEMBLE_WEIGHT_LSTM = 0.4  # LSTM weight
```

**Recommendations:**
- **Short-term focus:** Increase LightGBM weight (0.7-0.8)
- **Long-term focus:** Increase LSTM weight (0.5-0.6)
- **Balanced:** Keep default (0.6/0.4)

### Modify Lookback Window
```python
LOOKBACK = 48  # Number of timesteps (default: 12 hours)
```

**Recommendations:**
- Minimum: 24 timesteps (6 hours)
- Default: 48 timesteps (12 hours)
- Maximum: 96 timesteps (24 hours)

## Pipeline Details

### Step 1: Data Loading
```python
df = load_cleaned_data("data/processed/MEO_clean_15min.csv")
```
- Loads cleaned 15-minute interval data
- Normalizes column names
- Sets datetime index

### Step 2: Input Window Extraction
```python
window, last_timestamp = prepare_input_window(df, lookback=48)
```
- Extracts last 48 timesteps
- Records last known timestamp for forecast calculation

### Step 3: Model Loading
```python
lgbm_models = load_lightgbm_models("MEO")  # 36 models
lstm_model = load_lstm_model("MEO")         # 1 model
```
- Loads all trained models
- Validates model files exist

### Step 4: Feature Preparation
```python
X_lgbm = prepare_lightgbm_input(window)  # (1, 4)
X_lstm = prepare_lstm_input(window)      # (1, 48, 4)
```
- LightGBM: Uses last row only
- LSTM: Uses full sequence

### Step 5: Prediction Generation
```python
lgbm_preds = predict_lightgbm(lgbm_models, X_lgbm, "MEO")
lstm_preds = predict_lstm(lstm_model, X_lstm, "MEO")
```
- Generates predictions for all 9 horizons
- Returns structured dictionaries

### Step 6: Ensemble
```python
final_preds = ensemble_predictions(lgbm_preds, lstm_preds, 0.6, 0.4)
```
- Weighted average of both models
- Falls back to single model if one unavailable

### Step 7: Output Generation
```python
pred_df = build_prediction_table(final_preds, last_timestamp, "MEO")
save_predictions(pred_df, "MEO")
plot_predictions(pred_df, "MEO")
```
- Creates structured DataFrame
- Saves CSV and JSON
- Generates visualization

## Error Handling

The script includes comprehensive error handling:

### Missing Models
- ⚠ Warns if models not found
- ✅ Continues with available models
- ✅ Falls back to single model type if needed

### Insufficient Data
- ⚠ Warns if data shorter than lookback
- ✅ Uses all available data
- ✅ Adjusts window size automatically

### Prediction Failures
- ✅ Catches exceptions per model
- ✅ Logs errors without stopping pipeline
- ✅ Returns NaN for failed predictions

## Interpreting Results

### Prediction Confidence
- **Short horizons (15min-1h):** High confidence
- **Medium horizons (2h-6h):** Moderate confidence
- **Long horizons (12h-24h):** Lower confidence, trend indication

### Expected Accuracy
Based on typical GNSS error patterns:
- **15-30 min:** RMSE ~0.1-0.3 meters
- **1-3 hours:** RMSE ~0.3-0.6 meters
- **6-12 hours:** RMSE ~0.6-1.0 meters
- **24 hours:** RMSE ~1.0-2.0 meters

### Using Predictions
1. **Real-time correction:** Use short-term predictions (15min-1h)
2. **Planning:** Use medium-term predictions (2h-6h)
3. **Trend analysis:** Use long-term predictions (12h-24h)

## Troubleshooting

### No Predictions Generated
**Cause:** Models not trained yet  
**Solution:** Run `python train_models.py` first

### Missing Horizons
**Cause:** Some model files missing  
**Solution:** Check `models/lightgbm/` directory, retrain if needed

### Poor Prediction Quality
**Cause:** Insufficient training data or poor model performance  
**Solution:** 
- Increase training data
- Retrain with different hyperparameters
- Check data quality

### Memory Issues
**Cause:** Large lookback window  
**Solution:** Reduce `LOOKBACK` to 24 or 32

## Advanced Usage

### Predict for Specific Satellite Only
```python
from predict_day8 import predict_for_satellite

# Predict only for MEO
meo_preds = predict_for_satellite("MEO")
```

### Custom Ensemble Weights
```python
from predict_day8 import ensemble_predictions

final = ensemble_predictions(lgbm_preds, lstm_preds, 
                              weight_lgbm=0.7, 
                              weight_lstm=0.3)
```

### Access Raw Predictions
```python
# LightGBM only
lgbm_preds = predict_lightgbm(models, X, "MEO")

# LSTM only  
lstm_preds = predict_lstm(model, X_tensor, "MEO")
```

## Integration with Downstream Systems

### Load Predictions Programmatically
```python
import pandas as pd

# Load predictions
meo_preds = pd.read_csv("predictions/MEO_Day8_Predictions.csv")

# Access specific horizon
horizon_1h = meo_preds[meo_preds['horizon_label'] == '1h']
x_error_1h = horizon_1h['x_error_pred'].values[0]
```

### Real-time Streaming
For operational deployment:
1. Run prediction script on schedule (e.g., every 15 minutes)
2. Update with latest data window
3. Stream predictions to monitoring system
4. Compare predictions with actual values for model monitoring

## Next Steps

After generating predictions:
1. **Validate** predictions against actual Day 8 data (if available)
2. **Analyze** prediction errors by horizon
3. **Tune** ensemble weights based on validation results
4. **Deploy** to production for real-time forecasting
5. **Monitor** prediction accuracy over time

## References

- **Ensemble Methods:** Combining multiple models for improved accuracy
- **Time Series Forecasting:** Multi-horizon prediction strategies
- **GNSS Error Modeling:** Understanding satellite positioning errors

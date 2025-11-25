# GNSS Multi-Horizon Feature Engineering

## Overview
This pipeline transforms cleaned GNSS ephemeris time-series data into supervised learning datasets with comprehensive features for multi-horizon forecasting (15 minutes to 24 hours).

## Directory Structure
```
.
├── feature_engineering.py     # Main feature engineering script
├── data/
│   ├── processed/             # Input: Cleaned datasets
│   │   ├── MEO_clean_15min.csv
│   │   └── GEO_clean_15min.csv
│   └── features/              # Output: Feature datasets
│       ├── MEO_features.csv
│       └── GEO_features.csv
```

## Usage

```bash
python feature_engineering.py
```

## Feature Categories

### 1️⃣ Lag Features
Historical values at different time steps for each error column:
- **Lags**: t-1, t-2, t-4, t-8, t-12, t-16, t-24, t-48
- **Example**: `x_error_lag_1`, `x_error_lag_2`, etc.
- **Total per column**: 8 lag features
- **Purpose**: Capture short-term and long-term dependencies

### 2️⃣ Rolling Window Features
Statistical aggregations over moving windows for each error column:
- **Rolling Mean**: 3, 6, 12 steps
- **Rolling Std**: 3, 6, 12 steps
- **Rolling Min**: 3, 6, 12 steps
- **Rolling Max**: 3, 6, 12 steps
- **Example**: `x_error_rolling_mean_3`, `x_error_rolling_std_6`
- **Total per column**: 12 rolling features
- **Purpose**: Capture local trends and volatility

### 3️⃣ Trend Features
Rate of change indicators for each error column:
- **First Derivative** (diff1): Rate of change
- **Second Derivative** (diff2): Acceleration/deceleration
- **Example**: `x_error_diff1`, `x_error_diff2`
- **Total per column**: 2 trend features
- **Purpose**: Detect increasing/decreasing patterns

### 4️⃣ Time Features
Temporal context from datetime index:
- **hour**: Hour of day (0-23)
- **hour_sin**: Sin-transformed hour (cyclical)
- **hour_cos**: Cos-transformed hour (cyclical)
- **day_of_week**: Day of week (0=Monday, 6=Sunday)
- **day_index**: Sequential day number from start
- **Total**: 5 time features
- **Purpose**: Capture daily and weekly patterns

### 5️⃣ Multi-Horizon Targets
Future values at different forecast horizons for each error column:

| Horizon | Steps | Column Example |
|---------|-------|----------------|
| 15 min  | 1     | `x_error_t+1`  |
| 30 min  | 2     | `x_error_t+2`  |
| 45 min  | 3     | `x_error_t+3`  |
| 1 hour  | 4     | `x_error_t+4`  |
| 2 hours | 8     | `x_error_t+8`  |
| 3 hours | 12    | `x_error_t+12` |
| 6 hours | 24    | `x_error_t+24` |
| 12 hours| 48    | `x_error_t+48` |
| 24 hours| 96    | `x_error_t+96` |

- **Total per column**: 9 target horizons
- **Purpose**: Enable multi-step-ahead forecasting

## Feature Count Summary

For **4 error columns** (x_error, y_error, z_error, satclockerror):
- **Original columns**: 4
- **Lag features**: 4 × 8 = 32
- **Rolling features**: 4 × 12 = 48
- **Trend features**: 4 × 2 = 8
- **Time features**: 5
- **Target columns**: 4 × 9 = 36

**Total columns**: 4 + 32 + 48 + 8 + 5 + 36 = **133 columns**

## Pipeline Steps

1. **Load Data**: Load cleaned CSV files with datetime index
2. **Add Lag Features**: Create historical lookback values
3. **Add Rolling Features**: Compute window statistics
4. **Add Trend Features**: Calculate derivatives
5. **Add Time Features**: Extract temporal information
6. **Add Target Horizons**: Create future values for forecasting
7. **Drop NaN**: Remove rows with missing values (from edge effects)
8. **Save**: Export to `data/features/`

## NaN Handling

After feature creation:
- **Lag features** create NaN at the beginning (first 48 rows)
- **Target horizons** create NaN at the end (last 96 rows)
- **Total rows dropped**: ~144 rows per dataset
- This is expected and necessary for supervised learning

## Console Output

```
============================================================
GNSS MULTI-HORIZON FEATURE ENGINEERING PIPELINE
============================================================
✓ Directories ensured: data/features

→ Loading: data/processed/MEO_clean_15min.csv
  ✓ Loaded 12000 rows with 4 columns

############################################################
# FEATURE ENGINEERING: MEO
############################################################
...

============================================================
FEATURE ENGINEERING SUMMARY
============================================================

📊 MEO DATASET:
  • Original shape: (12000, 4)
  • Final shape: (11856, 133)
  • Lag features: 32
  • Rolling features: 48
  • Trend features: 8
  • Time features: 5
  • Total input features: 93
  • Target horizons: 36
  • Rows dropped (NaN): 144

📊 GEO DATASET:
  • Original shape: (10000, 4)
  • Final shape: (9856, 133)
  • Lag features: 32
  • Rolling features: 48
  • Trend features: 8
  • Time features: 5
  • Total input features: 93
  • Target horizons: 36
  • Rows dropped (NaN): 144

============================================================
FORECAST HORIZONS:
============================================================
  • 15min  → t+1
  • 30min  → t+2
  • 45min  → t+3
  • 1h     → t+4
  • 2h     → t+8
  • 3h     → t+12
  • 6h     → t+24
  • 12h    → t+48
  • 24h    → t+96

============================================================
✓ FEATURE ENGINEERING COMPLETED SUCCESSFULLY
============================================================
```

## Configuration

Modify these parameters in `feature_engineering.py`:

```python
LAG_STEPS = [1, 2, 4, 8, 12, 16, 24, 48]  # Lag intervals
ROLLING_WINDOWS = [3, 6, 12]               # Window sizes
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]  # Forecast steps
```

## Next Steps

After feature engineering, you can:
1. Split data into train/validation/test sets
2. Train multi-output regression models
3. Use separate models for each horizon
4. Implement sequence-to-sequence models
5. Apply attention mechanisms for long-horizon forecasting

## Notes

- Features are created using only **past information** (no data leakage)
- Time features use **cyclical encoding** for hour (sine/cosine)
- Rolling windows use `min_periods=1` to avoid early NaN
- All NaN rows are dropped at the end to ensure clean training data
- The index remains as datetime for easy time-based splitting

"""
GNSS Multi-Horizon Feature Engineering Pipeline
================================================
This script transforms cleaned GNSS ephemeris time-series data into
supervised learning datasets with lag features, rolling statistics,
trend features, time-based features, and multi-horizon targets.

Input: Cleaned CSV files (MEO_clean_15min.csv, GEO_clean_15min.csv)
Output: Feature-engineered datasets ready for forecasting models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = Path("data/processed")
FEATURES_DATA_DIR = Path("data/features")

MEO_FILE = "MEO_clean_15min.csv"
GEO_FILE = "GEO_clean_15min.csv"

# Column names for error metrics
ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]

# Lag steps (15-minute intervals)
LAG_STEPS = [1, 2, 4, 8, 12, 16, 24, 48]

# Rolling window sizes (number of steps)
ROLLING_WINDOWS = [3, 6, 12]

# Forecast horizons in 15-minute steps
# 15min, 30min, 45min, 1h, 2h, 3h, 6h, 12h, 24h
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured: {FEATURES_DATA_DIR}")


def print_dataset_info(df, dataset_name, stage):
    """
    Print diagnostic information about the dataset.
    
    Args:
        df: pandas DataFrame
        dataset_name: Name of the dataset (e.g., 'MEO', 'GEO')
        stage: Processing stage (e.g., 'loaded', 'features')
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset - {stage.upper()}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    if df.index.name == 'utc_time' or isinstance(df.index, pd.DatetimeIndex):
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Duration: {df.index.max() - df.index.min()}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"{'='*60}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cleaned_data(file_path):
    """
    Load cleaned CSV file and prepare for feature engineering.
    
    Args:
        file_path: Path to the cleaned CSV file
        
    Returns:
        pandas DataFrame with datetime index
    """
    try:
        print(f"\n→ Loading: {file_path}")
        df = pd.read_csv(file_path)
        
        # Normalize column names (remove extra spaces)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Convert to datetime and set as index
        if 'utc_time' in df.columns:
            df['utc_time'] = pd.to_datetime(df['utc_time'])
            df = df.set_index('utc_time')
        else:
            # If index is already datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'utc_time'
        
        # Sort by time
        df = df.sort_index()
        
        print(f"  ✓ Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        raise


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def add_lag_features(df, column_name, lags):
    """
    Add lag features for a given column.
    
    Creates features like: col_lag_1, col_lag_2, col_lag_4, etc.
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to create lags for
        lags: List of lag steps
        
    Returns:
        DataFrame with added lag features
    """
    df_new = df.copy()
    features_added = 0
    
    for lag in lags:
        feature_name = f"{column_name}_lag_{lag}"
        df_new[feature_name] = df_new[column_name].shift(lag)
        features_added += 1
    
    return df_new, features_added


def add_rolling_features(df, column_name, windows):
    """
    Add rolling window statistics for a given column.
    
    Creates features for:
    - Rolling mean
    - Rolling std
    - Rolling min
    - Rolling max
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to create rolling features for
        windows: List of window sizes
        
    Returns:
        DataFrame with added rolling features, count of features added
    """
    df_new = df.copy()
    features_added = 0
    
    for window in windows:
        # Rolling mean
        df_new[f"{column_name}_rolling_mean_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).mean()
        
        # Rolling std
        df_new[f"{column_name}_rolling_std_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).std()
        
        # Rolling min
        df_new[f"{column_name}_rolling_min_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).min()
        
        # Rolling max
        df_new[f"{column_name}_rolling_max_{window}"] = \
            df_new[column_name].rolling(window=window, min_periods=1).max()
        
        features_added += 4
    
    return df_new, features_added


def add_trend_features(df, column_name):
    """
    Add trend features (derivatives) for a given column.
    
    Creates:
    - First derivative (difference): rate of change
    - Second derivative (acceleration): rate of rate of change
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to create trend features for
        
    Returns:
        DataFrame with added trend features, count of features added
    """
    df_new = df.copy()
    
    # First derivative (difference)
    df_new[f"{column_name}_diff1"] = df_new[column_name].diff()
    
    # Second derivative (acceleration)
    df_new[f"{column_name}_diff2"] = df_new[f"{column_name}_diff1"].diff()
    
    return df_new, 2


def add_time_features(df):
    """
    Add time-based features from the datetime index.
    
    Creates:
    - Hour of day (0-23)
    - Sin-transformed hour (cyclical)
    - Cos-transformed hour (cyclical)
    - Day of week (0-6)
    - Day index (sequential day number)
    
    Args:
        df: pandas DataFrame with datetime index
        
    Returns:
        DataFrame with added time features, count of features added
    """
    df_new = df.copy()
    
    # Hour of day
    df_new['hour'] = df_new.index.hour
    
    # Cyclical encoding of hour (24-hour cycle)
    df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
    df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
    
    # Day of week (0=Monday, 6=Sunday)
    df_new['day_of_week'] = df_new.index.dayofweek
    
    # Day index (sequential day number from start)
    df_new['day_index'] = (df_new.index.date - df_new.index.date.min()).astype('timedelta64[D]').astype(int)
    
    return df_new, 5


def add_horizon_targets(df, column_name, horizons):
    """
    Add multi-horizon target columns for forecasting.
    
    Creates future values at different time horizons:
    col_t+1, col_t+2, col_t+3, ..., col_t+96
    
    Args:
        df: pandas DataFrame
        column_name: Name of the column to create targets for
        horizons: List of forecast horizon steps
        
    Returns:
        DataFrame with added target columns, count of targets added
    """
    df_new = df.copy()
    targets_added = 0
    
    for horizon in horizons:
        target_name = f"{column_name}_t+{horizon}"
        df_new[target_name] = df_new[column_name].shift(-horizon)
        targets_added += 1
    
    return df_new, targets_added


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_feature_dataset(df, dataset_name):
    """
    Build complete feature dataset with all engineering steps.
    
    Pipeline:
    1. Add lag features
    2. Add rolling window features
    3. Add trend features
    4. Add time features
    5. Add multi-horizon targets
    6. Drop NaN values
    
    Args:
        df: pandas DataFrame (cleaned data)
        dataset_name: Name of the dataset (e.g., 'MEO', 'GEO')
        
    Returns:
        Feature-engineered DataFrame, statistics dictionary
    """
    print(f"\n{'#'*60}")
    print(f"# FEATURE ENGINEERING: {dataset_name}")
    print(f"{'#'*60}")
    
    stats = {
        'original_shape': df.shape,
        'lag_features': 0,
        'rolling_features': 0,
        'trend_features': 0,
        'time_features': 0,
        'target_horizons': 0
    }
    
    # Print initial info
    print_dataset_info(df, dataset_name, "ORIGINAL")
    
    df_features = df.copy()
    
    # Step 1: Add lag features for each error column
    print(f"\n→ Adding lag features...")
    for col in ERROR_COLUMNS:
        if col in df_features.columns:
            df_features, n_lags = add_lag_features(df_features, col, LAG_STEPS)
            stats['lag_features'] += n_lags
    print(f"  ✓ Added {stats['lag_features']} lag features")
    
    # Step 2: Add rolling window features
    print(f"\n→ Adding rolling window features...")
    for col in ERROR_COLUMNS:
        if col in df_features.columns:
            df_features, n_rolling = add_rolling_features(df_features, col, ROLLING_WINDOWS)
            stats['rolling_features'] += n_rolling
    print(f"  ✓ Added {stats['rolling_features']} rolling window features")
    
    # Step 3: Add trend features
    print(f"\n→ Adding trend features...")
    for col in ERROR_COLUMNS:
        if col in df_features.columns:
            df_features, n_trend = add_trend_features(df_features, col)
            stats['trend_features'] += n_trend
    print(f"  ✓ Added {stats['trend_features']} trend features")
    
    # Step 4: Add time features
    print(f"\n→ Adding time features...")
    df_features, n_time = add_time_features(df_features)
    stats['time_features'] = n_time
    print(f"  ✓ Added {stats['time_features']} time features")
    
    # Step 5: Add multi-horizon targets
    print(f"\n→ Adding multi-horizon targets...")
    for col in ERROR_COLUMNS:
        if col in df_features.columns:
            df_features, n_targets = add_horizon_targets(df_features, col, FORECAST_HORIZONS)
            stats['target_horizons'] += n_targets
    print(f"  ✓ Added {stats['target_horizons']} target horizon columns")
    
    # Step 6: Drop NaN values
    print(f"\n→ Dropping rows with NaN values...")
    rows_before = len(df_features)
    df_features = df_features.dropna()
    rows_after = len(df_features)
    stats['rows_dropped'] = rows_before - rows_after
    print(f"  ✓ Dropped {stats['rows_dropped']} rows with NaN values")
    
    stats['final_shape'] = df_features.shape
    stats['total_features'] = (stats['lag_features'] + 
                                stats['rolling_features'] + 
                                stats['trend_features'] + 
                                stats['time_features'])
    
    # Print final info
    print_dataset_info(df_features, dataset_name, "FEATURES")
    
    return df_features, stats


def save_feature_dataset(df, file_name):
    """
    Save feature-engineered dataset to the features directory.
    
    Args:
        df: pandas DataFrame
        file_name: Output file name
    """
    output_path = FEATURES_DATA_DIR / file_name
    df.to_csv(output_path)
    print(f"\n✓ Saved feature dataset to: {output_path}")


def print_summary(meo_stats, geo_stats):
    """
    Print final summary of the feature engineering process.
    
    Args:
        meo_stats: Statistics dictionary for MEO dataset
        geo_stats: Statistics dictionary for GEO dataset
    """
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'='*60}")
    
    print("\n📊 MEO DATASET:")
    print(f"  • Original shape: {meo_stats['original_shape']}")
    print(f"  • Final shape: {meo_stats['final_shape']}")
    print(f"  • Lag features: {meo_stats['lag_features']}")
    print(f"  • Rolling features: {meo_stats['rolling_features']}")
    print(f"  • Trend features: {meo_stats['trend_features']}")
    print(f"  • Time features: {meo_stats['time_features']}")
    print(f"  • Total input features: {meo_stats['total_features']}")
    print(f"  • Target horizons: {meo_stats['target_horizons']}")
    print(f"  • Rows dropped (NaN): {meo_stats['rows_dropped']}")
    
    print("\n📊 GEO DATASET:")
    print(f"  • Original shape: {geo_stats['original_shape']}")
    print(f"  • Final shape: {geo_stats['final_shape']}")
    print(f"  • Lag features: {geo_stats['lag_features']}")
    print(f"  • Rolling features: {geo_stats['rolling_features']}")
    print(f"  • Trend features: {geo_stats['trend_features']}")
    print(f"  • Time features: {geo_stats['time_features']}")
    print(f"  • Total input features: {geo_stats['total_features']}")
    print(f"  • Target horizons: {geo_stats['target_horizons']}")
    print(f"  • Rows dropped (NaN): {geo_stats['rows_dropped']}")
    
    print(f"\n{'='*60}")
    print("FORECAST HORIZONS:")
    print(f"{'='*60}")
    horizon_labels = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
    for label, steps in zip(horizon_labels, FORECAST_HORIZONS):
        print(f"  • {label:6s} → t+{steps}")
    
    print(f"\n{'='*60}")
    print("✓ FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    try:
        print("\n" + "="*60)
        print("GNSS MULTI-HORIZON FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Ensure directories exist
        ensure_directories()
        
        # Process MEO dataset
        meo_path = PROCESSED_DATA_DIR / MEO_FILE
        meo_df = load_cleaned_data(meo_path)
        meo_features, meo_stats = build_feature_dataset(meo_df, "MEO")
        save_feature_dataset(meo_features, "MEO_features.csv")
        
        # Process GEO dataset
        geo_path = PROCESSED_DATA_DIR / GEO_FILE
        geo_df = load_cleaned_data(geo_path)
        geo_features, geo_stats = build_feature_dataset(geo_df, "GEO")
        save_feature_dataset(geo_features, "GEO_features.csv")
        
        # Print final summary
        print_summary(meo_stats, geo_stats)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

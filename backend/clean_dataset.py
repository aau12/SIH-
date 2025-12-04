"""
GNSS Ephemeris Data Cleaning Pipeline
======================================
This script performs comprehensive data cleaning for GNSS satellite ephemeris
and clock error data, preparing it for multi-horizon time-series forecasting
with 15-minute intervals.

Input: Raw CSV files with columns: utc_time, x_error, y_error, z_error, satclockerror
Output: Cleaned, resampled datasets ready for modeling (no scaling)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
SCALERS_DIR = Path("models/scalers")

MEO_FILES = ["DATA_MEO_Train.csv", "DATA_MEO_Train2.csv"]
GEO_FILES = ["DATA_GEO_Train.csv"]

RESAMPLE_FREQ = "15T"  # 15-minute intervals
OUTLIER_THRESHOLD = 3  # Z-score threshold
SMOOTHING_WINDOW = 3   # Rolling median window size

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured: {PROCESSED_DATA_DIR}, {SCALERS_DIR}")


def print_dataset_info(df, dataset_name, stage):
    """
    Print diagnostic information about the dataset.
    
    Args:
        df: pandas DataFrame
        dataset_name: Name of the dataset (e.g., 'MEO', 'GEO')
        stage: Processing stage (e.g., 'raw', 'cleaned')
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Dataset - {stage.upper()}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    print(f"Duration: {df.index.max() - df.index.min()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"{'='*60}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_and_prepare(file_path):
    """
    Load a CSV file and prepare it for processing.
    
    Steps:
    - Load CSV
    - Convert utc_time to datetime
    - Sort by time
    - Set datetime as index
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with datetime index
    """
    try:
        print(f"\n→ Loading: {file_path}")
        df = pd.read_csv(file_path)
        
        # Normalize column names (remove extra spaces)
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # Convert to datetime
        df["utc_time"] = pd.to_datetime(df["utc_time"])
        
        # Sort by time
        df = df.sort_values("utc_time")
        
        # Set as index
        df = df.set_index("utc_time")
        
        print(f"  ✓ Loaded {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")
        raise


def merge_meo_datasets(file_list):
    """
    Load and merge multiple MEO training files into one continuous time-series.
    
    Args:
        file_list: List of file names to merge
        
    Returns:
        Merged and deduplicated pandas DataFrame
    """
    print(f"\n{'='*60}")
    print("MERGING MEO DATASETS")
    print(f"{'='*60}")
    
    dfs = []
    for file_name in file_list:
        file_path = RAW_DATA_DIR / file_name
        df = load_and_prepare(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, axis=0)
    
    # Remove duplicate timestamps (keep first occurrence)
    original_len = len(merged_df)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    duplicates_removed = original_len - len(merged_df)
    
    # Sort by index
    merged_df = merged_df.sort_index()
    
    print(f"\n→ Merged {len(file_list)} files")
    print(f"  ✓ Total rows: {len(merged_df)}")
    print(f"  ✓ Duplicates removed: {duplicates_removed}")
    
    return merged_df


def load_geo_dataset(file_name):
    """
    Load the GEO training file.
    
    Args:
        file_name: Name of the GEO file
        
    Returns:
        pandas DataFrame with datetime index
    """
    print(f"\n{'='*60}")
    print("LOADING GEO DATASET")
    print(f"{'='*60}")
    
    file_path = RAW_DATA_DIR / file_name
    df = load_and_prepare(file_path)
    
    return df


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def resample_to_15min(df):
    """
    Resample data to 15-minute intervals and interpolate missing values.
    
    Steps:
    - Resample to 15-minute frequency using mean aggregation
    - Fill missing rows using time-based linear interpolation
    
    Args:
        df: pandas DataFrame with datetime index
        
    Returns:
        Resampled DataFrame
    """
    print(f"\n→ Resampling to {RESAMPLE_FREQ} intervals...")
    
    original_len = len(df)
    
    # Resample using mean
    df_resampled = df.resample(RESAMPLE_FREQ).mean()
    
    # Count missing values before interpolation
    missing_before = df_resampled.isnull().sum().sum()
    
    # Interpolate missing values
    df_resampled = df_resampled.interpolate(method='time')
    
    # Count missing values after interpolation
    missing_after = df_resampled.isnull().sum().sum()
    
    print(f"  ✓ Original rows: {original_len}")
    print(f"  ✓ Resampled rows: {len(df_resampled)}")
    print(f"  ✓ Missing values filled: {missing_before - missing_after}")
    
    return df_resampled, missing_before - missing_after


def remove_outliers_zscore(df, threshold=3):
    """
    Remove outliers using Z-score method.
    
    For each error column:
    - Calculate Z-score: z = (value - mean) / std
    - Replace values with NaN where |z| > threshold
    - Interpolate to fill the gaps
    
    Args:
        df: pandas DataFrame
        threshold: Z-score threshold (default: 3)
        
    Returns:
        DataFrame with outliers removed, count of outliers removed
    """
    print(f"\n→ Removing outliers (Z-score threshold: {threshold})...")
    
    df_clean = df.copy()
    total_outliers = 0
    
    for col in ERROR_COLUMNS:
        if col in df_clean.columns:
            # Calculate Z-scores
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean) / std)
            
            # Identify outliers
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            total_outliers += outlier_count
            
            # Replace outliers with NaN
            df_clean.loc[outliers, col] = np.nan
            
            # Interpolate to fill gaps
            df_clean[col] = df_clean[col].interpolate(method='time')
            
            print(f"  ✓ {col}: {outlier_count} outliers removed")
    
    print(f"  ✓ Total outliers removed: {total_outliers}")
    
    return df_clean, total_outliers


def smooth_noise(df, window=3):
    """
    Apply centered rolling median for noise smoothing.
    
    Args:
        df: pandas DataFrame
        window: Rolling window size (default: 3)
        
    Returns:
        Smoothed DataFrame
    """
    print(f"\n→ Applying noise smoothing (rolling median, window={window})...")
    
    df_smooth = df.copy()
    
    for col in ERROR_COLUMNS:
        if col in df_smooth.columns:
            df_smooth[col] = df_smooth[col].rolling(
                window=window, 
                center=True, 
                min_periods=1
            ).median()
    
    print(f"  ✓ Noise smoothing applied to {len(ERROR_COLUMNS)} columns")
    
    return df_smooth


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def clean_pipeline(df, dataset_name):
    """
    Execute the complete cleaning pipeline on a dataset.
    
    Pipeline steps:
    1. Resample to 15-minute intervals
    2. Remove outliers using Z-score
    3. Smooth noise with rolling median
    
    Args:
        df: pandas DataFrame
        dataset_name: Name of the dataset ('MEO' or 'GEO')
        
    Returns:
        Cleaned DataFrame, statistics dictionary
    """
    print(f"\n{'#'*60}")
    print(f"# CLEANING PIPELINE: {dataset_name}")
    print(f"{'#'*60}")
    
    stats = {}
    
    # Print initial info
    print_dataset_info(df, dataset_name, "RAW")
    stats['original_shape'] = df.shape
    stats['original_time_range'] = (df.index.min(), df.index.max())
    
    # Step 1: Resample
    df, missing_filled = resample_to_15min(df)
    stats['missing_filled'] = missing_filled
    
    # Step 2: Remove outliers
    df, outliers_removed = remove_outliers_zscore(df, threshold=OUTLIER_THRESHOLD)
    stats['outliers_removed'] = outliers_removed
    
    # Step 3: Smooth noise
    df = smooth_noise(df, window=SMOOTHING_WINDOW)
    
    # Print final info
    print_dataset_info(df, dataset_name, "CLEANED (UNSCALED)")
    stats['final_shape'] = df.shape
    stats['final_time_range'] = (df.index.min(), df.index.max())
    
    return df, stats


def save_cleaned_data(df, file_name):
    """
    Save cleaned dataset to the processed data directory.
    
    Args:
        df: pandas DataFrame
        file_name: Output file name
    """
    output_path = PROCESSED_DATA_DIR / file_name
    df.to_csv(output_path)
    print(f"\n✓ Saved cleaned data to: {output_path}")


def print_summary(meo_stats, geo_stats):
    """
    Print final summary of the cleaning process.
    
    Args:
        meo_stats: Statistics dictionary for MEO dataset
        geo_stats: Statistics dictionary for GEO dataset
    """
    print(f"\n{'='*60}")
    print("CLEANING SUMMARY")
    print(f"{'='*60}")
    
    print("\n📊 MEO DATASET:")
    print(f"  • Time range before: {meo_stats['original_time_range'][0]} to {meo_stats['original_time_range'][1]}")
    print(f"  • Time range after:  {meo_stats['final_time_range'][0]} to {meo_stats['final_time_range'][1]}")
    print(f"  • Original shape: {meo_stats['original_shape']}")
    print(f"  • Final shape: {meo_stats['final_shape']}")
    print(f"  • Missing rows fixed: {meo_stats['missing_filled']}")
    print(f"  • Outliers removed: {meo_stats['outliers_removed']}")
    
    print("\n📊 GEO DATASET:")
    print(f"  • Time range before: {geo_stats['original_time_range'][0]} to {geo_stats['original_time_range'][1]}")
    print(f"  • Time range after:  {geo_stats['final_time_range'][0]} to {geo_stats['final_time_range'][1]}")
    print(f"  • Original shape: {geo_stats['original_shape']}")
    print(f"  • Final shape: {geo_stats['final_shape']}")
    print(f"  • Missing rows fixed: {geo_stats['missing_filled']}")
    print(f"  • Outliers removed: {geo_stats['outliers_removed']}")
    
    print(f"\n{'='*60}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
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
        print("GNSS EPHEMERIS DATA CLEANING PIPELINE")
        print("="*60)
        
        # Ensure directories exist
        ensure_directories()
        
        # Process MEO datasets
        meo_df = merge_meo_datasets(MEO_FILES)
        meo_clean, meo_stats = clean_pipeline(meo_df, "MEO")
        save_cleaned_data(meo_clean, "MEO_clean_15min.csv")
        
        # Process GEO dataset
        geo_df = load_geo_dataset(GEO_FILES[0])
        geo_clean, geo_stats = clean_pipeline(geo_df, "GEO")
        save_cleaned_data(geo_clean, "GEO_clean_15min.csv")
        
        # Print final summary
        print_summary(meo_stats, geo_stats)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()

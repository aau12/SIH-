"""
GNSS Residual Analysis Module
==============================
Complete residual analysis for Day-8 predictions including:
- Normality tests (Shapiro-Wilk)
- Distribution analysis (histograms, QQ plots)
- Autocorrelation analysis (ACF, PACF)
- Drift detection
- Comprehensive summary reports
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

# Configuration
PREDICTIONS_DIR = Path("predictions")
PROCESSED_DATA_DIR = Path("data/processed")
RESIDUALS_DIR = Path("evaluation/residuals")

ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]
ALPHA = 0.05


def ensure_directories():
    """Create necessary directories."""
    RESIDUALS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured: {RESIDUALS_DIR}")


def load_predictions_and_ground_truth(satellite_type):
    """Load predictions and ground truth data."""
    print(f"\n→ Loading data for {satellite_type}...")
    
    pred_file = PREDICTIONS_DIR / f"{satellite_type}_Day8_Predictions.csv"
    gt_file = PROCESSED_DATA_DIR / f"{satellite_type}_clean_15min.csv"
    
    predictions_df = pd.read_csv(pred_file)
    predictions_df['timestamp_predicted'] = pd.to_datetime(predictions_df['timestamp_predicted'])
    
    ground_truth_df = pd.read_csv(gt_file, index_col=0, parse_dates=True)
    ground_truth_df.columns = ground_truth_df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    print(f"  ✓ Loaded {len(predictions_df)} predictions")
    print(f"  ✓ Loaded {len(ground_truth_df)} ground truth rows")
    
    return predictions_df, ground_truth_df


def compute_residuals(predictions_df, ground_truth_df):
    """Compute residuals for all variables and horizons."""
    print(f"\n→ Computing residuals...")
    
    residuals = {}
    
    for short_name in ERROR_COLUMNS_SHORT:
        residuals[short_name] = {}
        
        for horizon_label in HORIZON_LABELS:
            horizon_data = predictions_df[predictions_df['horizon_label'] == horizon_label]
            
            if len(horizon_data) == 0:
                continue
            
            residual_list = []
            
            for _, pred_row in horizon_data.iterrows():
                pred_time = pred_row['timestamp_predicted']
                
                if pred_time in ground_truth_df.index:
                    actual_col = [col for col in ERROR_COLUMNS if short_name in col.lower()][0]
                    actual_val = ground_truth_df.loc[pred_time, actual_col]
                    pred_val = pred_row[f'{short_name}_pred']
                    
                    residual = actual_val - pred_val
                    residual_list.append(residual)
            
            if residual_list:
                residuals[short_name][horizon_label] = np.array(residual_list)
    
    print(f"  ✓ Computed residuals for {len(residuals)} variables")
    
    return residuals


def run_shapiro_tests(residuals, satellite_type):
    """Perform Shapiro-Wilk normality tests."""
    print(f"\n→ Running Shapiro-Wilk tests for {satellite_type}...")
    
    results = []
    
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
            if horizon_label in residuals[short_name]:
                res_array = residuals[short_name][horizon_label]
                
                if len(res_array) >= 3:
                    try:
                        W, p = shapiro(res_array)
                        normal = 'Yes' if p > ALPHA else 'No'
                        
                        results.append({
                            'satellite': satellite_type,
                            'variable': short_name,
                            'horizon_min': horizon_min,
                            'W': W,
                            'p': p,
                            'normal': normal
                        })
                    except:
                        pass
    
    results_df = pd.DataFrame(results)
    print(f"  ✓ Completed {len(results_df)} tests")
    
    return results_df


def plot_histograms(residuals, satellite_type):
    """Generate histogram plots for residuals."""
    print(f"\n→ Generating histograms for {satellite_type}...")
    
    count = 0
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label in HORIZON_LABELS:
            if horizon_label in residuals[short_name]:
                res_array = residuals[short_name][horizon_label]
                
                if len(res_array) < 2:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(res_array, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
                ax.set_title(f'{satellite_type} - {short_name} - {horizon_label}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Residual (m)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(RESIDUALS_DIR / f"hist_{satellite_type.lower()}_{short_name}_{horizon_label}.png", dpi=150)
                plt.close()
                count += 1
    
    print(f"  ✓ Generated {count} histograms")


def plot_qq_plots(residuals, satellite_type):
    """Generate QQ plots for residuals."""
    print(f"\n→ Generating QQ plots for {satellite_type}...")
    
    count = 0
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label in HORIZON_LABELS:
            if horizon_label in residuals[short_name]:
                res_array = residuals[short_name][horizon_label]
                
                if len(res_array) < 3:
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sm.qqplot(res_array, line='45', ax=ax)
                ax.set_title(f'{satellite_type} - {short_name} - {horizon_label}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(RESIDUALS_DIR / f"qq_{satellite_type.lower()}_{short_name}_{horizon_label}.png", dpi=150)
                plt.close()
                count += 1
    
    print(f"  ✓ Generated {count} QQ plots")


def plot_acf_pacf(residuals, satellite_type):
    """Generate ACF and PACF plots."""
    print(f"\n→ Generating ACF/PACF plots for {satellite_type}...")
    
    count = 0
    for short_name in ERROR_COLUMNS_SHORT:
        all_residuals = []
        for horizon_label in HORIZON_LABELS:
            if horizon_label in residuals[short_name]:
                all_residuals.extend(residuals[short_name][horizon_label])
        
        if len(all_residuals) < 10:
            continue
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        plot_acf(all_residuals, lags=min(20, len(all_residuals)//2), ax=axes[0])
        axes[0].set_title(f'{satellite_type} - {short_name} - ACF', fontsize=14, fontweight='bold')
        
        plot_pacf(all_residuals, lags=min(20, len(all_residuals)//2), ax=axes[1])
        axes[1].set_title(f'{satellite_type} - {short_name} - PACF', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(RESIDUALS_DIR / f"acf_pacf_{satellite_type.lower()}_{short_name}.png", dpi=150)
        plt.close()
        count += 1
    
    print(f"  ✓ Generated {count} ACF/PACF plots")


def detect_drift(residuals, satellite_type):
    """Detect drift in residuals."""
    print(f"\n→ Detecting drift for {satellite_type}...")
    
    count = 0
    for short_name in ERROR_COLUMNS_SHORT:
        all_residuals = []
        for horizon_label in HORIZON_LABELS:
            if horizon_label in residuals[short_name]:
                all_residuals.extend(residuals[short_name][horizon_label])
        
        if len(all_residuals) < 6:
            continue
        
        res_series = pd.Series(all_residuals)
        rolling_mean = res_series.rolling(window=min(6, len(all_residuals)//2)).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(res_series.values, 'o-', alpha=0.6, label='Residuals')
        ax.plot(rolling_mean.values, 'r-', linewidth=2, label='Rolling Mean')
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{satellite_type} - {short_name} - Drift Detection', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Residual (m)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESIDUALS_DIR / f"drift_{satellite_type.lower()}_{short_name}.png", dpi=150)
        plt.close()
        count += 1
    
    print(f"  ✓ Generated {count} drift plots")


def generate_residual_summary(residuals, shapiro_df, satellite_type):
    """Generate comprehensive residual summary."""
    print(f"\n→ Generating summary for {satellite_type}...")
    
    summary = []
    
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
            if horizon_label in residuals[short_name]:
                res_array = residuals[short_name][horizon_label]
                
                rmse = np.sqrt(np.mean(res_array**2))
                mae = np.mean(np.abs(res_array))
                bias = np.mean(res_array)
                std = np.std(res_array)
                
                W = np.nan
                p = np.nan
                normal = 'N/A'
                
                if len(shapiro_df) > 0 and 'variable' in shapiro_df.columns:
                    shapiro_row = shapiro_df[
                        (shapiro_df['variable'] == short_name) & 
                        (shapiro_df['horizon_min'] == horizon_min)
                    ]
                    
                    if len(shapiro_row) > 0:
                        W = shapiro_row['W'].values[0]
                        p = shapiro_row['p'].values[0]
                        normal = shapiro_row['normal'].values[0]
                
                summary.append({
                    'satellite': satellite_type,
                    'variable': short_name,
                    'horizon_minutes': horizon_min,
                    'rmse': rmse,
                    'mae': mae,
                    'bias': bias,
                    'std': std,
                    'W_shapiro': W,
                    'p_shapiro': p,
                    'normality_flag': normal
                })
    
    summary_df = pd.DataFrame(summary)
    print(f"  ✓ Generated summary with {len(summary_df)} rows")
    
    return summary_df


def save_all_outputs(shapiro_df, summary_df):
    """Save all analysis outputs."""
    print(f"\n→ Saving outputs...")
    
    shapiro_path = RESIDUALS_DIR / "shapiro_results.csv"
    shapiro_df.to_csv(shapiro_path, index=False)
    print(f"  ✓ Saved: {shapiro_path}")
    
    summary_path = RESIDUALS_DIR / "residual_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved: {summary_path}")


def analyze_residuals(satellite_type):
    """Complete residual analysis pipeline."""
    print(f"\n{'='*70}")
    print(f"RESIDUAL ANALYSIS FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    
    try:
        predictions_df, ground_truth_df = load_predictions_and_ground_truth(satellite_type)
        residuals = compute_residuals(predictions_df, ground_truth_df)
        shapiro_df = run_shapiro_tests(residuals, satellite_type)
        plot_histograms(residuals, satellite_type)
        plot_qq_plots(residuals, satellite_type)
        plot_acf_pacf(residuals, satellite_type)
        detect_drift(residuals, satellite_type)
        summary_df = generate_residual_summary(residuals, shapiro_df, satellite_type)
        
        print(f"\n{'='*70}")
        print(f"✓ {satellite_type} ANALYSIS COMPLETED")
        print(f"{'='*70}\n")
        
        return shapiro_df, summary_df
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main execution."""
    print("\n" + "="*70)
    print(" "*15 + "GNSS RESIDUAL ANALYSIS")
    print("="*70)
    
    ensure_directories()
    
    all_shapiro = []
    all_summary = []
    
    for sat_type in ["MEO", "GEO"]:
        shapiro_df, summary_df = analyze_residuals(sat_type)
        all_shapiro.append(shapiro_df)
        all_summary.append(summary_df)
    
    combined_shapiro = pd.concat(all_shapiro, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    
    save_all_outputs(combined_shapiro, combined_summary)
    
    print("\n" + "="*70)
    print("✓ ALL RESIDUAL ANALYSIS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

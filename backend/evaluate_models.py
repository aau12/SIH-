"""
GNSS Day-8 Prediction Evaluation Module
========================================
Comprehensive evaluation of multi-horizon GNSS predictions including:
- RMSE, MAE, Bias, Std metrics
- Shapiro-Wilk normality tests
- QQ plots and residual distributions
- Dashboard-ready visualizations
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

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

PREDICTIONS_DIR = Path("predictions")
PROCESSED_DATA_DIR = Path("data/processed")
EVALUATION_DIR = Path("evaluation")
PLOTS_DIR = Path("evaluation/plots")
DASHBOARD_DIR = Path("evaluation/dashboard")

# Error columns
ERROR_COLUMNS = ["x_error (m)", "y_error (m)", "z_error (m)", "satclockerror (m)"]
ERROR_COLUMNS_SHORT = ["x_error", "y_error", "z_error", "satclockerror"]

# Forecast horizons
FORECAST_HORIZONS = [1, 2, 3, 4, 8, 12, 24, 48, 96]
HORIZON_LABELS = ["15min", "30min", "45min", "1h", "2h", "3h", "6h", "12h", "24h"]
HORIZON_MINUTES = [15, 30, 45, 60, 120, 180, 360, 720, 1440]

# Significance level for normality test
ALPHA = 0.05


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories for evaluation outputs."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured: {EVALUATION_DIR}, {PLOTS_DIR}, {DASHBOARD_DIR}")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_predictions(satellite_type):
    """
    Load prediction CSV file.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        pandas DataFrame with predictions
    """
    pred_file = PREDICTIONS_DIR / f"{satellite_type}_Day8_Predictions.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    print(f"\n→ Loading predictions from: {pred_file}")
    df = pd.read_csv(pred_file)
    df['timestamp_predicted'] = pd.to_datetime(df['timestamp_predicted'])
    
    print(f"  ✓ Loaded {len(df)} predictions")
    print(f"  ✓ Horizons: {df['horizon_label'].unique().tolist()}")
    
    return df


def load_ground_truth(satellite_type):
    """
    Load ground truth (cleaned data) for Day 8.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
        
    Returns:
        pandas DataFrame with ground truth
    """
    gt_file = PROCESSED_DATA_DIR / f"{satellite_type}_clean_15min.csv"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    print(f"\n→ Loading ground truth from: {gt_file}")
    df = pd.read_csv(gt_file, index_col=0, parse_dates=True)
    
    # Normalize column names
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  ✓ Time range: {df.index.min()} to {df.index.max()}")
    
    return df


def match_predictions_with_ground_truth(predictions_df, ground_truth_df):
    """
    Match predictions with actual ground truth values.
    
    Args:
        predictions_df: DataFrame with predictions
        ground_truth_df: DataFrame with ground truth
        
    Returns:
        DataFrame with matched predictions and actuals
    """
    print(f"\n→ Matching predictions with ground truth...")
    
    matched_data = []
    
    for _, pred_row in predictions_df.iterrows():
        pred_time = pred_row['timestamp_predicted']
        horizon_label = pred_row['horizon_label']
        horizon_min = pred_row['horizon_minutes']
        
        # Find corresponding ground truth
        if pred_time in ground_truth_df.index:
            gt_row = ground_truth_df.loc[pred_time]
            
            match = {
                'horizon_label': horizon_label,
                'horizon_minutes': horizon_min,
                'timestamp': pred_time
            }
            
            # Add predictions and actuals for each error column
            for error_col, short_name in zip(ERROR_COLUMNS, ERROR_COLUMNS_SHORT):
                match[f'{short_name}_pred'] = pred_row[f'{short_name}_pred']
                match[f'{short_name}_actual'] = gt_row[error_col]
                match[f'{short_name}_residual'] = gt_row[error_col] - pred_row[f'{short_name}_pred']
            
            matched_data.append(match)
    
    matched_df = pd.DataFrame(matched_data)
    
    print(f"  ✓ Matched {len(matched_df)} predictions with ground truth")
    print(f"  ✓ Coverage: {len(matched_df)}/{len(predictions_df)} predictions")
    
    return matched_df


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_rmse(actual, predicted):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_mae(actual, predicted):
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def compute_bias(actual, predicted):
    """Compute Bias (mean error)."""
    return np.mean(predicted - actual)


def compute_std_residuals(actual, predicted):
    """Compute Standard Deviation of Residuals."""
    return np.std(actual - predicted)


def compute_metrics(matched_df):
    """
    Compute all evaluation metrics for each horizon and variable.
    
    Args:
        matched_df: DataFrame with matched predictions and actuals
        
    Returns:
        DataFrame with metrics
    """
    print(f"\n→ Computing evaluation metrics...")
    
    metrics_list = []
    
    for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
        horizon_data = matched_df[matched_df['horizon_label'] == horizon_label]
        
        if len(horizon_data) == 0:
            print(f"  ⚠ No data for horizon {horizon_label}")
            continue
        
        metrics = {
            'horizon_label': horizon_label,
            'horizon_minutes': horizon_min
        }
        
        # Compute metrics for each error column
        for short_name in ERROR_COLUMNS_SHORT:
            actual = horizon_data[f'{short_name}_actual'].values
            predicted = horizon_data[f'{short_name}_pred'].values
            
            metrics[f'rmse_{short_name}'] = compute_rmse(actual, predicted)
            metrics[f'mae_{short_name}'] = compute_mae(actual, predicted)
            metrics[f'bias_{short_name}'] = compute_bias(actual, predicted)
            metrics[f'std_{short_name}'] = compute_std_residuals(actual, predicted)
        
        metrics_list.append(metrics)
        
        print(f"  • {horizon_label:6s}: RMSE(x)={metrics['rmse_x_error']:.4f}m, "
              f"MAE(x)={metrics['mae_x_error']:.4f}m")
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"  ✓ Computed metrics for {len(metrics_df)} horizons")
    
    return metrics_df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def compute_shapiro_wilk(matched_df):
    """
    Perform Shapiro-Wilk normality test on residuals.
    
    Args:
        matched_df: DataFrame with matched predictions and actuals
        
    Returns:
        DataFrame with Shapiro-Wilk test results
    """
    print(f"\n→ Performing Shapiro-Wilk normality tests...")
    
    shapiro_results = []
    
    for horizon_label, horizon_min in zip(HORIZON_LABELS, HORIZON_MINUTES):
        horizon_data = matched_df[matched_df['horizon_label'] == horizon_label]
        
        if len(horizon_data) < 3:
            print(f"  ⚠ Insufficient data for {horizon_label} (need ≥3 samples)")
            continue
        
        result = {
            'horizon_label': horizon_label,
            'horizon_minutes': horizon_min
        }
        
        # Test each error column
        for short_name in ERROR_COLUMNS_SHORT:
            residuals = horizon_data[f'{short_name}_residual'].values
            
            # Remove NaN values
            residuals = residuals[~np.isnan(residuals)]
            
            if len(residuals) >= 3:
                try:
                    W, p_value = shapiro(residuals)
                    result[f'W_{short_name}'] = W
                    result[f'p_{short_name}'] = p_value
                    result[f'normal_{short_name}'] = 'Yes' if p_value > ALPHA else 'No'
                except Exception as e:
                    print(f"    ⚠ Shapiro test failed for {short_name} at {horizon_label}: {e}")
                    result[f'W_{short_name}'] = np.nan
                    result[f'p_{short_name}'] = np.nan
                    result[f'normal_{short_name}'] = 'N/A'
            else:
                result[f'W_{short_name}'] = np.nan
                result[f'p_{short_name}'] = np.nan
                result[f'normal_{short_name}'] = 'N/A'
        
        shapiro_results.append(result)
        
        # Print summary
        normal_count = sum([result.get(f'normal_{s}') == 'Yes' for s in ERROR_COLUMNS_SHORT])
        print(f"  • {horizon_label:6s}: {normal_count}/4 variables normally distributed")
    
    shapiro_df = pd.DataFrame(shapiro_results)
    
    print(f"  ✓ Completed normality tests for {len(shapiro_df)} horizons")
    
    return shapiro_df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_qq_plots(matched_df, satellite_type):
    """
    Generate QQ plots for residuals.
    
    Args:
        matched_df: DataFrame with matched predictions and actuals
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n→ Generating QQ plots for {satellite_type}...")
    
    plot_count = 0
    
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label in HORIZON_LABELS:
            horizon_data = matched_df[matched_df['horizon_label'] == horizon_label]
            
            if len(horizon_data) < 3:
                continue
            
            residuals = horizon_data[f'{short_name}_residual'].values
            residuals = residuals[~np.isnan(residuals)]
            
            if len(residuals) < 3:
                continue
            
            # Create QQ plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sm.qqplot(residuals, line='45', ax=ax)
            ax.set_title(f'{satellite_type} - {short_name.replace("_", " ").title()} - {horizon_label}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Theoretical Quantiles', fontsize=12)
            ax.set_ylabel('Sample Quantiles', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = PLOTS_DIR / f"qq_{satellite_type.lower()}_{short_name}_{horizon_label}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_count += 1
    
    print(f"  ✓ Generated {plot_count} QQ plots")


def generate_residual_histograms(matched_df, satellite_type):
    """
    Generate histograms of residuals.
    
    Args:
        matched_df: DataFrame with matched predictions and actuals
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n→ Generating residual histograms for {satellite_type}...")
    
    plot_count = 0
    
    for short_name in ERROR_COLUMNS_SHORT:
        for horizon_label in HORIZON_LABELS:
            horizon_data = matched_df[matched_df['horizon_label'] == horizon_label]
            
            if len(horizon_data) < 3:
                continue
            
            residuals = horizon_data[f'{short_name}_residual'].values
            residuals = residuals[~np.isnan(residuals)]
            
            if len(residuals) < 3:
                continue
            
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax.set_title(f'{satellite_type} - {short_name.replace("_", " ").title()} Residuals - {horizon_label}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Residual (m)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            ax.text(0.02, 0.98, f'Mean: {mean_res:.4f}m\nStd: {std_res:.4f}m',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save plot
            plot_path = PLOTS_DIR / f"hist_{satellite_type.lower()}_{short_name}_{horizon_label}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_count += 1
    
    print(f"  ✓ Generated {plot_count} histograms")


def generate_dashboard_plots(metrics_df, matched_df, satellite_type):
    """
    Generate summary plots for dashboard.
    
    Args:
        metrics_df: DataFrame with metrics
        matched_df: DataFrame with matched data
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n→ Generating dashboard plots for {satellite_type}...")
    
    # 1. RMSE vs Horizon
    fig, ax = plt.subplots(figsize=(12, 6))
    for short_name in ERROR_COLUMNS_SHORT:
        ax.plot(metrics_df['horizon_minutes'], metrics_df[f'rmse_{short_name}'],
               'o-', linewidth=2, markersize=8, label=short_name.replace('_', ' ').title())
    ax.set_xlabel('Forecast Horizon (minutes)', fontsize=12)
    ax.set_ylabel('RMSE (meters)', fontsize=12)
    ax.set_title(f'{satellite_type} - RMSE vs Forecast Horizon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / f"{satellite_type.lower()}_rmse_vs_horizon.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. MAE vs Horizon
    fig, ax = plt.subplots(figsize=(12, 6))
    for short_name in ERROR_COLUMNS_SHORT:
        ax.plot(metrics_df['horizon_minutes'], metrics_df[f'mae_{short_name}'],
               'o-', linewidth=2, markersize=8, label=short_name.replace('_', ' ').title())
    ax.set_xlabel('Forecast Horizon (minutes)', fontsize=12)
    ax.set_ylabel('MAE (meters)', fontsize=12)
    ax.set_title(f'{satellite_type} - MAE vs Forecast Horizon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / f"{satellite_type.lower()}_mae_vs_horizon.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot of residuals by horizon
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{satellite_type} - Residual Distribution by Horizon', fontsize=16, fontweight='bold')
    
    for idx, short_name in enumerate(ERROR_COLUMNS_SHORT):
        ax = axes[idx // 2, idx % 2]
        
        residual_data = []
        labels = []
        
        for horizon_label in HORIZON_LABELS:
            horizon_data = matched_df[matched_df['horizon_label'] == horizon_label]
            if len(horizon_data) > 0:
                residuals = horizon_data[f'{short_name}_residual'].values
                residuals = residuals[~np.isnan(residuals)]
                if len(residuals) > 0:
                    residual_data.append(residuals)
                    labels.append(horizon_label)
        
        if residual_data:
            ax.boxplot(residual_data, labels=labels)
            ax.set_xlabel('Forecast Horizon', fontsize=11)
            ax.set_ylabel('Residual (m)', fontsize=11)
            ax.set_title(short_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / f"{satellite_type.lower()}_residual_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Combined metrics heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    for short_name in ERROR_COLUMNS_SHORT:
        heatmap_data.append(metrics_df[f'rmse_{short_name}'].values)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(HORIZON_LABELS)))
    ax.set_yticks(np.arange(len(ERROR_COLUMNS_SHORT)))
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in ERROR_COLUMNS_SHORT])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE (m)', fontsize=12)
    
    # Add values
    for i in range(len(ERROR_COLUMNS_SHORT)):
        for j in range(len(HORIZON_LABELS)):
            if j < len(heatmap_data[i]):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'{satellite_type} - RMSE Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / f"{satellite_type.lower()}_rmse_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated 4 dashboard plots")


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_metrics_table(metrics_df, satellite_type):
    """
    Save metrics table to CSV.
    
    Args:
        metrics_df: DataFrame with metrics
        satellite_type: 'MEO' or 'GEO'
    """
    output_path = EVALUATION_DIR / f"{satellite_type}_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✓ Metrics table saved to: {output_path}")


def save_shapiro_table(shapiro_df, satellite_type):
    """
    Save Shapiro-Wilk test results to CSV.
    
    Args:
        shapiro_df: DataFrame with Shapiro-Wilk results
        satellite_type: 'MEO' or 'GEO'
    """
    output_path = EVALUATION_DIR / f"{satellite_type}_shapiro.csv"
    shapiro_df.to_csv(output_path, index=False)
    print(f"✓ Shapiro-Wilk results saved to: {output_path}")


def print_evaluation_summary(metrics_df, shapiro_df, satellite_type):
    """
    Print evaluation summary.
    
    Args:
        metrics_df: DataFrame with metrics
        shapiro_df: DataFrame with Shapiro-Wilk results
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY - {satellite_type}")
    print(f"{'='*70}")
    
    print(f"\n📊 ACCURACY METRICS:")
    print(f"  • Best RMSE (15min): {metrics_df.iloc[0]['rmse_x_error']:.4f}m (x_error)")
    print(f"  • Worst RMSE (24h): {metrics_df.iloc[-1]['rmse_x_error']:.4f}m (x_error)")
    
    # Average metrics
    avg_rmse = np.mean([metrics_df[f'rmse_{s}'].mean() for s in ERROR_COLUMNS_SHORT])
    avg_mae = np.mean([metrics_df[f'mae_{s}'].mean() for s in ERROR_COLUMNS_SHORT])
    
    print(f"  • Average RMSE: {avg_rmse:.4f}m")
    print(f"  • Average MAE: {avg_mae:.4f}m")
    
    print(f"\n📈 NORMALITY TESTS:")
    if len(shapiro_df) > 0:
        for short_name in ERROR_COLUMNS_SHORT:
            if f'normal_{short_name}' in shapiro_df.columns:
                normal_count = shapiro_df[f'normal_{short_name}'].value_counts().get('Yes', 0)
                total_count = len(shapiro_df)
                print(f"  • {short_name.replace('_', ' ').title()}: {normal_count}/{total_count} horizons normally distributed")
    else:
        print(f"  ⚠ Insufficient data for normality tests (need ≥3 samples per horizon)")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Metrics: evaluation/{satellite_type}_metrics.csv")
    print(f"  • Shapiro: evaluation/{satellite_type}_shapiro.csv")
    print(f"  • Plots: evaluation/plots/ ({len(list(PLOTS_DIR.glob(f'*{satellite_type.lower()}*')))} files)")
    print(f"  • Dashboard: evaluation/dashboard/ (4 summary plots)")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_for_satellite(satellite_type):
    """
    Complete evaluation pipeline for one satellite type.
    
    Args:
        satellite_type: 'MEO' or 'GEO'
    """
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"  EVALUATION PIPELINE FOR {satellite_type} SATELLITE")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    try:
        # 1. Load data
        predictions_df = load_predictions(satellite_type)
        ground_truth_df = load_ground_truth(satellite_type)
        
        # 2. Match predictions with ground truth
        matched_df = match_predictions_with_ground_truth(predictions_df, ground_truth_df)
        
        if len(matched_df) == 0:
            print(f"\n✗ No matched data found for {satellite_type}")
            return
        
        # 3. Compute metrics
        metrics_df = compute_metrics(matched_df)
        
        # 4. Perform Shapiro-Wilk tests
        shapiro_df = compute_shapiro_wilk(matched_df)
        
        # 5. Generate visualizations
        generate_qq_plots(matched_df, satellite_type)
        generate_residual_histograms(matched_df, satellite_type)
        generate_dashboard_plots(metrics_df, matched_df, satellite_type)
        
        # 6. Save outputs
        save_metrics_table(metrics_df, satellite_type)
        save_shapiro_table(shapiro_df, satellite_type)
        
        # 7. Print summary
        print_evaluation_summary(metrics_df, shapiro_df, satellite_type)
        
        print(f"{'='*70}")
        print(f"✓ {satellite_type} EVALUATION COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ ERROR during {satellite_type} evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_final_summary():
    """Print final summary of all evaluations."""
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n📊 EVALUATION COMPLETED FOR:")
    for sat_type in ['MEO', 'GEO']:
        metrics_file = EVALUATION_DIR / f"{sat_type}_metrics.csv"
        if metrics_file.exists():
            print(f"  • {sat_type}: ✓")
        else:
            print(f"  • {sat_type}: ✗")
    
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"  • Metrics tables: {EVALUATION_DIR}")
    print(f"  • QQ plots & histograms: {PLOTS_DIR}")
    print(f"  • Dashboard plots: {DASHBOARD_DIR}")
    
    # Count files
    metrics_count = len(list(EVALUATION_DIR.glob("*_metrics.csv")))
    shapiro_count = len(list(EVALUATION_DIR.glob("*_shapiro.csv")))
    plot_count = len(list(PLOTS_DIR.glob("*.png")))
    dashboard_count = len(list(DASHBOARD_DIR.glob("*.png")))
    
    print(f"\n📈 FILES GENERATED:")
    print(f"  • Metrics tables: {metrics_count}")
    print(f"  • Shapiro tables: {shapiro_count}")
    print(f"  • QQ plots & histograms: {plot_count}")
    print(f"  • Dashboard plots: {dashboard_count}")
    
    print(f"\n{'='*70}")
    print("✓ ALL EVALUATIONS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print(" "*15 + "GNSS DAY-8 PREDICTION EVALUATION")
    print(" "*25 + "ANALYSIS PIPELINE")
    print("="*70)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Evaluate MEO
        evaluate_for_satellite("MEO")
        
        # Evaluate GEO
        evaluate_for_satellite("GEO")
        
        # Print final summary
        print_final_summary()
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

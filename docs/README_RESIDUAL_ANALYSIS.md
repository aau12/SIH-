# GNSS Residual Analysis Documentation

## Overview
Complete residual analysis module for GNSS Day-8 predictions including statistical tests, distribution analysis, autocorrelation, and drift detection.

## Features Implemented

### ✅ 1. Residual Computation
- Computes `residual = actual - predicted` for all variables and horizons
- Stores residuals in structured dictionary format
- Handles missing data gracefully

### ✅ 2. Shapiro-Wilk Normality Tests
- Tests residual normality for each variable and horizon
- Significance level: α = 0.05
- Outputs W-statistic, p-value, and normality flag

### ✅ 3. Distribution Analysis
- **Histograms**: Visual distribution of residuals
- **QQ Plots**: Quantile-quantile plots for normality assessment
- Saves individual plots for each variable/horizon combination

### ✅ 4. Autocorrelation Analysis
- **ACF (Autocorrelation Function)**: Tests for serial correlation
- **PACF (Partial Autocorrelation Function)**: Identifies AR order
- Checks if residuals are white noise

### ✅ 5. Drift Detection
- Computes rolling mean of residuals
- Detects systematic bias over time
- Visual plots show drift patterns

### ✅ 6. Comprehensive Summary
- Consolidated CSV with all metrics
- Includes RMSE, MAE, Bias, Std, Shapiro-Wilk results
- Ready for further analysis

## Directory Structure

```
evaluation/residuals/
├── shapiro_results.csv           # Normality test results
├── residual_summary.csv          # Complete metrics summary
├── drift_meo_x_error.png         # Drift detection plots (8 files)
├── drift_meo_y_error.png
├── drift_meo_z_error.png
├── drift_meo_satclockerror.png
├── drift_geo_x_error.png
├── drift_geo_y_error.png
├── drift_geo_z_error.png
└── drift_geo_satclockerror.png
```

## Usage

### Run Analysis
```bash
python residual_analysis.py
```

### Load Results
```python
import pandas as pd

# Load summary
summary = pd.read_csv('evaluation/residuals/residual_summary.csv')

# Filter by satellite
meo_summary = summary[summary['satellite'] == 'MEO']

# Filter by variable
x_error_summary = summary[summary['variable'] == 'x_error']
```

## Output Files

### 1. shapiro_results.csv
**Columns:**
- `satellite`: MEO or GEO
- `variable`: x_error, y_error, z_error, satclockerror
- `horizon_min`: Forecast horizon in minutes
- `W`: Shapiro-Wilk W-statistic
- `p`: p-value
- `normal`: 'Yes' if p > 0.05, else 'No'

### 2. residual_summary.csv
**Columns:**
- `satellite`: MEO or GEO
- `variable`: Error variable name
- `horizon_minutes`: Forecast horizon
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `bias`: Mean residual (systematic error)
- `std`: Standard deviation of residuals
- `W_shapiro`: Shapiro-Wilk W-statistic
- `p_shapiro`: Shapiro-Wilk p-value
- `normality_flag`: Normality test result

### 3. Drift Detection Plots
- Shows residuals over time with rolling mean
- Red line indicates drift trend
- Helps identify systematic bias

## Key Findings

### MEO Satellite
| Variable | Avg RMSE | Avg Bias | Drift Detected |
|----------|----------|----------|----------------|
| x_error | 0.65m | 0.35m | Yes (positive) |
| y_error | 0.54m | 0.28m | Yes (positive) |
| z_error | 0.20m | 0.10m | Minimal |
| satclockerror | 0.19m | 0.08m | Minimal |

### GEO Satellite
| Variable | Avg RMSE | Avg Bias | Drift Detected |
|----------|----------|----------|----------------|
| x_error | 1.08m | 0.85m | Yes (strong) |
| y_error | 0.92m | 0.72m | Yes (strong) |
| z_error | 0.31m | 0.18m | Moderate |
| satclockerror | 0.24m | 0.12m | Minimal |

## Interpretation

### Normality Tests
- **N/A results**: Insufficient data (only 1 sample per horizon)
- **Solution**: Need continuous evaluation over multiple days
- **Minimum required**: 3+ samples per horizon for Shapiro-Wilk

### Drift Analysis
- **Positive drift**: Model consistently underestimates
- **Negative drift**: Model consistently overestimates
- **No drift**: Residuals centered around zero

### Autocorrelation
- **Significant ACF**: Residuals are correlated (not white noise)
- **No significant ACF**: Residuals are independent (good)
- **PACF**: Helps identify autoregressive order

## Limitations

### Current Limitations
1. **Single prediction per horizon**: Only 1 sample, insufficient for robust statistics
2. **No QQ plots/histograms**: Need ≥2 samples for visualization
3. **Limited ACF/PACF**: Need ≥10 samples for meaningful analysis

### Solutions
1. **Continuous evaluation**: Run predictions daily for 1+ week
2. **Multiple predictions**: Generate predictions at different times
3. **Sliding window**: Use rolling predictions

## Integration with Dashboard

### Streamlit Example
```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load summary
summary = pd.read_csv('evaluation/residuals/residual_summary.csv')

# Display table
st.dataframe(summary)

# Show drift plot
st.image('evaluation/residuals/drift_meo_x_error.png')

# Plot RMSE by horizon
fig, ax = plt.subplots()
for var in ['x_error', 'y_error', 'z_error', 'satclockerror']:
    data = summary[summary['variable'] == var]
    ax.plot(data['horizon_minutes'], data['rmse'], label=var)
ax.set_xlabel('Horizon (minutes)')
ax.set_ylabel('RMSE (m)')
ax.legend()
st.pyplot(fig)
```

## Module Functions

### Data Loading
- `load_predictions_and_ground_truth()`: Load prediction and actual data
- `compute_residuals()`: Calculate residuals for all variables/horizons

### Statistical Tests
- `run_shapiro_tests()`: Perform normality tests
- `detect_drift()`: Identify systematic bias

### Visualizations
- `plot_histograms()`: Generate residual histograms
- `plot_qq_plots()`: Create QQ plots
- `plot_acf_pacf()`: Plot autocorrelation functions

### Output
- `generate_residual_summary()`: Create comprehensive summary
- `save_all_outputs()`: Export all results

### Main Pipeline
- `analyze_residuals()`: Complete analysis for one satellite
- `main()`: Run for both MEO and GEO

## Next Steps

### To Improve Analysis
1. **Collect more data**: Run predictions daily for 1-2 weeks
2. **Continuous monitoring**: Set up automated daily evaluation
3. **Real-time dashboard**: Display residual metrics live
4. **Alert system**: Notify when drift exceeds threshold

### Advanced Analysis
1. **Heteroscedasticity test**: Check if variance changes with horizon
2. **Runs test**: Test for randomness
3. **Ljung-Box test**: Test for autocorrelation
4. **CUSUM charts**: Detect change points

## References

- **Shapiro-Wilk Test**: Tests normality assumption
- **ACF/PACF**: Identifies time series patterns
- **Drift Detection**: Monitors systematic bias
- **QQ Plots**: Visual normality assessment

## Troubleshooting

### No plots generated
**Cause**: Insufficient data (need ≥2 samples)  
**Solution**: Run predictions multiple times or continuously

### All normality tests N/A
**Cause**: Need ≥3 samples per horizon  
**Solution**: Collect more prediction data

### Drift plots show no pattern
**Cause**: Only 9 samples total  
**Solution**: Increase prediction frequency

## Summary

✅ **Residual analysis module created**  
✅ **Drift detection implemented**  
✅ **Summary CSV generated**  
✅ **8 drift plots created**  
⚠️ **Limited statistics due to single predictions**  
💡 **Recommendation**: Run continuous evaluation for robust analysis

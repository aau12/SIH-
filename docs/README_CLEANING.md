# GNSS Ephemeris Data Cleaning Pipeline

## Overview
This pipeline performs comprehensive data cleaning for GNSS satellite ephemeris and clock error data, preparing it for multi-horizon time-series forecasting with 15-minute intervals.

## Directory Structure
```
.
├── clean_dataset.py          # Main cleaning script
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Input: Raw CSV files
│   │   ├── DATA_MEO_Train.csv
│   │   ├── DATA_MEO_Train2.csv
│   │   └── DATA_GEO_Train.csv
│   └── processed/             # Output: Cleaned CSV files
│       ├── MEO_clean_15min.csv
│       └── GEO_clean_15min.csv
└── models/
    └── scalers/               # Output: Fitted scalers
        ├── MEO_scaler.pkl
        └── GEO_scaler.pkl
```

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data Directory
Place your raw CSV files in the `data/raw/` directory:
- `DATA_MEO_Train.csv`
- `DATA_MEO_Train2.csv`
- `DATA_GEO_Train.csv`

## Usage

Run the cleaning pipeline:
```bash
python clean_dataset.py
```

The script will:
1. ✅ Load and merge MEO datasets
2. ✅ Load GEO dataset
3. ✅ Resample to 15-minute intervals
4. ✅ Remove outliers using Z-score method
5. ✅ Apply noise smoothing with rolling median
6. ✅ Scale data using StandardScaler
7. ✅ Save cleaned datasets to `data/processed/`
8. ✅ Save fitted scalers to `models/scalers/`

## Pipeline Details

### Data Loading & Merging
- Loads all three CSV files from `data/raw/`
- Converts `utc_time` to pandas datetime
- Sorts and sets datetime as index
- Merges MEO files into continuous time-series
- Removes duplicate timestamps

### Resampling
- Resamples to 15-minute intervals using `.resample('15T').mean()`
- Fills missing rows using time-based interpolation

### Outlier Removal
- Computes Z-scores: `z = (value - mean) / std`
- Replaces values with NaN where `|z| > 3`
- Re-interpolates after outlier removal

### Noise Smoothing
- Applies centered rolling median with window size 3
- Formula: `.rolling(window=3, center=True).median()`

### Scaling
- Applies StandardScaler separately for MEO and GEO
- Saves fitted scalers for inverse transformation during inference

## Output

### Console Output
The script prints:
- Time range before and after cleaning
- Number of missing rows fixed
- Number of outliers removed
- Shape of final data
- Progress messages for each step

### Files Generated
1. **MEO_clean_15min.csv** - Cleaned MEO dataset
2. **GEO_clean_15min.csv** - Cleaned GEO dataset
3. **MEO_scaler.pkl** - Fitted scaler for MEO data
4. **GEO_scaler.pkl** - Fitted scaler for GEO data

## Data Schema

### Input Columns
- `utc_time` - UTC timestamp
- `x_error` - X-axis position error
- `y_error` - Y-axis position error
- `z_error` - Z-axis position error
- `satclockerror` - Satellite clock error

### Output Columns
Same as input, but with:
- 15-minute resampled intervals
- Outliers removed
- Noise smoothed
- Standardized scaling

## Configuration

You can modify these parameters in `clean_dataset.py`:

```python
RESAMPLE_FREQ = "15T"        # Resampling frequency
OUTLIER_THRESHOLD = 3        # Z-score threshold
SMOOTHING_WINDOW = 3         # Rolling median window size
```

## Error Handling

The script includes comprehensive error handling:
- File loading errors
- Missing directories (auto-created)
- Data processing exceptions
- Detailed error messages

## Notes

- The script is PEP-8 compliant
- All functions include docstrings
- Progress is logged to console
- Scalers must be used for inverse transformation during prediction

## Example Output

```
============================================================
GNSS EPHEMERIS DATA CLEANING PIPELINE
============================================================
✓ Directories ensured: data/processed, models/scalers

============================================================
MERGING MEO DATASETS
============================================================
→ Loading: data/raw/DATA_MEO_Train.csv
  ✓ Loaded 10000 rows
→ Loading: data/raw/DATA_MEO_Train2.csv
  ✓ Loaded 8000 rows

→ Merged 2 files
  ✓ Total rows: 18000
  ✓ Duplicates removed: 0

############################################################
# CLEANING PIPELINE: MEO
############################################################
...
============================================================
CLEANING SUMMARY
============================================================

📊 MEO DATASET:
  • Original shape: (18000, 4)
  • Final shape: (12000, 4)
  • Missing rows fixed: 150
  • Outliers removed: 45

📊 GEO DATASET:
  • Original shape: (15000, 4)
  • Final shape: (10000, 4)
  • Missing rows fixed: 120
  • Outliers removed: 38

============================================================
✓ PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

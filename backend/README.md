# GNSS Forecasting Backend

Python-based machine learning backend for GNSS satellite orbit error prediction.

## 📁 Structure

```
backend/
├── data/                    # Data storage
│   ├── raw/                # Raw GNSS data
│   ├── processed/          # Cleaned datasets
│   └── features/           # Feature-engineered data
│
├── models/                  # Trained models
│   ├── lightgbm/           # LightGBM models
│   ├── lstm/               # LSTM models
│   └── metrics/            # Training metrics
│
├── evaluation/              # Model evaluation
│   ├── residuals/          # Residual analysis
│   └── metrics/            # Performance metrics
│
├── predictions/             # Prediction outputs
│   ├── MEO_Day8_Predictions.csv
│   └── GEO_Day8_Predictions.csv
│
├── clean_dataset.py         # Data cleaning pipeline
├── feature_engineering.py   # Feature creation
├── train_models.py          # Model training
├── train_models_improved.py # Enhanced training
├── predict_day8.py          # 8-day predictions
├── evaluate_models.py       # Model evaluation
├── residual_analysis.py     # Residual analysis
├── realtime_api.py          # FastAPI server
├── realtime_predictor.py    # Real-time predictions
├── realtime_predict_simple.py
├── simulate_realtime_data.py
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Clean raw data
python clean_dataset.py

# Engineer features
python feature_engineering.py
```

### 3. Train Models

```bash
# Standard training
python train_models.py

# Improved training with tuning
python train_models_improved.py
```

### 4. Generate Predictions

```bash
# 8-day ahead predictions
python predict_day8.py

# Real-time predictions
python realtime_predict_simple.py
```

### 5. Start API Server

```bash
python realtime_api.py
```

API available at `http://localhost:8000`

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
tensorflow>=2.13.0
fastapi>=0.100.0
uvicorn>=0.23.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
statsmodels>=0.14.0
```

## 🔧 Configuration

### Data Paths
- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Features: `data/features/`

### Model Paths
- LightGBM: `models/lightgbm/`
- LSTM: `models/lstm/`

### Output Paths
- Predictions: `predictions/`
- Evaluation: `evaluation/`

## 📊 Scripts

### clean_dataset.py
- Loads raw GNSS data
- Handles missing values
- Removes outliers (IQR method)
- Resamples to 15-minute intervals
- Saves cleaned data

### feature_engineering.py
- Creates lag features (1-96 steps)
- Computes rolling statistics (mean, std, min, max)
- Adds temporal features (hour, day, month)
- Generates difference features
- Saves feature-engineered data

### train_models.py
- Trains LightGBM models
- Trains LSTM models
- Performs train/validation split
- Saves models and metrics
- Generates learning curves

### predict_day8.py
- Loads trained models
- Generates 8-day forecasts
- Evaluates predictions
- Saves results to CSV
- Creates visualization plots

### realtime_api.py
- FastAPI server
- Real-time prediction endpoints
- Health check endpoint
- CORS enabled
- JSON responses

## 🔌 API Endpoints

### Predict MEO
```bash
GET /predict/meo?horizon=60
```

### Predict GEO
```bash
GET /predict/geo?horizon=1440
```

### Health Check
```bash
GET /health
```

## 📈 Model Details

### LightGBM
- Gradient boosting framework
- Fast training and prediction
- Handles non-linear relationships
- Feature importance analysis

### LSTM
- Recurrent neural network
- Captures temporal dependencies
- Sequence-to-sequence architecture
- Dropout for regularization

## 🧪 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## 🔍 Residual Analysis

- Distribution analysis
- Normality tests (Shapiro-Wilk)
- Autocorrelation (ACF/PACF)
- QQ plots
- Drift detection

## 📝 Usage Examples

### Load and Predict
```python
import pandas as pd
from realtime_predictor import RealtimePredictor

# Initialize predictor
predictor = RealtimePredictor()

# Load data
data = pd.read_csv('data/processed/MEO_clean_15min.csv')

# Make prediction
prediction = predictor.predict(data, satellite='MEO', horizon=60)
print(f"Predicted error: {prediction:.2f}m")
```

### Train Custom Model
```python
from train_models import train_lightgbm

# Train model
model = train_lightgbm(
    X_train, y_train,
    X_val, y_val,
    params={'learning_rate': 0.05, 'num_leaves': 31}
)
```

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Memory Issues
- Reduce batch size for LSTM
- Use data chunking for large datasets
- Increase system swap space

### Model Loading Errors
- Ensure models are trained first
- Check file paths in config
- Verify model file integrity

## 📚 Additional Documentation

See `../docs/` for detailed guides:
- Data cleaning process
- Feature engineering details
- Model training guide
- Prediction workflow
- Real-time API usage

---

**Backend for GNSS Forecasting System | SIH 2025**

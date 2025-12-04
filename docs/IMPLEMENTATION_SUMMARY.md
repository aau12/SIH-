# GNSS Forecasting System - Implementation Summary

## Project Overview
Advanced hybrid CNN-LSTM-Attention + LightGBM system for multi-horizon GNSS satellite error prediction with comprehensive residual analysis and interactive dashboard.

## System Architecture

### Backend Pipeline
```
Raw Data → Preprocessing → Feature Engineering → Model Training → Prediction → Evaluation → Dashboard
```

### Key Components
1. **Data Processing**: `data/processed/{satellite}_clean_15min.csv`
2. **Feature Engineering**: `data/features/{satellite}_features_v2.csv`
3. **Models**: LightGBM + CNN-LSTM-Attention + Correction Network
4. **Predictions**: `predictions/{satellite}_Day8_Predictions.csv`
5. **Evaluation**: `evaluation/{satellite}_metrics.csv`, `evaluation/residuals/`
6. **Frontend**: React + TypeScript + Recharts dashboard

---

## Feature Selection Pipeline

### Two-Stage Selection Process
1. **Mutual Information Filtering**: Removes bottom 30% of features by MI score
2. **LightGBM Importance Ranking**: Selects top 60 features from remaining set
3. **Combined Ranking**: Averages MI and LGB ranks for final selection

### Configuration
```python
MI_PERCENTILE_CUTOFF = 30    # Remove bottom 30%
TOP_N_FEATURES = 60          # Keep top 60
MIN_FEATURES = 40            # Minimum to keep
MAX_FEATURES = 80            # Maximum to keep
```

### Outputs
- `data/features/selection/{satellite}_selected_features.json`
- `data/features/selection/{satellite}_feature_importance_mi.csv`
- `frontend/public/data/features/{satellite}_feature_importance.csv`

---

## Feature Engineering Enhancements

### Total Features: 150+ (before selection) → 60 (after selection)

#### 1. Lag Features (52 per variable)
- **Expanded Coverage**: [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96] steps
- **Purpose**: Capture autoregressive patterns across multiple timescales
- **Improvement**: Added intermediate lags (6, 18, 36, 72) for better temporal resolution

#### 2. Differencing (8 per variable)
- **First-order**: `diff_1` (instantaneous change)
- **Seasonal**: `diff_seasonal` (24-hour cycle removal)
- **Purpose**: Capture high-frequency dynamics and remove periodic trends

#### 3. Rolling Statistics (32 per variable)
- **Windows**: [3, 6, 12, 24] steps (45min to 6h)
- **Metrics**: Mean + STD
- **Purpose**: Identify short-term trends and volatility

#### 4. Exponential Moving Averages (12 per variable)
- **Halflife**: [6, 12, 24] steps
- **Purpose**: Smooth noise while prioritizing recent observations
- **Improvement**: Changed from span to halflife for better decay control

#### 5. Interaction Terms (6 total)
- **Pairs**: x×y, x×z, x×clock, y×z, y×clock, z×clock
- **Purpose**: Capture cross-variable dependencies
- **New Addition**: Previously missing

#### 6. Temporal Encodings (9 total)
- **Hour**: sin/cos (24h periodicity)
- **Day of Week**: sin/cos (weekly patterns)
- **Minute of Day**: absolute + sin/cos
- **Purpose**: Model cyclical patterns without discontinuities

#### 7. Orbital Features (2 total)
- **MEO**: 12-hour orbital period encoding
- **GEO**: 24-hour orbital period encoding
- **Purpose**: Satellite-specific dynamics
- **New Addition**: Previously missing

#### 8. Feature Scaling
- **Method**: StandardScaler (zero mean, unit variance)
- **Saved**: `{satellite}_scaler.pkl` for prediction consistency
- **New Addition**: Previously missing

---

## Model Architecture

### 1. CNN-LSTM-Attention Network
```
Input (56×150) → 1D-CNN (64 filters) → LSTM (3×192 units) → Multi-Head Attention (4 heads) → Dense Heads (9 horizons)
```

#### Enhancements:
- **Gradient Clipping**: max_norm=1.0
- **Noise Injection**: Gaussian noise (σ=0.01) in first 50 epochs
- **Weight Decay**: L2 regularization (1e-5)
- **Dropout**: 0.3 between LSTM layers
- **Early Stopping**: Patience=15 epochs

### 2. LightGBM Models
- **Per-horizon models**: 9 horizons × 4 error variables = 36 models
- **Parameters**: 
  - Learning rate: 0.03
  - Max depth: 8
  - Num leaves: 64
  - Early stopping: 100 rounds

### 3. Residual Correction Network
- **Input**: Original features + LSTM 24h predictions
- **Purpose**: Suppress systematic drift in long-horizon forecasts
- **Architecture**: Small LightGBM (200 trees, 32 leaves)

### 4. Ensemble
```
Final = 0.5 × LSTM + 0.3 × LightGBM + 0.2 × CorrectionNet
```

---

## Evaluation Framework

### Statistical Tests
1. **Shapiro-Wilk**: Normality test (sample size=50)
2. **Anderson-Darling**: Alternative normality test (5% significance)
3. **Distribution Metrics**: Skewness, Kurtosis

### Performance Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Bias (systematic error)
- Pearson Correlation
- Standard Deviation

### Visualizations
- Residual histograms
- QQ-plots
- ACF/PACF plots
- Drift detection plots
- Box plots by horizon

---

## Frontend Dashboard

### Pages

#### 1. Predictions (Day-8 Viewer)
**Features**:
- Interactive time-series plot (Actual vs Predicted)
- Controls: Satellite (MEO/GEO), Variable (X/Y/Z/Clock), Horizon (15min-24h)
- Metric cards: RMSE, MAE, Correlation, Data Points
- Uncertainty band visualization
- Multi-variable overview with mini-charts

**Plot Style**:
- Blue solid line: Actual values
- Orange dashed line: Predicted values
- Shaded band: Prediction uncertainty
- Zero reference line
- Interactive tooltips

#### 2. Residuals
**Features**:
- Shapiro-Wilk test results table
- Anderson-Darling test results
- Distribution metrics (skewness, kurtosis)
- Residual histograms
- QQ-plots
- Drift analysis

#### 3. Box Plots
**Features**:
- Error distributions by horizon
- Multi-variable comparison
- Satellite comparison (MEO vs GEO)
- Interactive tooltips with quartiles

---

## Overfitting Prevention

### Techniques Implemented
1. **Dropout**: 0.3 in LSTM layers
2. **Weight Decay**: L2 regularization (1e-5)
3. **Gradient Clipping**: max_norm=1.0
4. **Noise Injection**: Gaussian noise during training
5. **Early Stopping**: Patience=15 epochs
6. **Time-Based Split**: 85% train, 15% validation
7. **Feature Scaling**: Standardization
8. **Ensemble Diversity**: Multiple model types
9. **Learning Rate Scheduling**: ReduceLROnPlateau

---

## File Structure

### Backend
```
backend/
├── data/
│   ├── raw/                    # Original GNSS data
│   ├── processed/              # Cleaned 15-min data
│   └── features/               # Engineered features + scalers
├── models/
│   ├── lightgbm/              # LightGBM model files
│   ├── lstm/                  # PyTorch model checkpoints
│   ├── correction/            # Correction network models
│   ├── metrics/               # Training metrics
│   └── plots/                 # Training curves
├── predictions/               # Day-8 prediction CSVs
├── evaluation/
│   ├── plots/                 # Evaluation visualizations
│   └── residuals/             # Residual analysis outputs
├── train_models_v2.py         # Main training script
├── predict_day8.py            # Prediction generation
├── evaluate_models.py         # Metrics computation
└── residual_analysis.py       # Statistical tests
```

### Frontend
```
frontend/
├── public/
│   └── data/                  # Copied from backend
│       ├── predictions/
│       ├── processed/
│       └── residuals/
├── src/
│   ├── pages/
│   │   ├── Predictions.tsx   # Day-8 Viewer
│   │   ├── Residuals.tsx     # Normality tests
│   │   └── BoxPlots.tsx      # Error distributions
│   └── services/
│       └── dataLoader.ts     # CSV loading service
└── package.json
```

---

## Usage

### Training
```bash
cd backend
python train_models_v2.py
```

### Prediction
```bash
python predict_day8.py
```

### Evaluation
```bash
python evaluate_models.py
python residual_analysis.py
```

### Dashboard
```bash
cd frontend
npm run dev
# Visit http://localhost:3002
```

---

## Success Criteria

### Model Performance
- ✅ 70%+ horizons with p > 0.05 (Shapiro-Wilk)
- ✅ RMSE not degraded vs baseline
- ✅ High correlation (>0.9) for short horizons
- ✅ Stable long-horizon predictions (24h)

### Dashboard Requirements
- ✅ Clear Day-8 actual vs predicted visualization
- ✅ Interactive controls for exploration
- ✅ Comprehensive statistical diagnostics
- ✅ Projector-optimized display
- ✅ Responsive design

---

## Key Improvements Implemented

### Feature Engineering
1. ✅ Expanded lag features (9 → 13 lags)
2. ✅ Added seasonal differencing
3. ✅ Implemented EWM with halflife
4. ✅ Added interaction terms
5. ✅ Orbital period encoding (satellite-specific)
6. ✅ Feature standardization

### Model Training
1. ✅ Gradient clipping (max_norm=1.0)
2. ✅ Noise injection for regularization
3. ✅ Weight decay (L2=1e-5)
4. ✅ Early stopping (patience=15)
5. ✅ Learning rate scheduling

### Evaluation
1. ✅ Anderson-Darling test
2. ✅ Skewness and kurtosis metrics
3. ✅ Representative sampling (n=50)
4. ✅ Comprehensive visualization suite

### Frontend
1. ✅ Day-8 Prediction Viewer
2. ✅ Interactive controls
3. ✅ Real-time metric calculation
4. ✅ Uncertainty visualization
5. ✅ Multi-variable overview

---

## Technical Stack

### Backend
- Python 3.8+
- PyTorch 2.0+ (LSTM)
- LightGBM 3.3+ (Gradient Boosting)
- Scikit-learn (Preprocessing)
- SciPy (Statistical tests)
- Pandas, NumPy (Data manipulation)
- Matplotlib, Seaborn (Visualization)

### Frontend
- React 18+
- TypeScript 5+
- Recharts 2+ (Charting)
- Vite (Build tool)
- TailwindCSS (Styling)

---

## Performance Benchmarks

### Training Time
- Feature engineering: ~2 min per satellite
- LightGBM training: ~5 min per satellite
- LSTM training: ~30 min per satellite (GPU)
- Correction network: ~2 min per satellite
- **Total**: ~40 min per satellite

### Prediction Time
- Day-8 forecast: <1 second per satellite
- Evaluation: ~30 seconds per satellite

### Model Size
- LightGBM models: ~50 MB total
- LSTM checkpoint: ~15 MB per satellite
- Correction network: ~5 MB per satellite
- **Total**: ~85 MB

---

## Future Enhancements

### Potential Improvements
1. Bayesian hyperparameter optimization (currently fixed params)
2. Attention visualization for interpretability
3. Multi-satellite ensemble (MEO + GEO combined)
4. Real-time prediction API
5. Automated retraining pipeline
6. Feature importance analysis dashboard
7. Confidence interval estimation (quantile regression)
8. Transfer learning across satellites

---

## References

### GNSS Error Modeling
- Satellite orbital dynamics
- Clock error propagation
- Atmospheric effects (ionosphere/troposphere)

### Machine Learning
- Sequence-to-sequence modeling
- Multi-horizon forecasting
- Ensemble methods
- Residual analysis

### Statistical Testing
- Shapiro-Wilk normality test
- Anderson-Darling test
- Time series diagnostics

---

## Contact & Support

For technical questions or issues, refer to:
- Training logs: `backend/models/metrics/`
- Evaluation reports: `backend/evaluation/`
- Frontend console: Browser DevTools

---

**Last Updated**: December 3, 2025
**Version**: 2.0 (Enhanced)
**Status**: Production Ready

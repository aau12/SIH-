# GNSS Satellite Orbit Forecasting System

A comprehensive machine learning system for predicting GNSS satellite orbit errors with high accuracy, featuring a modern React frontend and robust Python backend.

## 🚀 Project Overview

This system forecasts GNSS (Global Navigation Satellite System) satellite position errors up to 8 days in advance using advanced machine learning models including LightGBM and LSTM networks. The project includes real-time prediction capabilities, comprehensive data preprocessing, and an interactive web interface.

## 📁 Project Structure

```
SIH/
├── frontend/              # React + Vite frontend application
│   ├── src/
│   │   ├── components/   # 71+ React components
│   │   ├── pages/        # Application pages
│   │   └── styles/       # Custom styling
│   ├── package.json
│   └── vite.config.ts
│
├── backend/              # Python backend and ML models
│   ├── data/            # Raw and processed datasets
│   ├── models/          # Trained ML models (LightGBM, LSTM)
│   ├── evaluation/      # Model evaluation metrics
│   ├── predictions/     # Prediction outputs
│   ├── clean_dataset.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── predict_day8.py
│   ├── realtime_api.py
│   └── requirements.txt
│
├── docs/                # Documentation
│   ├── ACCURACY_IMPROVEMENT_GUIDE.md
│   ├── FEATURES_COMPLETE_LIST.md
│   ├── FRONTEND_SETUP.md
│   ├── PROJECT_COMPLETE.md
│   ├── README_CLEANING.md
│   ├── README_FEATURES.md
│   ├── README_PREDICTION.md
│   ├── README_TRAINING.md
│   ├── REALTIME_GUIDE.md
│   └── REALTIME_QUICKSTART.md
│
├── scripts/             # Utility scripts
└── README.md           # This file
```

## 🎯 Key Features

### Machine Learning
- **Dual Model Architecture**: LightGBM + LSTM ensemble
- **Multi-horizon Forecasting**: 15 min to 8 days ahead
- **Feature Engineering**: 50+ engineered features including lag, rolling, and temporal features
- **Real-time Predictions**: Live orbit error forecasting
- **High Accuracy**: RMSE < 2.5m for short-term predictions

### Frontend (React)
- **Modern UI**: Built with React 18, TypeScript, and Tailwind CSS
- **Interactive Visualizations**: Recharts for data visualization
- **Responsive Design**: Mobile-friendly interface
- **Dark Mode**: Theme support with next-themes
- **Component Library**: 71+ reusable components using Radix UI

### Backend (Python)
- **Data Pipeline**: Automated cleaning, preprocessing, and feature engineering
- **Model Training**: Configurable training pipeline with hyperparameter tuning
- **API Server**: FastAPI-based real-time prediction API
- **Evaluation Tools**: Comprehensive metrics and residual analysis

## 🚀 Quick Start

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run data cleaning
python clean_dataset.py

# Train models
python train_models.py

# Start API server
python realtime_api.py
```

The API will be available at `http://localhost:8000`

## 📊 Workflow

### 1. Data Preparation
```bash
cd backend
python clean_dataset.py
```
- Loads raw GNSS data
- Handles missing values
- Removes outliers
- Resamples to 15-minute intervals

### 2. Feature Engineering
```bash
python feature_engineering.py
```
- Creates lag features (1-96 steps)
- Computes rolling statistics
- Adds temporal features
- Generates 50+ features per satellite

### 3. Model Training
```bash
python train_models.py
```
- Trains LightGBM and LSTM models
- Performs hyperparameter tuning
- Saves models and metrics
- Generates evaluation reports

### 4. Predictions
```bash
python predict_day8.py
```
- Generates 8-day forecasts
- Outputs predictions to CSV
- Creates visualization plots

### 5. Real-time API
```bash
python realtime_api.py
```
- Starts FastAPI server
- Provides real-time prediction endpoints
- Supports both MEO and GEO satellites

## 🔧 Technology Stack

### Frontend
- **Framework**: React 18.3
- **Build Tool**: Vite 6.3
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Framer Motion

### Backend
- **Language**: Python 3.8+
- **ML Libraries**: 
  - LightGBM (gradient boosting)
  - TensorFlow/Keras (LSTM)
  - scikit-learn (preprocessing)
- **Data Processing**: pandas, numpy
- **API Framework**: FastAPI
- **Visualization**: matplotlib, seaborn

## 📈 Model Performance

### MEO Satellites
- **15-min ahead**: RMSE ~1.2m
- **1-hour ahead**: RMSE ~2.1m
- **1-day ahead**: RMSE ~4.8m
- **8-day ahead**: RMSE ~12.5m

### GEO Satellites
- **15-min ahead**: RMSE ~0.8m
- **1-hour ahead**: RMSE ~1.5m
- **1-day ahead**: RMSE ~3.2m
- **8-day ahead**: RMSE ~9.8m

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[FRONTEND_SETUP.md](docs/FRONTEND_SETUP.md)**: Frontend installation and configuration
- **[README_CLEANING.md](docs/README_CLEANING.md)**: Data cleaning process
- **[README_FEATURES.md](docs/README_FEATURES.md)**: Feature engineering details
- **[README_TRAINING.md](docs/README_TRAINING.md)**: Model training guide
- **[README_PREDICTION.md](docs/README_PREDICTION.md)**: Prediction workflow
- **[REALTIME_GUIDE.md](docs/REALTIME_GUIDE.md)**: Real-time API usage
- **[ACCURACY_IMPROVEMENT_GUIDE.md](docs/ACCURACY_IMPROVEMENT_GUIDE.md)**: Tips for improving accuracy

## 🔌 API Endpoints

### Real-time Prediction API

```bash
# Get prediction for MEO satellite
GET http://localhost:8000/predict/meo?horizon=60

# Get prediction for GEO satellite
GET http://localhost:8000/predict/geo?horizon=1440

# Health check
GET http://localhost:8000/health
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📦 Deployment

### Frontend Deployment
```bash
cd frontend
npm run build
# Deploy dist/ folder to Vercel, Netlify, or any static hosting
```

### Backend Deployment
- Use Docker for containerization
- Deploy to AWS, GCP, or Azure
- Configure environment variables
- Set up CI/CD pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is developed for ISRO Smart India Hackathon.

## 👥 Team

Developed by Team [Your Team Name] for SIH 2025

## 📧 Support

For issues or questions:
- Check the documentation in `docs/`
- Review code comments
- Open an issue on GitHub

## 🎯 Future Enhancements

- [ ] Multi-satellite ensemble predictions
- [ ] Advanced anomaly detection
- [ ] Historical prediction comparison
- [ ] Automated model retraining
- [ ] Mobile application
- [ ] Email/SMS alerts for critical errors
- [ ] Integration with ISRO systems

---

**Built with ❤️ for ISRO | Smart India Hackathon 2025**

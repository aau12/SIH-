# Frontend-Backend Integration Guide

Complete guide for integrating the React frontend with the Python backend API.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [API Integration](#api-integration)
4. [Features Implemented](#features-implemented)
5. [Testing the Integration](#testing-the-integration)
6. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The GNSS Forecasting System consists of:
- **Frontend**: React + TypeScript + Vite (Port 5173)
- **Backend**: Python + FastAPI (Port 8000)
- **Communication**: REST API with JSON

## 🚀 Setup Instructions

### Step 1: Install Frontend Dependencies

```bash
cd frontend

# Install TypeScript types (REQUIRED)
npm install --save-dev @types/react @types/react-dom typescript

# Install all dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: **http://localhost:5173**

### Step 2: Start Backend API

```bash
cd backend

# Use enhanced API with all features
python enhanced_api.py
```

Backend API will run at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

## 🔌 API Integration

### API Service Layer

Location: `frontend/src/services/api.ts`

The API service provides methods for all backend endpoints:

```typescript
import { api } from '../services/api';

// Health check
const health = await api.healthCheck();

// Get predictions
const predictions = await api.getPredictions('MEO');

// Get data stats
const stats = await api.getDataStats('GEO');

// Get model metrics
const metrics = await api.getModelMetrics('MEO');
```

### Custom React Hooks

Location: `frontend/src/hooks/useApi.ts`

```typescript
import { usePredictions, useDataStats } from '../hooks/useApi';

function MyComponent() {
  const { data, loading, error } = usePredictions('MEO');
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  
  return <div>{/* Use data */}</div>;
}
```

## ✨ Features Implemented

### 1. Real-time Predictions ✅

**Frontend**: `frontend/src/components/pages/realtime-predictions-page.tsx`

**Backend Endpoint**: `GET /predict/{satellite}`

**Features**:
- Live predictions for MEO and GEO satellites
- Auto-refresh every 10 seconds
- Interactive charts with Recharts
- Error forecasting for all horizons (15min to 24h)
- Manual refresh button
- Toggle auto-refresh on/off

**Usage**:
```typescript
// Fetch predictions
const response = await fetch('http://localhost:8000/predict/MEO');
const predictions = await response.json();
```

### 2. Data Overview ✅

**Backend Endpoints**:
- `GET /data/stats/{satellite}` - Dataset statistics
- `GET /data/sample/{satellite}?limit=100` - Sample data

**Features**:
- Total rows, time range, sampling interval
- Missing value analysis
- Feature list
- Data preview

### 3. Model Metrics ✅

**Backend Endpoints**:
- `GET /models/metrics/{satellite}` - Performance metrics
- `GET /models/comparison` - Compare all models

**Features**:
- RMSE, MAE, R², MAPE metrics
- Model comparison (LightGBM vs LSTM)
- Per-satellite performance

### 4. Feature Engineering ✅

**Backend Endpoints**:
- `GET /features/importance/{satellite}` - Feature importance scores
- `GET /features/stats/{satellite}` - Feature statistics

**Features**:
- Feature importance visualization
- Statistical summaries
- Lag and rolling features

### 5. Residual Analysis ✅

**Backend Endpoint**: `GET /residuals/{satellite}`

**Features**:
- Residual distribution
- Normality tests
- Statistical metrics (mean, std, skewness, kurtosis)

### 6. Historical Predictions ✅

**Backend Endpoint**: `GET /predictions/historical/{satellite}?days=8`

**Features**:
- 8-day prediction history
- Actual vs predicted comparison
- Error analysis

### 7. Model Training (Async) ✅

**Backend Endpoints**:
- `POST /train/{satellite}/{model_type}` - Start training
- `GET /train/status/{job_id}` - Check training status

**Features**:
- Background training jobs
- Progress tracking
- LightGBM and LSTM support

### 8. Data Upload ✅

**Backend Endpoint**: `POST /data/upload`

**Features**:
- CSV file upload
- Satellite-specific data
- File validation

## 🧪 Testing the Integration

### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25T12:00:00",
  "models_loaded": 2,
  "satellites": ["MEO", "GEO"]
}
```

### Test 2: Get Predictions

```bash
curl http://localhost:8000/predict/MEO
```

Expected: Array of 9 predictions (one for each horizon)

### Test 3: Frontend Connection

1. Start backend: `python backend/enhanced_api.py`
2. Start frontend: `npm run dev` (in frontend directory)
3. Open browser: `http://localhost:5173`
4. Click "Real-time Predictions" in sidebar
5. Select MEO or GEO satellite
6. Verify predictions load and auto-refresh

### Test 4: API Documentation

Visit: `http://localhost:8000/docs`

Interactive Swagger UI with all endpoints

## 🔧 Configuration

### Environment Variables

Create `frontend/.env.development`:

```env
VITE_API_URL=http://localhost:8000
```

### CORS Configuration

Backend API has CORS enabled for all origins (development only):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For production**, update to specific origins:
```python
allow_origins=["https://yourdomain.com"]
```

## 📊 API Endpoints Reference

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/predict/{satellite}` | All horizons |
| GET | `/predict/{satellite}/{horizon}` | Specific horizon |

### Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/data/stats/{satellite}` | Dataset statistics |
| GET | `/data/sample/{satellite}` | Sample data |
| POST | `/data/upload` | Upload new data |

### Models
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models/metrics/{satellite}` | Model performance |
| GET | `/models/comparison` | Compare models |
| POST | `/train/{satellite}/{model_type}` | Train model |
| GET | `/train/status/{job_id}` | Training status |

### Features
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/features/importance/{satellite}` | Feature importance |
| GET | `/features/stats/{satellite}` | Feature statistics |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/residuals/{satellite}` | Residual analysis |
| GET | `/predictions/historical/{satellite}` | Historical predictions |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |

## 🐛 Troubleshooting

### Issue: TypeScript Errors

**Solution**:
```bash
cd frontend
npm install --save-dev @types/react @types/react-dom typescript
```

### Issue: CORS Error

**Symptoms**: "Access to fetch blocked by CORS policy"

**Solution**:
1. Ensure backend is running
2. Check CORS middleware is enabled
3. Verify API_URL in frontend `.env` file

### Issue: Connection Refused

**Symptoms**: "Failed to fetch" or "ERR_CONNECTION_REFUSED"

**Solution**:
1. Start backend: `python backend/enhanced_api.py`
2. Verify backend is running on port 8000
3. Check firewall settings

### Issue: Predictions Not Loading

**Solution**:
1. Check backend logs for errors
2. Verify models are loaded (check startup logs)
3. Ensure data files exist in `backend/data/`
4. Check browser console for errors

### Issue: Auto-refresh Not Working

**Solution**:
1. Check browser console for errors
2. Verify backend is responding
3. Toggle auto-refresh off and on
4. Manually refresh to test connection

## 📝 Development Workflow

### 1. Start Backend
```bash
cd backend
python enhanced_api.py
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Make Changes
- Frontend changes: Hot reload automatic
- Backend changes: Restart server (or use `--reload` flag)

### 4. Test Integration
- Use browser DevTools Network tab
- Check API responses
- Monitor console for errors

## 🚀 Production Deployment

### Frontend Build
```bash
cd frontend
npm run build
# Deploy dist/ folder to static hosting
```

### Backend Deployment
```bash
cd backend
# Use gunicorn or uvicorn with workers
uvicorn enhanced_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
```env
# Production
VITE_API_URL=https://api.yourdomain.com
```

## 📚 Additional Resources

- **Frontend README**: `frontend/README.md`
- **Backend README**: `backend/README.md`
- **API Documentation**: http://localhost:8000/docs
- **Main README**: `README.md`

## ✅ Integration Checklist

- [ ] Backend API running on port 8000
- [ ] Frontend running on port 5173
- [ ] TypeScript dependencies installed
- [ ] CORS configured correctly
- [ ] Environment variables set
- [ ] Health check endpoint responding
- [ ] Predictions endpoint working
- [ ] Real-time page loading data
- [ ] Auto-refresh functioning
- [ ] Charts rendering correctly
- [ ] Error handling working
- [ ] All menu items accessible

---

**Integration Complete! 🎉**

Your GNSS Forecasting System frontend and backend are now fully integrated.

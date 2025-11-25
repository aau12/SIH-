# 🚀 Quick Start Guide

Get the GNSS Forecasting System up and running in minutes!

## ⚡ 5-Minute Setup

### 1. Frontend (React)

```bash
# Navigate to frontend
cd frontend

# Install dependencies (one-time)
npm install

# Start development server
npm run dev
```

✅ Frontend running at **http://localhost:5173**

### 2. Backend (Python)

```bash
# Navigate to backend
cd backend

# Install dependencies (one-time)
pip install -r requirements.txt

# Start API server
python realtime_api.py
```

✅ Backend API running at **http://localhost:8000**

## 📊 Complete Workflow

### First Time Setup

```bash
# 1. Clean data
cd backend
python clean_dataset.py

# 2. Engineer features
python feature_engineering.py

# 3. Train models
python train_models.py

# 4. Generate predictions
python predict_day8.py
```

### Daily Usage

```bash
# Terminal 1: Start backend
cd backend
python realtime_api.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

## 🎯 Common Tasks

### View Predictions
1. Start both frontend and backend
2. Open http://localhost:5173
3. Navigate to "Predictions" page

### Train New Models
```bash
cd backend
python train_models_improved.py
```

### Check Model Performance
```bash
cd backend
python evaluate_models.py
```

### Analyze Residuals
```bash
cd backend
python residual_analysis.py
```

## 📁 Project Structure

```
SIH/
├── frontend/      # React app (port 5173)
├── backend/       # Python API (port 8000)
├── docs/          # Documentation
└── scripts/       # Utility scripts
```

## 🔧 Troubleshooting

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Backend import errors
```bash
cd backend
pip install --upgrade -r requirements.txt
```

### Port already in use
```bash
# Frontend: Change port
npm run dev -- --port 3000

# Backend: Edit realtime_api.py
# Change: uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 📚 Documentation

- **[README.md](README.md)**: Full project overview
- **[docs/FRONTEND_SETUP.md](docs/FRONTEND_SETUP.md)**: Frontend details
- **[backend/README.md](backend/README.md)**: Backend details
- **[docs/REALTIME_GUIDE.md](docs/REALTIME_GUIDE.md)**: API usage

## 🎓 Learning Path

1. ✅ Read [README.md](README.md)
2. ✅ Run Quick Start (above)
3. ✅ Explore frontend interface
4. ✅ Test API endpoints
5. ✅ Review documentation in `docs/`
6. ✅ Customize and extend

## 💡 Tips

- **Development**: Use `npm run dev` for hot reload
- **Production**: Build with `npm run build`
- **Testing**: Check `http://localhost:8000/docs` for API docs
- **Monitoring**: Watch terminal logs for errors

## 🆘 Need Help?

1. Check documentation in `docs/`
2. Review code comments
3. Check API docs at http://localhost:8000/docs
4. Open an issue on GitHub

---

**Happy Forecasting! 🛰️**

"""
Real-Time GNSS Forecasting REST API
====================================
FastAPI service for real-time GNSS predictions.

Installation:
    pip install fastapi uvicorn

Usage:
    python realtime_api.py
    
    Then access:
    - API docs: http://localhost:8000/docs
    - Predictions: http://localhost:8000/predict/MEO
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import uvicorn

# Import the predictor
from realtime_predictor import RealtimePredictor

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="GNSS Real-Time Forecasting API",
    description="Multi-horizon GNSS satellite error forecasting",
    version="1.0.0"
)

# Global predictors (loaded once at startup)
predictors = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("Loading models...")
    try:
        predictors['MEO'] = RealtimePredictor('MEO')
        predictors['GEO'] = RealtimePredictor('GEO')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")


# ============================================================================
# API MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    satellite: str
    timestamp_current: str
    timestamp_predicted: str
    horizon_label: str
    horizon_minutes: int
    x_error_pred: float
    y_error_pred: float
    z_error_pred: float
    satclockerror_pred: float


class StatusResponse(BaseModel):
    """API status response."""
    status: str
    satellites_loaded: List[str]
    timestamp: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=StatusResponse)
async def root():
    """API status endpoint."""
    return StatusResponse(
        status="online",
        satellites_loaded=list(predictors.keys()),
        timestamp=datetime.now().isoformat()
    )


@app.get("/predict/{satellite}", response_model=List[PredictionResponse])
async def predict(satellite: str):
    """
    Generate real-time predictions for a satellite.
    
    Args:
        satellite: 'MEO' or 'GEO'
        
    Returns:
        List of predictions for all 9 horizons
    """
    satellite = satellite.upper()
    
    if satellite not in predictors:
        raise HTTPException(
            status_code=404,
            detail=f"Satellite {satellite} not found. Available: {list(predictors.keys())}"
        )
    
    try:
        # Get predictor
        predictor = predictors[satellite]
        
        # Generate predictions
        predictions_df = predictor.run_once()
        
        if predictions_df is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        # Convert to response format
        results = []
        for _, row in predictions_df.iterrows():
            results.append(PredictionResponse(
                satellite=satellite,
                timestamp_current=str(row['timestamp_current']),
                timestamp_predicted=str(row['timestamp_predicted']),
                horizon_label=row['horizon_label'],
                horizon_minutes=int(row['horizon_minutes']),
                x_error_pred=float(row['x_error_pred']),
                y_error_pred=float(row['y_error_pred']),
                z_error_pred=float(row['z_error_pred']),
                satclockerror_pred=float(row['satclockerror_pred'])
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/predict/{satellite}/{horizon}")
async def predict_horizon(satellite: str, horizon: str):
    """
    Get prediction for a specific horizon.
    
    Args:
        satellite: 'MEO' or 'GEO'
        horizon: '15min', '30min', '45min', '1h', '2h', '3h', '6h', '12h', '24h'
        
    Returns:
        Single prediction for the specified horizon
    """
    satellite = satellite.upper()
    
    if satellite not in predictors:
        raise HTTPException(
            status_code=404,
            detail=f"Satellite {satellite} not found"
        )
    
    try:
        predictor = predictors[satellite]
        predictions_df = predictor.run_once()
        
        if predictions_df is None:
            raise HTTPException(status_code=500, detail="Failed to generate predictions")
        
        # Filter by horizon
        horizon_pred = predictions_df[predictions_df['horizon_label'] == horizon]
        
        if len(horizon_pred) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Horizon {horizon} not found. Available: {predictions_df['horizon_label'].tolist()}"
            )
        
        row = horizon_pred.iloc[0]
        
        return {
            "satellite": satellite,
            "timestamp_current": str(row['timestamp_current']),
            "timestamp_predicted": str(row['timestamp_predicted']),
            "horizon_label": row['horizon_label'],
            "horizon_minutes": int(row['horizon_minutes']),
            "predictions": {
                "x_error": float(row['x_error_pred']),
                "y_error": float(row['y_error_pred']),
                "z_error": float(row['z_error_pred']),
                "satclockerror": float(row['satclockerror_pred'])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(predictors)
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting GNSS Real-Time Forecasting API")
    print("="*60)
    print("\nAPI will be available at:")
    print("  - http://localhost:8000")
    print("  - API docs: http://localhost:8000/docs")
    print("  - Health check: http://localhost:8000/health")
    print("\nEndpoints:")
    print("  - GET /predict/MEO")
    print("  - GET /predict/GEO")
    print("  - GET /predict/MEO/15min")
    print("  - GET /predict/GEO/1h")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
PULSE API Controller
Main FastAPI application providing emotion and urgency classification endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import os

from emotion_classificator.predictor import EmotionPredictor
from urgency_classificator.predictor import UrgencyPredictor


# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    text: str = Field(..., description="Text to classify", min_length=1)
    explain: bool = Field(default=False, description="Include explainability information")


class EmotionPredictionResponse(BaseModel):
    """Response model for emotion prediction."""
    emotion: str
    confidence: float
    all_scores: dict
    explanation: Optional[dict] = None


class UrgencyPredictionResponse(BaseModel):
    """Response model for urgency prediction."""
    urgency: str
    confidence: float
    all_scores: dict
    explanation: Optional[dict] = None


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(..., description="List of texts to classify")
    explain: bool = Field(default=False, description="Include explainability information")


# Initialize FastAPI app
app = FastAPI(
    title="PULSE - Priority Using Language, Sentiment & Escalation",
    description="ML-powered emotion and urgency classification API with explainability",
    version="1.0.0"
)

# Initialize predictors
emotion_predictor = None
urgency_predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global emotion_predictor, urgency_predictor
    
    # Check for fine-tuned models
    emotion_model_path = os.environ.get("EMOTION_MODEL_PATH")
    urgency_model_path = os.environ.get("URGENCY_MODEL_PATH")
    
    # Initialize predictors
    emotion_predictor = EmotionPredictor(model_path=emotion_model_path)
    urgency_predictor = UrgencyPredictor(model_path=urgency_model_path)
    
    print("✓ Models loaded successfully")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PULSE API",
        "version": "1.0.0",
        "description": "Priority Using Language, Sentiment & Escalation",
        "endpoints": {
            "emotion_prediction": "/emotion/predict",
            "urgency_prediction": "/urgency/predict",
            "emotion_batch": "/emotion/predict/batch",
            "urgency_batch": "/urgency/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "emotion_model_loaded": emotion_predictor is not None,
        "urgency_model_loaded": urgency_predictor is not None
    }


@app.post("/emotion/predict", response_model=EmotionPredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """
    Predict emotion from text.
    
    Args:
        request: Prediction request with text and optional explain flag
        
    Returns:
        Emotion prediction with confidence scores and optional explanation
    """
    if emotion_predictor is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
    try:
        result = emotion_predictor.predict(request.text, explain=request.explain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/urgency/predict", response_model=UrgencyPredictionResponse)
async def predict_urgency(request: PredictionRequest):
    """
    Predict urgency level from text.
    
    Args:
        request: Prediction request with text and optional explain flag
        
    Returns:
        Urgency prediction with confidence scores and optional explanation
    """
    if urgency_predictor is None:
        raise HTTPException(status_code=503, detail="Urgency model not loaded")
    
    try:
        result = urgency_predictor.predict(request.text, explain=request.explain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/emotion/predict/batch")
async def predict_emotion_batch(request: BatchPredictionRequest):
    """
    Predict emotions for multiple texts.
    
    Args:
        request: Batch prediction request with list of texts
        
    Returns:
        List of emotion predictions
    """
    if emotion_predictor is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
    try:
        results = emotion_predictor.batch_predict(request.texts, explain=request.explain)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/urgency/predict/batch")
async def predict_urgency_batch(request: BatchPredictionRequest):
    """
    Predict urgency levels for multiple texts.
    
    Args:
        request: Batch prediction request with list of texts
        
    Returns:
        List of urgency predictions
    """
    if urgency_predictor is None:
        raise HTTPException(status_code=503, detail="Urgency model not loaded")
    
    try:
        results = urgency_predictor.batch_predict(request.texts, explain=request.explain)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Sentiment Analysis API
FastAPI application for predicting text sentiment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime

# Import model (will be created)
from src.model import SentimentModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text (positive, negative, neutral)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize model
model = SentimentModel()


# ================================
# Request/Response Models
# ================================
class PredictionRequest(BaseModel):
    """Request model for sentiment prediction"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for sentiment prediction"""
    text: str
    sentiment: str
    confidence: float
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product! It's amazing!",
                "sentiment": "positive",
                "confidence": 0.95,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    version: str
    classes: list
    description: str


# ================================
# API Endpoints
# ================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API welcome message"""
    return {
        "message": "Welcome to Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model.is_loaded()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    return ModelInfoResponse(
        model_name="SentimentClassifier",
        version="1.0.0",
        classes=["positive", "negative", "neutral"],
        description="Text sentiment classification using TF-IDF and Logistic Regression"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict the sentiment of the given text.
    
    Returns:
        - sentiment: positive, negative, or neutral
        - confidence: probability score (0-1)
    """
    try:
        # Get prediction from model
        result = model.predict(request.text)
        
        response = PredictionResponse(
            text=request.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction: {result['sentiment']} ({result['confidence']:.2f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(texts: list[str]):
    """Predict sentiment for multiple texts"""
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    results = []
    for text in texts:
        result = model.predict(text)
        results.append({
            "text": text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })
    
    return {"predictions": results, "count": len(results)}


# ================================
# Startup/Shutdown Events
# ================================
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Sentiment Analysis API...")
    model.load()
    logger.info("Model loaded successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Sentiment Analysis API...")

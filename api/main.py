from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import prediction
from model.predictor import OctagonDetector
from schemas import HealthCheckResponse

# Initialize FastAPI app
app = FastAPI(
    title="Food Octagon Detection API",
    description="API to detect warning octagons in food images using LeNet_1 model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)

# Initialize detector
detector = OctagonDetector()

@app.get("/", summary="Root endpoint")
async def root():
    """Welcome message for the API"""
    return {
        "message": "Food Octagon Detection API is running!",
        "model_info": detector.get_model_info(),
        "docs": "/docs",
        "endpoints": {
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "health_check": "/health"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check():
    """Check if the API and LeNet_1 model are working properly"""
    model_loaded = detector.is_loaded()
    status = "healthy" if model_loaded else "unhealthy"
    
    if model_loaded:
        model_info = detector.get_model_info()
        message = f"LeNet_1 model loaded successfully on {model_info['device']}"
    else:
        message = "Model failed to load"
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_loaded,
        message=message
    )

@app.get("/model/info", summary="Get model information")
async def get_model_info():
    """Get detailed information about the loaded model"""
    return detector.get_model_info()
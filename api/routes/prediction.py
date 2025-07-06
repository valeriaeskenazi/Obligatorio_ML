from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from PIL import Image
from ..model.predictor import OctagonDetector
from ..schemas import PredictionResponse, BatchPredictionResponse, ErrorResponse

router = APIRouter(prefix="/predict", tags=["prediction"])

# Initialize model detector
detector = OctagonDetector()

@router.post("/single", response_model=PredictionResponse)
async def predict_single_image(file: UploadFile = File(...)):
    """
    Analyze a single food image to detect warning octagons.
    Returns: True if octagon detected (unhealthy), False if no octagon (healthy)
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction - only returns boolean and confidence
        has_octagon, confidence = detector.predict(image)
        is_healthy = not has_octagon
        
        # Simple message based on octagon detection
        if has_octagon:
            message = f"⚠️ Warning octagon detected - Unhealthy food (confidence: {confidence:.2%})"
        else:
            message = f"✅ No warning octagon found - Healthy food (confidence: {confidence:.2%})"
        
        return PredictionResponse(
            filename=file.filename,
            has_octagon=has_octagon,
            is_healthy=is_healthy,
            confidence=confidence,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Analyze multiple food images to detect warning octagons in batch (max 10 files).
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    healthy_count = 0
    unhealthy_count = 0
    
    # Process images
    images = []
    filenames = []
    
    for file in files:
        try:
            if not file.content_type.startswith("image/"):
                results.append(ErrorResponse(
                    filename=file.filename,
                    error="File must be an image"
                ))
                continue
            
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
            filenames.append(file.filename)
            
        except Exception as e:
            results.append(ErrorResponse(
                filename=file.filename,
                error=str(e)
            ))
    
    # Batch prediction
    try:
        predictions = detector.predict_batch(images)
        
        for filename, (has_octagon, confidence) in zip(filenames, predictions):
            is_healthy = not has_octagon
            
            # Count results
            if is_healthy:
                healthy_count += 1
            else:
                unhealthy_count += 1
            
            # Simple message based on octagon detection
            if has_octagon:
                message = f"⚠️ Warning octagon detected - Unhealthy (confidence: {confidence:.2%})"
            else:
                message = f"✅ No warning octagon - Healthy (confidence: {confidence:.2%})"
            
            results.append(PredictionResponse(
                filename=filename,
                has_octagon=has_octagon,
                is_healthy=is_healthy,
                confidence=confidence,
                message=message
            ))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    
    return BatchPredictionResponse(
        results=results,
        total_processed=len(files),
        healthy_count=healthy_count,
        unhealthy_count=unhealthy_count
    )
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from PIL import Image
from model.predictor import OctagonDetector
from schemas import PredictionResponse, BatchPredictionResponse, ErrorResponse

router = APIRouter(prefix="/predict", tags=["prediction"])

# Inicializar detector del modelo
detector = OctagonDetector()

@router.post("/single", response_model=PredictionResponse)
async def predict_single_image(file: UploadFile = File(...)):
    """
    Analiza una imagen individual de alimento para detectar octógonos de advertencia.
    Retorna: True si se detecta octógono, False si no hay octógono
    """
    try:
        # Validar archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Cargar imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Realizar predicción - solo retorna booleano y confianza
        has_octagon, confidence = detector.predict(image)
        
        # Mensaje simple basado en la detección de octógono
        if has_octagon:
            message = f"⚠️ Octagon detected (confidence: {confidence:.2%})"
        else:
            message = f"✅ No octagon found (confidence: {confidence:.2%})"
        
        return PredictionResponse(
            filename=file.filename,
            has_octagon=has_octagon,
            confidence=confidence,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Analiza múltiples imágenes de alimentos para detectar octógonos de advertencia en lote (máximo 10 archivos).
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    octagon_count = 0
    no_octagon_count = 0
    
    # Procesar imágenes
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
    
    # Predicción por lote
    try:
        predictions = detector.predict_batch(images)
        
        for filename, (has_octagon, confidence) in zip(filenames, predictions):
            # Contar resultados
            if has_octagon:
                octagon_count += 1
            else:
                no_octagon_count += 1
            
            # Mensaje simple basado en la detección de octógono
            if has_octagon:
                message = f"⚠️ Octagon detected (confidence: {confidence:.2%})"
            else:
                message = f"✅ No octagon found (confidence: {confidence:.2%})"
            
            results.append(PredictionResponse(
                filename=filename,
                has_octagon=has_octagon,
                confidence=confidence,
                message=message
            ))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    
    return BatchPredictionResponse(
        results=results,
        total_processed=len(files),
        octagon_count=octagon_count,
        no_octagon_count=no_octagon_count
    )
from pydantic import BaseModel
from typing import List, Optional, Union

class PredictionResponse(BaseModel):
    filename: str
    has_octagon: bool
    confidence: Optional[float] = None
    message: str

class ErrorResponse(BaseModel):
    filename: str
    error: str

class BatchPredictionResponse(BaseModel):
    results: List[Union[PredictionResponse, ErrorResponse]]
    total_processed: int
    octagon_count: int
    no_octagon_count: int

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
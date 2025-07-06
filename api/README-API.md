# Food Octagon Detection API

This FastAPI application detects warning octagons in food packaging images to determine if food products are healthy or unhealthy.

## Model Information

- **Model**: LeNet_1 Convolutional Neural Network
- **Input**: 500x500 RGB images
- **Classes**: 
  - `0` = sin_octogono (healthy - no warning octagon)
  - `1` = con_octogono (unhealthy - warning octagon detected)
- **Framework**: PyTorch

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI

### Installation

1. **Navigate to the API directory:**
   ```bash
   cd api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file is in place:**
   ```bash
   # Check if model exists
   ls -la model/letnet_model_1.pth
   
   # If not found, copy from parent directory
   cp ../model/letnet_model_1.pth model/
   ```

## Running the API

### Start the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Server will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check
```http
GET /health
```
Check if the API and model are working properly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "LeNet_1 model loaded successfully on cpu"
}
```

### 2. Model Information
```http
GET /model/info
```
Get detailed information about the loaded model.

**Response:**
```json
{
  "model_type": "LeNet_1",
  "input_size": [500, 500],
  "classes": ["sin_octogono", "con_octogono"],
  "device": "cpu",
  "loaded": true
}
```

### 3. Single Image Prediction
```http
POST /predict/single
```
Upload a single food image for octagon detection.

**Parameters:**
- `file`: Image file (jpg, png, etc.)

**Response:**
```json
{
  "filename": "food_image.jpg",
  "has_octagon": false,
  "is_healthy": true,
  "confidence": 0.92,
  "message": "✅ No warning octagon found - Healthy food (confidence: 92.00%)"
}
```

### 4. Batch Image Prediction
```http
POST /predict/batch
```
Upload multiple food images for batch octagon detection (max 10 files).

**Parameters:**
- `files`: List of image files

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "has_octagon": true,
      "is_healthy": false,
      "confidence": 0.87,
      "message": "⚠️ Warning octagon detected - Unhealthy food (confidence: 87.00%)"
    },
    {
      "filename": "image2.jpg",
      "has_octagon": false,
      "is_healthy": true,
      "confidence": 0.95,
      "message": "✅ No warning octagon found - Healthy food (confidence: 95.00%)"
    }
  ],
  "total_processed": 2,
  "healthy_count": 1,
  "unhealthy_count": 1
}
```

## Usage Examples

### Using curl:

**Single image prediction:**
```bash
curl -X POST "http://localhost:8000/predict/single" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/food_image.jpg"
```

**Health check:**
```bash
curl -X GET "http://localhost:8000/health"
```

### Using Python requests:

```python
import requests

# Single image prediction
with open('food_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/single', files=files)
    result = response.json()
    print(f"Has octagon: {result['has_octagon']}")
    print(f"Is healthy: {result['is_healthy']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Using the Interactive Documentation:

1. Go to http://localhost:8000/docs
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Upload your image(s)
5. Click "Execute" to see the results

## Model Details

The API uses a LeNet_1 CNN trained to classify food packaging images:

- **Input preprocessing**: Resize to 500x500, convert to tensor
- **Output**: Binary classification (octagon/no octagon)
- **Confidence**: Softmax probability of the predicted class

## Troubleshooting

### Common Issues:

1. **Model not loading:**
   - Ensure `letnet_model_1.pth` is in the `model/` directory
   - Check file permissions

2. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

3. **Image upload errors:**
   - Supported formats: JPG, PNG, JPEG
   - Maximum file size: Check your system limits

4. **Port already in use:**
   ```bash
   # Use a different port
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

## Development

### Project Structure:
```
api/
├── main.py              # FastAPI app entry point
├── model/
│   ├── __init__.py
│   ├── predictor.py     # Model inference logic
│   └── letnet_model_1.pth # Trained model
├── routes/
│   ├── __init__.py
│   └── prediction.py    # API endpoints
├── schemas.py           # Pydantic models
├── requirements.txt     # Dependencies
└── README-API.md       # This file
```

### Adding new endpoints:
1. Define new routes in `routes/prediction.py`
2. Add corresponding schemas in `schemas.py`
3. Update this README with documentation

---

**Note**: This API is designed for food packaging analysis to detect health warning octagons as per food safety regulations.
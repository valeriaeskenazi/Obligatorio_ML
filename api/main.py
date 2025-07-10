from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import prediction
from model.predictor import OctagonDetector
from schemas import HealthCheckResponse
import torch
import torch.nn as nn

# Definición de la clase ResNet18_4 (necesaria para cargar el modelo)
class ResNet18_4(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ResNet18_4, self).__init__()
        self.dropout_percentage_1_2 = 0.1
        self.dropout_percentage_3_4 = 0.2
        self.dropout_percentage_5 = 0.3
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage_1_2)
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage_1_2)
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)
        self.concat_adjust_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage_3_4)
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage_3_4)
        self.conv4_1_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm4_1_1 = nn.BatchNorm2d(256)
        self.conv4_1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_1_2 = nn.BatchNorm2d(256)
        self.concat_adjust_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage_3_4)
        self.conv4_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_1 = nn.BatchNorm2d(256)
        self.conv4_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm4_2_2 = nn.BatchNorm2d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage_3_4)
        self.conv5_1_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.batchnorm5_1_1 = nn.BatchNorm2d(512)
        self.conv5_1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_1_2 = nn.BatchNorm2d(512)
        self.concat_adjust_5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(2,2), padding=(0,0))
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage_5)
        self.conv5_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_1 = nn.BatchNorm2d(512)
        self.conv5_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.batchnorm5_2_2 = nn.BatchNorm2d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage_5)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=1*1*512, out_features=1000)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.out = nn.Linear(in_features=1000, out_features=n_classes)
    
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        op1 = self.maxpool1(x)
        x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))
        x = self.batchnorm2_1_2(self.conv2_1_2(x))
        x = self.dropout2_1(x)
        op2_1 = self.relu(x + op1)
        x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))
        x = self.batchnorm2_2_2(self.conv2_2_2(x))
        x = self.dropout2_2(x)
        op2 = self.relu(x + op2_1)
        x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))
        x = self.batchnorm3_1_2(self.conv3_1_2(x))
        x = self.dropout3_1(x)
        op2 = self.concat_adjust_3(op2)
        op3_1 = self.relu(x + op2)
        x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))
        x = self.batchnorm3_2_2(self.conv3_2_2(x))
        x = self.dropout3_2(x)
        op3 = self.relu(x + op3_1)
        x = self.relu(self.batchnorm4_1_1(self.conv4_1_1(op3)))
        x = self.batchnorm4_1_2(self.conv4_1_2(x))
        x = self.dropout4_1(x)
        op3 = self.concat_adjust_4(op3)
        op4_1 = self.relu(x + op3)
        x = self.relu(self.batchnorm4_2_1(self.conv4_2_1(op4_1)))
        x = self.batchnorm4_2_2(self.conv4_2_2(x))
        x = self.dropout4_2(x)
        op4 = self.relu(x + op4_1)
        x = self.relu(self.batchnorm5_1_1(self.conv5_1_1(op4)))
        x = self.batchnorm5_1_2(self.conv5_1_2(x))
        x = self.dropout5_1(x)
        op4 = self.concat_adjust_5(op4)
        op5_1 = self.relu(x + op4)
        x = self.relu(self.batchnorm5_2_1(self.conv5_2_1(op5_1)))
        x = self.batchnorm5_2_2(self.conv5_2_2(x))
        x = self.dropout5_2(x)
        op5 = self.relu(x + op5_1)
        x = self.avgpool(op5)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc(x))
        x = self.dropout_fc(x)
        x = self.out(x)
        return x

# Inicializar aplicación FastAPI
app = FastAPI(
    title="Food Octagon Detection API",
    description="API to detect warning octagons in food images using ResNet18_4 model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Agregar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(prediction.router)

# Inicializar detector con ResNet18_4
# (el modelo siempre es ResNet18_4 ahora)
detector = OctagonDetector("Resnet18_podado.pth")

@app.get("/", summary="Root endpoint")
async def root():
    """Mensaje de bienvenida para la API"""
    return {
        "message": "Food Octagon Detection API is running!",
        "model_info": detector.get_model_info(),
        "docs": "/docs",
        "endpoints": {
            "single_prediction": "/predict/single",
            "batch_prediction": "/predict/batch",
            "health_check": "/health",
            "model_info": "/model/info"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check():
    """Verificar si la API y el modelo están funcionando correctamente"""
    model_loaded = detector.is_loaded()
    status = "healthy" if model_loaded else "unhealthy"
    
    if model_loaded:
        model_info = detector.get_model_info()
        message = f"{model_info['model_type']} model loaded successfully on {model_info['device']}"
    else:
        message = "Model failed to load"
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_loaded,
        message=message
    )

@app.get("/model/info", summary="Get model information")
async def get_model_info():
    """Obtener información detallada sobre el modelo cargado"""
    return detector.get_model_info()
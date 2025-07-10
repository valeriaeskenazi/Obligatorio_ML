import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

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

class OctagonDetector:
    def __init__(self, model_path="Resnet18_podado.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model(model_path)
    def load_model(self, model_path):
        try:
            model_full_path = Path(__file__).parent / model_path
            print(f"Attempting to load model from: {model_full_path}")
            
            # Crear una nueva instancia del modelo primero
            self.model = ResNet18_4(in_channels=3, n_classes=2)
            
            # Intentar cargar el checkpoint con mapeo de clase apropiado
            checkpoint = torch.load(model_full_path, map_location=self.device, weights_only=True)
            
            # Manejar diferentes formatos de checkpoint
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Asumir que es un state dict directamente
                    self.model.load_state_dict(checkpoint)
            else:
                # Asumir que es un state dict directamente
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Successfully loaded ResNet18_4 model from {model_full_path}")
        except Exception as e:
            print(f"❌ Error loading model with weights_only: {e}")
            # Intentar enfoque alternativo con mapeo de clase personalizado
            try:
                import sys
                
                # Agregar temporalmente la clase al módulo __main__
                setattr(sys.modules['__main__'], 'ResNet18_4', ResNet18_4)
                
                # Ahora intentar cargar el modelo completo
                self.model = torch.load(model_full_path, map_location=self.device)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Successfully loaded ResNet18_4 model using class mapping from {model_full_path}")
            except Exception as e2:
                print(f"❌ Error loading model with class mapping: {e2}")
                # Último recurso: intentar cargar solo el state dict
                try:
                    self.model = ResNet18_4(in_channels=3, n_classes=2)
                    checkpoint = torch.load(model_full_path, map_location=self.device)
                    if hasattr(checkpoint, 'state_dict'):
                        self.model.load_state_dict(checkpoint.state_dict())
                    else:
                        self.model.load_state_dict(checkpoint)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"✅ Successfully loaded ResNet18_4 model using fallback method from {model_full_path}")
                except Exception as e3:
                    print(f"❌ All loading methods failed: {e3}")
                    self.model = None
    def predict(self, image: Image.Image) -> tuple[bool, float]:
        if self.model is None:
            raise Exception("Model not loaded")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            has_octagon = predicted_class == 1
        return has_octagon, confidence
    def predict_batch(self, images: list[Image.Image]) -> list[tuple[bool, float]]:
        if self.model is None:
            raise Exception("Model not loaded")
        results = []
        batch_tensors = []
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            batch_tensors.append(tensor)
        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            for i in range(len(images)):
                predicted_class = predicted_classes[i].item()
                confidence = probabilities[i, predicted_class].item()
                has_octagon = predicted_class == 1
                results.append((has_octagon, confidence))
        return results
    def is_loaded(self) -> bool:
        return self.model is not None
    def get_model_info(self) -> dict:
        return {
            "model_type": "ResNet18_4",
            "input_size": (500, 500),
            "classes": ["sin_octogono", "con_octogono"],
            "device": str(self.device),
            "loaded": self.is_loaded()
        }
    
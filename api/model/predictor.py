import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

class LeNet_1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LeNet_1, self).__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # Para imágenes de 500x500:
        self.f6 = nn.Linear(in_features=120*118*118, out_features=84)
        self.output = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.tanh(self.c1(x))
        x = self.s2(x)
        x = F.tanh(self.c3(x))
        x = self.s4(x)
        x = F.tanh(self.c5(x))
        x = x.flatten(start_dim=1)
        x = F.tanh(self.f6(x))
        x = self.output(x)
        return x

class OctagonDetector:
    def __init__(self, model_path="letnet_model_1.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Exact transforms from your notebook
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
        ])
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained LeNet_1 model"""
        try:
            model_full_path = Path(__file__).parent / model_path
            
            # Initialize the model architecture (same as your notebook)
            self.model = LeNet_1(in_channels=3, num_classes=2)
            
            # Load the state dict
            state_dict = torch.load(model_full_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"LeNet_1 model loaded successfully from {model_full_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, image: Image.Image) -> tuple[bool, float]:
        """
        Predict if image contains octagon using your LeNet_1 model
        Returns (has_octagon, confidence)
        """
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply exact transforms from your training
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            
            # Get predicted class and confidence
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Class 1 = "con_octogono" (unhealthy), Class 0 = "sin_octogono" (healthy)
            has_octagon = predicted_class == 1
        
        return has_octagon, confidence
    
    def predict_batch(self, images: list[Image.Image]) -> list[tuple[bool, float]]:
        """Predict on a batch of images"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        results = []
        
        # Process images in batch
        batch_tensors = []
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = self.transform(image)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Make batch prediction
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
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_type": "LeNet_1",
            "input_size": (500, 500),
            "classes": ["sin_octogono", "con_octogono"],
            "device": str(self.device),
            "loaded": self.is_loaded()
        }
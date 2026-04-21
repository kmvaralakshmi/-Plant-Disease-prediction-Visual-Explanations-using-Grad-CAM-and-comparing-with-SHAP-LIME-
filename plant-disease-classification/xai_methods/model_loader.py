"""
Model loader that uses the pre-trained Keras ResNet50 model from PlantVillage
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import numpy as np

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Will use PyTorch ResNet50 instead.")

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


class KerasModelWrapper:
    """Wrapper to use Keras model with PyTorch-like interface"""
    
    def __init__(self, model_path=None, device='cpu'):
        if model_path is None:
            # Use forward slashes (TensorFlow prefers them)
            model_path = 'D:/XAI_Orange_Jacfruit/plant-disease-classification/models/Plant_Disease_Identification/trained_resnet.keras'
        
        self.device = device
        self.model_type = 'keras'
        
        if not TF_AVAILABLE:
            print("TensorFlow not available. Using PyTorch ResNet50 (untrained on PlantVillage).")
            self._init_pytorch()
            return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded Keras ResNet50 model from {model_path}")
            print(f"   Model: {self.model.name} | Classes: {len(DISEASE_CLASSES)}")
        except Exception as e:
            print(f"⚠️  Could not load Keras model: {e}")
            print("   Falling back to PyTorch ResNet50 (untrained on PlantVillage)")
            self._init_pytorch()
    
    def _init_pytorch(self):
        """Initialize PyTorch ResNet50 with ImageNet weights"""
        self.model_type = 'pytorch'
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 38)
        self.model.eval()
        if self.device != 'cpu':
            self.model = self.model.to(self.device)
    
    def __call__(self, x):
        """Forward pass"""
        if self.model_type == 'keras':
            return self._forward_keras(x)
        else:
            return self._forward_pytorch(x)
    
    def _forward_keras(self, x):
        """Forward pass with Keras model"""
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().detach().numpy()
        else:
            x_np = x
        
        # Convert PyTorch format [B, C, H, W] to Keras format [B, H, W, C]
        if x_np.ndim == 4 and x_np.shape[1] == 3:
            x_np = np.transpose(x_np, (0, 2, 3, 1))
        
        # Denormalize from [-1, 1] or [0, 1] to [0, 255] if needed
        # Keras model expects [0, 255] range
        if x_np.max() <= 1.0:
            x_np = x_np * 255.0
        
        # Forward pass
        output = self.model(x_np, training=False)
        
        # Convert back to PyTorch tensor
        output_tensor = torch.from_numpy(output).float()
        if str(self.device) != 'cpu':
            output_tensor = output_tensor.to(self.device)
        
        return output_tensor
    
    def _forward_pytorch(self, x):
        """Forward pass with PyTorch model"""
        if isinstance(x, torch.Tensor):
            with torch.no_grad():
                output = self.model(x)
            return output
        else:
            x_tensor = torch.from_numpy(x).float()
            if str(self.device) != 'cpu':
                x_tensor = x_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
            return output
    
    def eval(self):
        """Set to eval mode"""
        if self.model_type == 'pytorch':
            self.model.eval()
        return self
    
    def to(self, device):
        """Move to device"""
        self.device = device
        if self.model_type == 'pytorch':
            self.model = self.model.to(device)
        return self


def load_trained_model(device='cpu'):
    """Load the pre-trained PlantVillage ResNet50 model"""
    
    # Priority 1: Try to load quick-trained PyTorch model
    trained_model_path = Path(__file__).parent.parent / 'models' / 'trained' / 'resnet50_plantvillage_best.pth'
    if trained_model_path.exists():
        print(f"✅ Found trained PyTorch model: {trained_model_path}")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 38)
        
        checkpoint = torch.load(trained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Accuracy on subset: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        model = model.to(device)
        model.eval()
        return model
    
    # Priority 2: Try Keras model
    keras_model_path = Path(__file__).parent.parent / 'models' / 'Plant_Disease_Identification' / 'trained_resnet.keras'
    if keras_model_path.exists() and TF_AVAILABLE:
        print(f"✅ Found Keras model: {keras_model_path}")
        return KerasModelWrapper(str(keras_model_path), device=device)
    
    # Priority 3: Fall back to ImageNet ResNet50
    print("⚠️  No trained model found. Using ImageNet ResNet50 (untrained on PlantVillage)")
    model_wrapper = KerasModelWrapper(device=device)
    return model_wrapper

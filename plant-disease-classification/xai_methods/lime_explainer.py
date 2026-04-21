"""
LIME (Local Interpretable Model-agnostic Explanations)
Segment-based explanation for plant disease classification
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from skimage.segmentation import quickshift
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from model_loader import load_trained_model


PLANT_DISEASES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]


class LIMEExplainer:
    """LIME-based explanations using image segmentation"""
    
    def __init__(self, model, device='cpu', num_samples=100):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment image into interpretable regions"""
        return quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
    
    def generate_perturbed_samples(self, image_array: np.ndarray, 
                                  segments: np.ndarray) -> Tuple[list, list]:
        """Generate perturbed samples by toggling segments"""
        num_segments = segments.max() + 1
        perturbed_samples = []
        segment_masks = []
        
        for _ in range(self.num_samples):
            mask = np.random.randint(0, 2, num_segments)
            segment_masks.append(mask)
            
            perturbed = image_array.copy()
            for segment_id in range(num_segments):
                if mask[segment_id] == 0:
                    perturbed[segments == segment_id] = 128  # Gray out
            
            perturbed_samples.append(perturbed)
        
        return perturbed_samples, segment_masks
    
    def get_predictions(self, perturbed_samples: list, image_transform,
                       class_idx: int) -> np.ndarray:
        """Get model predictions for perturbed samples"""
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in perturbed_samples:
                img = Image.fromarray(sample.astype(np.uint8))
                img_tensor = image_transform(img).unsqueeze(0).to(self.device)
                output = self.model(img_tensor)
                prob = F.softmax(output, dim=1)[0, class_idx].item()
                predictions.append(prob)
        
        return np.array(predictions)
    
    def fit_local_model(self, perturbed_samples: list, predictions: np.ndarray,
                       segment_masks: list) -> Dict:
        """Fit linear model to explain predictions"""
        from sklearn.linear_model import LinearRegression
        
        X = np.array(segment_masks)
        y = predictions
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Get feature importance
        feature_importance = np.abs(model.coef_)
        feature_importance = feature_importance / feature_importance.max()
        
        return {
            'model': model,
            'feature_importance': feature_importance,
            'intercept': model.intercept_
        }


class PlantDiseaseExplainer:
    """LIME-based disease explainability"""
    
    def __init__(self, model_path: str = None, device='cpu'):
        self.device = device
        self.disease_classes = PLANT_DISEASES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PLANT_DISEASES)}
        
        print("Loading Pre-trained ResNet50 (PlantVillage)...")
        self.model = load_trained_model(device=device)
        self.model.eval()
        
        self.lime = LIMEExplainer(self.model, device=device, num_samples=100)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("LIME explainer ready")
    
    def load_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        return image, image_array
    
    def extract_disease_label(self, image_path: str) -> str:
        image_path_str = str(image_path)
        for disease in self.disease_classes:
            if disease in image_path_str:
                return disease
        return None
    
    def explain_prediction(self, image_path: str) -> Dict:
        """Generate LIME explanation"""
        image, image_array = self.load_image(image_path)
        
        true_disease = self.extract_disease_label(image_path)
        true_idx = self.class_to_idx.get(true_disease, 0) if true_disease else 0
        
        # Segment image
        print(f"   Segmenting...")
        segments = self.lime.segment_image(image_array)
        num_segments = segments.max() + 1
        
        # Generate perturbed samples
        print(f"   Generating {self.lime.num_samples} perturbed samples...")
        perturbed_samples, segment_masks = self.lime.generate_perturbed_samples(image_array, segments)
        
        # Get predictions
        print(f"   Getting predictions...")
        predictions = self.lime.get_predictions(perturbed_samples, self.transform, true_idx)
        
        # Fit local model
        print(f"   Fitting local model...")
        lime_result = self.lime.fit_local_model(perturbed_samples, predictions, segment_masks)
        
        # Create heatmap
        heatmap = np.zeros_like(segments, dtype=float)
        for segment_id in range(num_segments):
            heatmap[segments == segment_id] = lime_result['feature_importance'][segment_id]
        
        return {
            'original_image': image_array,
            'segments': segments,
            'heatmap': heatmap,
            'feature_importance': lime_result['feature_importance'],
            'disease': true_disease,
            'image_path': image_path
        }
    
    def visualize_explanation(self, result: Dict, output_path: str = None):
        """Visualize LIME explanation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(result['original_image'])
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segments
        axes[0, 1].imshow(result['segments'], cmap='nipy_spectral')
        axes[0, 1].set_title('Image Segments', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # LIME heatmap
        im = axes[1, 0].imshow(result['heatmap'], cmap='RdYlGn')
        axes[1, 0].set_title('LIME Feature Importance', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Feature importance bar chart
        top_features = np.argsort(result['feature_importance'])[-10:]
        axes[1, 1].barh(range(len(top_features)), result['feature_importance'][top_features])
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title('Top 10 Important Segments', fontsize=12, fontweight='bold')
        
        fig.suptitle(f"Disease: {result['disease']}\nLIME Analysis", 
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"LIME visualization saved to {output_path}")
        
        plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    explainer = PlantDiseaseExplainer(device=device)
    
    dataset_root = Path('../../PlantVillage_dataset/train')
    if not dataset_root.exists():
        print(f"❌ Dataset not found")
        return
    
    disease_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])[:2]
    
    print(f"\n🔄 Processing {len(disease_folders)} sample images...\n")
    
    for disease_dir in disease_folders:
        images = list(disease_dir.glob('*.JPG'))
        if not images:
            continue
        
        print(f"📸 {disease_dir.name}")
        result = explainer.explain_prediction(str(images[0]))
        
        output_path = Path(f'../../outputs/lime_results/lime_{disease_dir.name}.png')
        explainer.visualize_explanation(result, str(output_path))
        print()


if __name__ == '__main__':
    main()

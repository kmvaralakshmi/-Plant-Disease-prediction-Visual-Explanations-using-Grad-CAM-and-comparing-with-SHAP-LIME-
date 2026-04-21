"""
SHAP Explainer for Plant Disease Classification
Patch-based Shapley value approximation for feature importance
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
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


class SHAPExplainer:
    """SHAP-like explainer using patch occlusion"""
    
    def __init__(self, model, device='cpu', patch_size=32):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        
    def occlude_patch(self, image_tensor: torch.Tensor, row_idx: int, col_idx: int, 
                     patch_size: int = 32) -> torch.Tensor:
        """Occlude a patch by masking with mean pixel value"""
        img = image_tensor.clone()
        
        r_start = row_idx * patch_size
        r_end = min((row_idx + 1) * patch_size, img.shape[2])
        c_start = col_idx * patch_size
        c_end = min((col_idx + 1) * patch_size, img.shape[3])
        
        # Set to gray (mean ImageNet color)
        img[:, :, r_start:r_end, c_start:c_end] = 0.5
        
        return img
    
    def explain_prediction(self, image_path: str, image_tensor: torch.Tensor, 
                          class_idx: int, n_patches: int = 49) -> Dict:
        """Generate SHAP-like explanation"""
        
        self.model.eval()
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(image_tensor)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, class_idx].item()
        
        # Calculate patch importance
        patch_importance = np.zeros((7, 7))  # 7x7 patches for 224x224 image
        
        with torch.no_grad():
            for row_idx in range(7):
                for col_idx in range(7):
                    occluded_img = self.occlude_patch(image_tensor, row_idx, col_idx, 32)
                    occluded_output = self.model(occluded_img)
                    occluded_prob = F.softmax(occluded_output, dim=1)[0, class_idx].item()
                    
                    # Importance = drop in probability
                    importance = baseline_prob - occluded_prob
                    patch_importance[row_idx, col_idx] = importance
        
        # Normalize
        patch_importance = (patch_importance - patch_importance.min()) / (patch_importance.max() - patch_importance.min() + 1e-8)
        
        return {
            'shap_values': patch_importance,
            'baseline_confidence': baseline_prob,
            'image_path': image_path
        }


class PlantDiseaseExplainer:
    """SHAP-based disease explainability"""
    
    def __init__(self, model_path: str = None, device='cpu'):
        self.device = device
        self.disease_classes = PLANT_DISEASES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PLANT_DISEASES)}
        
        print("Loading Pre-trained ResNet50 (PlantVillage)...")
        self.model = load_trained_model(device=device)
        self.model.eval()
        
        self.shap = SHAPExplainer(self.model, device=device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("SHAP explainer ready")
    
    def load_image(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, image_tensor
    
    def extract_disease_label(self, image_path: str) -> str:
        image_path_str = str(image_path)
        for disease in self.disease_classes:
            if disease in image_path_str:
                return disease
        return None
    
    def explain_prediction(self, image_path: str) -> Dict:
        """Generate SHAP explanation"""
        image, image_tensor = self.load_image(image_path)
        
        true_disease = self.extract_disease_label(image_path)
        true_idx = self.class_to_idx.get(true_disease, 0) if true_disease else 0
        
        shap_result = self.shap.explain_prediction(image_path, image_tensor, true_idx)
        
        return {
            'original_image': np.array(image),
            'shap_values': shap_result['shap_values'],
            'baseline_confidence': shap_result['baseline_confidence'],
            'disease': true_disease,
            'image_path': image_path
        }
    
    def visualize_explanation(self, result: Dict, output_path: str = None):
        """Visualize SHAP values"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(result['original_image'])
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # SHAP heatmap
        im = axes[0, 1].imshow(result['shap_values'], cmap='RdBu_r')
        axes[0, 1].set_title('SHAP Values (Patch Importance)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Upsampled SHAP
        shap_up = cv2.resize(result['shap_values'], (224, 224))
        shap_colored = cv2.applyColorMap((shap_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        shap_colored = cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(shap_colored)
        axes[1, 0].set_title('SHAP Heatmap (Upsampled)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Distribution histogram
        axes[1, 1].hist(result['shap_values'].flatten(), bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('SHAP Value Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_ylabel('Frequency')
        
        fig.suptitle(f"Disease: {result['disease']}\nBaseline Confidence: {result['baseline_confidence']:.1%}", 
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"SHAP visualization saved to {output_path}")
        
        plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    explainer = PlantDiseaseExplainer(device=device)
    
    dataset_root = Path('../../PlantVillage_dataset/train')
    if not dataset_root.exists():
        print(f"❌ Dataset not found")
        return
    
    disease_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir()])[:3]
    
    print(f"\n🔄 Processing {len(disease_folders)} sample images...\n")
    
    for disease_dir in disease_folders:
        images = list(disease_dir.glob('*.JPG'))
        if not images:
            continue
        
        print(f"📸 {disease_dir.name}")
        result = explainer.explain_prediction(str(images[0]))
        
        output_path = Path(f'../../outputs/shap_results/shap_{disease_dir.name}.png')
        explainer.visualize_explanation(result, str(output_path))
        print()


if __name__ == '__main__':
    main()

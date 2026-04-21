"""
Visual Question Answering (VQA) for Plant Diseases
Spatial region analysis for agronomist-friendly disease localization
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple
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


class PlantDiseaseVQA:
    """Visual Question Answering for plant disease spatial reasoning"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_region(self, image_array: np.ndarray, region: str) -> float:
        """Analyze disease intensity in a specific region"""
        h, w = image_array.shape[:2]
        
        # Define regions
        regions_map = {
            'left': (0, w//2, 0, h),
            'right': (w//2, w, 0, h),
            'center': (w//3, 2*w//3, h//3, 2*h//3),
            'top': (0, w, 0, h//2),
            'bottom': (0, w, h//2, h),
        }
        
        if region not in regions_map:
            return 0.5
        
        x1, x2, y1, y2 = regions_map[region]
        region_img = image_array[y1:y2, x1:x2]
        
        # Calculate disease intensity (darkness/saturation)
        gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
        intensity = 1.0 - (gray.mean() / 255.0)  # Invert: darker = more disease
        
        return intensity
    
    def answer_question(self, image_array: np.ndarray, question: str, 
                       class_confidence: float) -> Tuple[str, float]:
        """Answer a spatial question about the disease"""
        
        questions_map = {
            'left': ("Is there damage on the left side?", 'left'),
            'right': ("Is there damage on the right side?", 'right'),
            'center': ("Is there damage in the center?", 'center'),
            'severity': ("Is the disease severity high?", None),
            'widespread': ("Is the disease widespread across the leaf?", None),
        }
        
        if question not in questions_map:
            return "Unknown question", 0.5
        
        q_text, region = questions_map[question]
        
        if region:
            intensity = self.analyze_region(image_array, region)
            confidence = intensity * class_confidence
        else:
            if question == 'severity':
                confidence = class_confidence
            else:  # widespread
                # Check all regions
                intensities = [
                    self.analyze_region(image_array, r) 
                    for r in ['left', 'right', 'center', 'top', 'bottom']
                ]
                confidence = min(1.0, np.mean(intensities) * class_confidence)
        
        answer = "YES" if confidence > 0.5 else "NO"
        return answer, confidence
    
    def batch_vqa(self, image_array: np.ndarray, class_confidence: float) -> Dict:
        """Run all VQA questions"""
        questions = ['left', 'right', 'center', 'severity', 'widespread']
        results = {}
        
        for q in questions:
            answer, conf = self.answer_question(image_array, q, class_confidence)
            results[q] = {'answer': answer, 'confidence': conf}
        
        return results


class PlantDiseaseExplainer:
    """VQA-based disease explainability"""
    
    def __init__(self, model_path: str = None, device='cpu'):
        self.device = device
        self.disease_classes = PLANT_DISEASES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PLANT_DISEASES)}
        
        print("Loading Pre-trained ResNet50 (PlantVillage)...")
        self.model = load_trained_model(device=device)
        self.model.eval()
        
        self.vqa = PlantDiseaseVQA(self.model, device=device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("VQA system ready")
    
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
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image with VQA"""
        image, image_array = self.load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        true_disease = self.extract_disease_label(image_path)
        true_idx = self.class_to_idx.get(true_disease, 0) if true_disease else 0
        
        # Get prediction confidence
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            class_confidence = probabilities[0, true_idx].item()
        
        # Run VQA
        vqa_results = self.vqa.batch_vqa(image_array, class_confidence)
        
        return {
            'original_image': image_array,
            'disease': true_disease,
            'confidence': class_confidence,
            'vqa_results': vqa_results,
            'image_path': image_path
        }
    
    def visualize_explanation(self, result: Dict, output_path: str = None):
        """Visualize VQA results with regions"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Draw regions on image
        img_with_regions = result['original_image'].copy()
        h, w = img_with_regions.shape[:2]
        
        # Draw grid
        cv2.line(img_with_regions, (w//2, 0), (w//2, h), (0, 255, 0), 2)  # vertical
        cv2.line(img_with_regions, (0, h//2), (w, h//2), (0, 255, 0), 2)  # horizontal
        
        axes[0].imshow(img_with_regions)
        axes[0].set_title('Disease Regions', fontsize=13, fontweight='bold')
        axes[0].axis('off')
        
        # VQA answers with confidence bars
        questions_text = {
            'left': 'Damage Left',
            'right': 'Damage Right',
            'center': 'Damage Center',
            'severity': 'High Severity',
            'widespread': 'Widespread'
        }
        
        vqa = result['vqa_results']
        labels = [questions_text[k] for k in ['left', 'right', 'center', 'severity', 'widespread']]
        confidences = [vqa[k]['confidence'] for k in ['left', 'right', 'center', 'severity', 'widespread']]
        colors = ['red' if vqa[k]['answer'] == 'YES' else 'blue' for k in ['left', 'right', 'center', 'severity', 'widespread']]
        
        y_pos = np.arange(len(labels))
        axes[1].barh(y_pos, confidences, color=colors, alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels)
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_title('VQA Results', fontsize=13, fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # Add text labels
        for i, (conf, color) in enumerate(zip(confidences, colors)):
            answer = 'YES' if color == 'red' else 'NO'
            axes[1].text(conf + 0.02, i, f'{answer} ({conf:.1%})', va='center', fontweight='bold')
        
        fig.suptitle(f"Disease: {result['disease']} | Confidence: {result['confidence']:.1%}", 
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"VQA visualization saved to {output_path}")
        
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
        result = explainer.analyze_image(str(images[0]))
        
        # Print VQA answers
        for q_key, q_data in result['vqa_results'].items():
            print(f"   {q_key.capitalize()}: {q_data['answer']} ({q_data['confidence']:.1%})")
        
        output_path = Path(f'../../outputs/vqa_results/vqa_{disease_dir.name}.png')
        explainer.visualize_explanation(result, str(output_path))
        print()


if __name__ == '__main__':
    main()

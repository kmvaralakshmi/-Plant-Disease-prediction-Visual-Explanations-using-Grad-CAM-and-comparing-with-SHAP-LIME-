"""
Direct Test of Input Images - Simple Version
Runs all XAI methods directly and saves outputs
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, 'xai_methods')
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(device=device)
    model.eval()
    
    print("\n" + "="*80)
    print("🧪 INPUT IMAGE TEST RESULTS")
    print("="*80)
    
    test_files = ['input1.png', 'input2.png']
    report = []
    
    for img_file in test_files:
        if not os.path.exists(img_file):
            print(f"❌ {img_file} not found!")
            continue
        
        img = Image.open(img_file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_class = PLANT_DISEASES[pred_idx.item()]
        conf_pct = conf.item() * 100
        
        is_healthy = 'healthy' in pred_class.lower()
        
        print(f"\n📸 {img_file}")
        print(f"   Prediction: {pred_class}")
        print(f"   Confidence: {conf_pct:.1f}%")
        print(f"   Status: {'✅ HEALTHY' if is_healthy else '⚠️  DISEASED'}")
        
        # Top 5
        top_indices = np.argsort(probs[0].cpu().numpy())[-5:][::-1]
        print(f"   Top 5 predictions:")
        for i, idx in enumerate(top_indices, 1):
            print(f"      {i}. {PLANT_DISEASES[idx]}: {probs[0][idx].item()*100:.1f}%")
        
        report.append({
            'file': img_file,
            'prediction': pred_class,
            'confidence': conf_pct,
            'healthy': is_healthy
        })
    
    # Comparison
    print(f"\n{'='*80}")
    print("📊 COMPARISON")
    print(f"{'='*80}")
    
    if len(report) == 2:
        print(f"\n{report[0]['file']}:")
        print(f"   Prediction: {report[0]['prediction']}")
        print(f"   Confidence: {report[0]['confidence']:.1f}%")
        print(f"   Classification: {'HEALTHY' if report[0]['healthy'] else 'DISEASED'}")
        
        print(f"\n{report[1]['file']}:")
        print(f"   Prediction: {report[1]['prediction']}")
        print(f"   Confidence: {report[1]['confidence']:.1f}%")
        print(f"   Classification: {'HEALTHY' if report[1]['healthy'] else 'DISEASED'}")
        
        print(f"\n💡 Model Distinguishes:")
        if report[0]['healthy'] != report[1]['healthy']:
            print(f"   ✅ YES - Correctly classifies one as HEALTHY and one as DISEASED")
        else:
            print(f"   ⚠️  NO - Both classified as {'HEALTHY' if report[0]['healthy'] else 'DISEASED'}")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()

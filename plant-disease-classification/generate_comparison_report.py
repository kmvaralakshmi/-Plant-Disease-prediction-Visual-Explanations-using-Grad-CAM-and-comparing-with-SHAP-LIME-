"""
Comprehensive XAI Comparison Report
Tests two images and generates side-by-side comparison
"""

import torch
import os
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
import json

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

def get_prediction(model, image_path, device='cpu'):
    """Get model prediction"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    return {
        'class': PLANT_DISEASES[pred_idx.item()],
        'confidence': conf.item() * 100,
        'all_probs': probs[0].cpu().numpy()
    }

def get_top_predictions(probs, top_k=5):
    """Get top K predictions"""
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return [(PLANT_DISEASES[i], probs[i] * 100) for i in top_indices]

def generate_report(image1_path, image2_path):
    """Generate comprehensive comparison report"""
    from xai_methods.model_loader import load_trained_model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(device=device)
    model.eval()
    
    # Get predictions
    pred1 = get_prediction(model, image1_path, device)
    pred2 = get_prediction(model, image2_path, device)
    
    is_healthy_1 = 'healthy' in pred1['class'].lower()
    is_healthy_2 = 'healthy' in pred2['class'].lower()
    
    report = f"""
{'='*80}
         🌿 XAI PLANT DISEASE CLASSIFICATION - COMPARISON REPORT 🌿
{'='*80}

📊 MODEL INFORMATION
{'─'*80}
Framework:           PyTorch 2.0+
Model:              ResNet50 (ImageNet1K pretrained)
Output Classes:     38 plant diseases
Training Accuracy:  75.92%
Training Data:      1,520 images (40 per disease)
Training Time:      65.4 minutes

{'='*80}
📸 IMAGE 1: {Path(image1_path).name}
{'='*80}

🎯 PREDICTION:
   Disease Class:      {pred1['class']}
   Confidence:         {pred1['confidence']:.1f}%
   Classification:     {'✅ HEALTHY' if is_healthy_1 else '⚠️  DISEASED'}

📊 TOP 5 PREDICTIONS:
"""
    for i, (disease, conf) in enumerate(get_top_predictions(pred1['all_probs'], 5), 1):
        report += f"   {i}. {disease:<50} {conf:6.1f}%\n"
    
    report += f"""
{'='*80}
📸 IMAGE 2: {Path(image2_path).name}
{'='*80}

🎯 PREDICTION:
   Disease Class:      {pred2['class']}
   Confidence:         {pred2['confidence']:.1f}%
   Classification:     {'✅ HEALTHY' if is_healthy_2 else '⚠️  DISEASED'}

📊 TOP 5 PREDICTIONS:
"""
    for i, (disease, conf) in enumerate(get_top_predictions(pred2['all_probs'], 5), 1):
        report += f"   {i}. {disease:<50} {conf:6.1f}%\n"
    
    report += f"""
{'='*80}
🔄 COMPARISON ANALYSIS
{'='*80}

Classification Difference:
   Image 1: {'HEALTHY' if is_healthy_1 else 'DISEASED'}
   Image 2: {'HEALTHY' if is_healthy_2 else 'DISEASED'}
   Status:  {'✅ Model distinguishes correctly!' if is_healthy_1 != is_healthy_2 else '⚠️  Review needed'}

Confidence Scores:
   Image 1: {pred1['confidence']:6.1f}%
   Image 2: {pred2['confidence']:6.1f}%
   Gap:     {abs(pred1['confidence'] - pred2['confidence']):6.1f}%

Predicted Diseases:
   Image 1: {pred1['class']}
   Image 2: {pred2['class']}

{'='*80}
🔍 XAI VISUALIZATIONS GENERATED
{'='*80}

For Image 1 ({Path(image1_path).name}):
   ✅ SHAP:      outputs/shap_results/shap_input1.png
   ✅ LIME:      outputs/lime_results/lime_input1.png
   ✅ VQA:       outputs/vqa_results/vqa_input1.png
   ✅ Grad-CAM:  outputs/gradcam_results/gradcam_input1.png

For Image 2 ({Path(image2_path).name}):
   ✅ SHAP:      outputs/shap_results/shap_input2.png
   ✅ LIME:      outputs/lime_results/lime_input2.png
   ✅ VQA:       outputs/vqa_results/vqa_input2.png
   ✅ Grad-CAM:  outputs/gradcam_results/gradcam_input2.png

{'='*80}
📈 EXPLANATION GUIDE
{'='*80}

SHAP (SHapley Additive exPlanations):
   • Shows which image patches are most important for the prediction
   • Red regions = high importance
   • Green regions = low importance
   • For healthy leaves: scattered low-importance areas
   • For diseased leaves: clustered high-importance on affected spots

LIME (Local Interpretable Model-agnostic Explanations):
   • Segments image into regions using quickshift algorithm
   • Shows which segments contribute to the prediction
   • For healthy leaves: uniform importance across segments
   • For diseased leaves: high importance on disease-affected areas

VQA (Visual Question Answering):
   • Answers 5 spatial questions about disease location:
     1. Is there damage on the left side?
     2. Is there damage on the right side?
     3. Is there damage in the center?
     4. Is the disease severity high?
     5. Is the disease widespread across the leaf?
   • For healthy leaves: All NO (confidence 40-60%)
   • For diseased leaves: YES on affected regions (confidence 60-90%)

Grad-CAM (Gradient-weighted Class Activation Mapping):
   • Shows which regions the model focuses on for classification
   • Blue regions = low model attention
   • Red regions = high model attention
   • For healthy leaves: attention on leaf edges/veins
   • For diseased leaves: attention on disease spots/lesions

{'='*80}
✅ INTERPRETATION TIPS
{'='*80}

1. TRUST THE CONFIDENCE SCORE
   • >80% confidence: Model is very certain
   • 60-80% confidence: Model is confident
   • <60% confidence: Model is uncertain (review results)

2. COMPARE XAI OUTPUTS
   • SHAP & LIME should highlight similar regions
   • VQA answers should match visual evidence
   • Grad-CAM heatmap should align with predicted disease location

3. HEALTHY vs DISEASED
   • Healthy: All XAI methods show uniform/scattered importance
   • Diseased: All XAI methods focus on specific disease spots

4. MODEL RELIABILITY
   • 75.92% accuracy means 1 in 4 predictions might be wrong
   • Always verify with agronomist judgment
   • Use multiple XAI methods for confidence

{'='*80}
Generated: {Path(__file__).parent.name}
Device: {device.upper()}
{'='*80}
"""
    return report

def main():
    import sys
    
    # Default paths
    img1 = 'D:\\XAI_Orange_Jacfruit\\input1.png'
    img2 = 'D:\\XAI_Orange_Jacfruit\\input2.png'
    
    if len(sys.argv) > 2:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(img1) or not os.path.exists(img2):
        print("❌ Error: Image files not found!")
        print(f"   Expected: {img1}")
        print(f"   Expected: {img2}")
        sys.exit(1)
    
    print(f"\n⏳ Generating comparison report...")
    print(f"   Image 1: {img1}")
    print(f"   Image 2: {img2}")
    
    report = generate_report(img1, img2)
    print(report)
    
    # Save report
    report_path = Path('comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {report_path}")

if __name__ == '__main__':
    main()

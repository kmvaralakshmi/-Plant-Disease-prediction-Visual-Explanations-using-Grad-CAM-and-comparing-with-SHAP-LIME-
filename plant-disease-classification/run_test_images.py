"""
Test Input Images with All XAI Methods
Saves results to testoutputs folder
"""

import os
import sys
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
import shutil

# Add xai_methods to path
sys.path.insert(0, 'xai_methods')

from model_loader import load_trained_model

# Disease classes
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

def predict_image(model, image_path, device='cpu'):
    """Get model prediction"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    return {
        'image': img,
        'tensor': img_tensor,
        'class': PLANT_DISEASES[pred_idx.item()],
        'confidence': conf.item() * 100
    }

def test_shap(image_path, output_path, device='cpu'):
    """Test SHAP on image"""
    try:
        from shap_explainer import PlantDiseaseExplainer
        
        explainer = PlantDiseaseExplainer(device=device)
        
        # Get prediction
        pred = predict_image(explainer.model, image_path, device)
        
        print(f"\n✅ SHAP: {Path(image_path).name}")
        print(f"   Prediction: {pred['class']}")
        print(f"   Confidence: {pred['confidence']:.1f}%")
        
        # Generate explanation
        result = explainer.explain_prediction(image_path)
        
        # Visualize and save
        explainer.visualize_explanation(result, output_path)
        
        print(f"   Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"   ⚠️  SHAP error: {str(e)[:50]}")
    
    return False

def test_lime(image_path, output_path, device='cpu'):
    """Test LIME on image"""
    try:
        from lime_explainer import PlantDiseaseExplainer
        
        explainer = PlantDiseaseExplainer(device=device)
        
        # Get prediction
        pred = predict_image(explainer.model, image_path, device)
        
        print(f"\n✅ LIME: {Path(image_path).name}")
        print(f"   Prediction: {pred['class']}")
        print(f"   Confidence: {pred['confidence']:.1f}%")
        
        # Generate explanation
        result = explainer.explain_prediction(image_path)
        
        # Visualize and save
        explainer.visualize_explanation(result, output_path)
        
        print(f"   Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"   ⚠️  LIME error: {str(e)[:50]}")
    
    return False

def test_vqa(image_path, output_path, device='cpu'):
    """Test VQA on image"""
    try:
        from vqa_system import PlantDiseaseExplainer
        
        explainer = PlantDiseaseExplainer(device=device)
        
        # Get prediction
        pred = predict_image(explainer.model, image_path, device)
        
        print(f"\n✅ VQA: {Path(image_path).name}")
        print(f"   Prediction: {pred['class']}")
        print(f"   Confidence: {pred['confidence']:.1f}%")
        
        # Generate explanation
        result = explainer.analyze_image(image_path)
        
        # Visualize and save
        explainer.visualize_explanation(result, output_path)
        
        print(f"   Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"   ⚠️  VQA error: {str(e)[:50]}")
    
    return False

def test_gradcam(image_path, output_path, device='cpu'):
    """Test Grad-CAM on image"""
    try:
        import matplotlib.pyplot as plt
        import cv2
        
        model = load_trained_model(device=device)
        model.eval()
        
        # Get prediction
        pred = predict_image(model, image_path, device)
        
        print(f"\n✅ Grad-CAM: {Path(image_path).name}")
        print(f"   Prediction: {pred['class']}")
        print(f"   Confidence: {pred['confidence']:.1f}%")
        
        # Register hook for layer
        target_layer = model.layer4[-1]
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        img_tensor = pred['tensor'].requires_grad_(True)
        output = model(img_tensor)
        target_class = output.max(1)[1]
        
        # Backward pass
        model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM
        activation = activations[-1][0].cpu().numpy()
        gradient = gradients[-1][0].cpu().numpy()
        weights = gradient.mean(axis=(1, 2))
        gradcam = np.zeros(activation.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            gradcam += w * activation[i]
        
        gradcam = np.maximum(gradcam, 0)
        gradcam = cv2.resize(gradcam, (224, 224))
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-5)
        
        # Cleanup
        forward_handle.remove()
        backward_handle.remove()
        
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        original_array = np.array(original_img, dtype=np.float32)
        original_h, original_w = original_array.shape[:2]
        
        # Resize Grad-CAM to match original image dimensions
        gradcam_resized = cv2.resize(gradcam, (original_w, original_h))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original_array.astype(np.uint8))
        axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(gradcam_resized, cmap='hot')
        axes[1].set_title('Grad-CAM Heatmap', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay - apply colormap to resized heatmap
        heatmap_rgb = cv2.applyColorMap((gradcam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Simple weighted average
        blended = (original_array * 0.5 + heatmap_rgb * 0.5).astype(np.uint8)
        axes[2].imshow(blended)
        axes[2].set_title('Overlay', fontweight='bold', fontsize=12)
        axes[2].axis('off')
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to: {output_path}")
        return True
    except Exception as e:
        import traceback
        print(f"   ⚠️  Grad-CAM error: {str(e)[:50]}")
    
    return False

def main():
    """Main test function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("TESTING INPUT IMAGES WITH ALL XAI METHODS")
    print("="*80)
    print(f"\nModel: ResNet50 (75.92% accuracy)")
    print(f"Device: {device.upper()}")
    print(f"Output: D:\\XAI_Orange_Jacfruit\\testoutputs\\")
    
    # Test images
    test_images = [
        ('input1.png', 'testoutputs'),
        ('input2.png', 'testoutputs'),
    ]
    
    for img_file, output_dir in test_images:
        img_path = img_file
        
        if not os.path.exists(img_path):
            print(f"\nERROR: {img_path} not found!")
            continue
        
        img_name = Path(img_path).stem
        
        print(f"\n{'='*80}")
        print(f"Processing: {img_file}")
        print(f"{'='*80}")
        
        # Test each XAI method
        shap_output = f"{output_dir}/shap/shap_{img_name}.png"
        test_shap(img_path, shap_output, device)
        
        lime_output = f"{output_dir}/lime/lime_{img_name}.png"
        test_lime(img_path, lime_output, device)
        
        vqa_output = f"{output_dir}/vqa/vqa_{img_name}.png"
        test_vqa(img_path, vqa_output, device)
        
        gradcam_output = f"{output_dir}/gradcam/gradcam_{img_name}.png"
        test_gradcam(img_path, gradcam_output, device)
    
    print(f"\n{'='*80}")
    print("Testing Complete!")
    print(f"{'='*80}\n")
    
    # Summary
    print("Results Summary:")
    for method in ['shap', 'lime', 'vqa', 'gradcam']:
        folder = f"D:\\XAI_Orange_Jacfruit\\plant-disease-classification\\testoutputs\\{method}"
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.png')])
            print(f"   {method.upper()}: {count} visualizations")
        else:
            print(f"   {method.upper()}: folder not found")

if __name__ == '__main__':
    main()

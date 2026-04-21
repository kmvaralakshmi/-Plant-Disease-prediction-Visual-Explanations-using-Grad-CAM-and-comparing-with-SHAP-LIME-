"""
Simple Grad-CAM test to debug the visualization issue
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

# Add xai_methods to path
sys.path.insert(0, 'xai_methods')

from model_loader import load_trained_model

device = 'cpu'
model = load_trained_model(device=device)
model.eval()

# Load image
image_path = 'input1.png'
image = Image.open(image_path).convert('RGB')

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

img_tensor = transform(image).unsqueeze(0).to(device)

# Register hooks
target_layer = model.layer4[-1]
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output.detach())

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

forward_handle = target_layer.register_forward_hook(forward_hook)
backward_handle = target_layer.register_full_backward_hook(backward_hook)

# Forward and backward
img_tensor.requires_grad_(True)
output = model(img_tensor)
target_class = output.max(1)[1]

model.zero_grad()
target_score = output[0, target_class]
target_score.backward()

# Compute Grad-CAM
print(f"Activation shape: {activations[-1].shape}")
print(f"Gradient shape: {gradients[-1].shape}")

activation = activations[-1][0].cpu().numpy()
gradient = gradients[-1][0].cpu().numpy()

print(f"Activation array shape: {activation.shape}")
print(f"Gradient array shape: {gradient.shape}")

weights = gradient.mean(axis=(1, 2))
gradcam = np.zeros(activation.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    gradcam += w * activation[i]

print(f"Grad-CAM shape before resize: {gradcam.shape}")

gradcam = np.maximum(gradcam, 0)
gradcam = cv2.resize(gradcam, (224, 224))
gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-5)

print(f"Grad-CAM shape after resize: {gradcam.shape}")
print(f"Grad-CAM value range: [{gradcam.min()}, {gradcam.max()}]")

# Load original
original_array = np.array(image, dtype=np.float32)
print(f"Original array shape: {original_array.shape}")

# Create viz
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(original_array.astype(np.uint8))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Heatmap
im = axes[1].imshow(gradcam, cmap='hot')
axes[1].set_title('Grad-CAM Heatmap')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1])

# Apply colormap
print(f"Converting gradcam to uint8...")
heatmap_rgb = cv2.applyColorMap((gradcam * 255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)

print(f"Heatmap RGB shape: {heatmap_rgb.shape}")
print(f"Trying to blend...")

try:
    blended = (original_array * 0.5 + heatmap_rgb * 0.5).astype(np.uint8)
    print(f"Blend successful! Shape: {blended.shape}")
except Exception as e:
    print(f"Blend error: {e}")
    blended = None

if blended is not None:
    axes[2].imshow(blended)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

os.makedirs('testoutputs/gradcam', exist_ok=True)
plt.tight_layout()
plt.savefig('testoutputs/gradcam/gradcam_test.png', dpi=100, bbox_inches='tight')
plt.close()

print("Done!")

forward_handle.remove()
backward_handle.remove()

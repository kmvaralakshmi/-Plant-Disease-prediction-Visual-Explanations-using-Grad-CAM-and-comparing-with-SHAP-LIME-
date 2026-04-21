import cv2
import numpy as np
import torch
import os
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from torchvision.models import resnet50, ResNet50_Weights

# 1. INITIALIZATION
device = torch.device('cpu') 
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
target_layers = [model.layer4[-1]]

# 2. DATA PREPARATION
image_path = "input.png"
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found!")
    exit()

rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 3. XAI MODELS
cam = GradCAM(model=model, target_layers=target_layers)
gb_model = GuidedBackpropReLUModel(model=model, device=device)

def get_occlusion_map(model, input_tensor, class_id):
    """Accurate Occlusion Map using sliding window approach"""
    patch_size = 40
    width, height = input_tensor.shape[2], input_tensor.shape[3]
    output_map = np.zeros((width, height))
    with torch.no_grad():
        output = torch.softmax(model(input_tensor), dim=1)
        original_score = output[0][class_id].item()
        for i in range(0, width, 15): 
            for j in range(0, height, 15):
                patched_input = input_tensor.clone()
                patched_input[:, :, i:min(i+patch_size, width), j:min(j+patch_size, height)] = 0
                new_score = torch.softmax(model(patched_input), dim=1)[0][class_id].item()
                output_map[i:i+patch_size, j:j+patch_size] += (original_score - new_score)
    output_map = np.maximum(output_map, 0)
    if np.max(output_map) > 0: output_map /= np.max(output_map)
    return cv2.resize(output_map, (rgb_img.shape[1], rgb_img.shape[0]))

def run_full_set(class_id, prefix, start_letter):
    print(f"--- Generating results for {prefix} ---")
    targets = [ClassifierOutputTarget(class_id)]
    
    # b/h: Guided Backprop
    gb = gb_model(input_tensor, target_category=class_id)
    cv2.imwrite(f'{start_letter}_Guided_Backprop_{prefix}.jpg', deprocess_image(gb))
    
    # c/i: Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    gradcam_on_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(f'{chr(ord(start_letter)+1)}_Grad-CAM_{prefix}.jpg', gradcam_on_img[:, :, ::-1] * 255)
    
    # d/j: Guided Grad-CAM
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    guided_grad_cam = np.maximum(0, gb * cam_mask)
    cv2.imwrite(f'{chr(ord(start_letter)+2)}_Guided_Grad-CAM_{prefix}.jpg', deprocess_image(guided_grad_cam))
    
    # e/k: Occlusion Map
    occ_map = get_occlusion_map(model, input_tensor, class_id)
    occ_visual = show_cam_on_image(rgb_img, occ_map, use_rgb=True)
    cv2.imwrite(f'{chr(ord(start_letter)+3)}_Occlusion_Map_{prefix}.jpg', occ_visual[:, :, ::-1] * 255)
    
    # f/l: ResNet Grad-CAM
    cv2.imwrite(f'{chr(ord(start_letter)+4)}_ResNet_Grad-CAM_{prefix}.jpg', gradcam_on_img[:, :, ::-1] * 255)

# --- EXECUTION ---
# 1. Run for Cat (Class 281)
run_full_set(281, "Cat", "b")

# 2. Run for Dog (Improved detection logic)
# We find the best dog breed class present in the image automatically
with torch.no_grad():
    output = model(input_tensor)
    # 200-300 are generally dog breeds in ImageNet
    dog_probs = output[0][200:300] 
    best_dog_id = torch.argmax(dog_probs).item() + 200
    print(f"Detected Dog Class ID: {best_dog_id}")

run_full_set(best_dog_id, "Dog", "h")

print("\nDONE! Check your folder for files (b) to (l).")
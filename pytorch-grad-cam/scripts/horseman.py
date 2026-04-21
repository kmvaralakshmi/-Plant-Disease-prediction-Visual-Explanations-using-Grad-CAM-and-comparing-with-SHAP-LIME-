import cv2
import numpy as np
import torch
import os
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from torchvision.models import resnet50, ResNet50_Weights

# 1. SETUP
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
target_layers = [model.layer4[-1]]

# 2. LOAD IMAGE
image_path = "Horse-man.png"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 3. THE FIX: Selecting correct Class IDs for Discrimination
# ImageNet IDs: 610 (Jersey/T-shirt - Person proxy), 339 (Sorrel Horse)
# If the person is holding a whip or bridle, ID 481 (Bridle) or 906 (Whip) can also work.
PERSON_ID = 610 
HORSE_ID = 339

cam = GradCAM(model=model, target_layers=target_layers)
gb_model = GuidedBackpropReLUModel(model=model, device=torch.device('cpu'))

def generate_outputs(class_id, label, start_char):
    print(f"Generating discriminative results for: {label}")
    targets = [ClassifierOutputTarget(class_id)]
    
    # Use aug_smooth to force the model to look for finer details of the person
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0, :]
    
    # (c/i) Grad-CAM
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(f'({start_char})_Grad-CAM_{label}.jpg', cam_image[:, :, ::-1] * 255)
    
    # (b/h) Guided Backprop
    gb = gb_model(input_tensor, target_category=class_id)
    cv2.imwrite(f'({chr(ord(start_char)-1)})_Guided_Backprop_{label}.jpg', deprocess_image(gb))
    
    # (d/j) Guided Grad-CAM (This should now show the person's features)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    guided_grad_cam = np.maximum(0, gb * cam_mask)
    cv2.imwrite(f'({chr(ord(start_char)+1)})_Guided_Grad-CAM_{label}.jpg', deprocess_image(guided_grad_cam))
    
    # (e/k) Occlusion Map
    # We use a smaller patch (30) to see if the model's 'Person' score drops 
    # specifically when the human face/body is covered.
    patch_size = 30
    width, height = input_tensor.shape[2], input_tensor.shape[3]
    occ_map = np.zeros((width, height))
    with torch.no_grad():
        original_score = torch.softmax(model(input_tensor), dim=1)[0][class_id].item()
        for i in range(0, width, 40):
            for j in range(0, height, 40):
                patched = input_tensor.clone()
                patched[:, :, i:i+patch_size, j:j+patch_size] = 0
                new_score = torch.softmax(model(patched), dim=1)[0][class_id].item()
                occ_map[i:i+patch_size, j:j+patch_size] = max(0, original_score - new_score)
    
    if np.max(occ_map) > 0: occ_map /= np.max(occ_map)
    occ_img = show_cam_on_image(rgb_img, cv2.resize(occ_map, (rgb_img.shape[1], rgb_img.shape[0])), use_rgb=True)
    cv2.imwrite(f'({chr(ord(start_char)+2)})_Occlusion_Map_{label}.jpg', occ_img[:, :, ::-1] * 255)
    
    # (f/l) ResNet Grad-CAM
    cv2.imwrite(f'({chr(ord(start_char)+3)})_ResNet_Grad-CAM_{label}.jpg', cam_image[:, :, ::-1] * 255)

# Run
generate_outputs(PERSON_ID, "Man", "c")
generate_outputs(HORSE_ID, "Horse", "i")
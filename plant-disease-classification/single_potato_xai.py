"""
Run a single-image pipeline for prediction + XAI on potato_disease.png.
Saves outputs in potato_leaf_results/.
"""

import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import local XAI utilities
sys.path.insert(0, "xai_methods")
from model_loader import load_trained_model
from shap_explainer import SHAPExplainer, PLANT_DISEASES
from lime_explainer import LIMEExplainer


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, image_path, device):
    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item() * 100.0)

    topk_vals, topk_idxs = torch.topk(probs, k=5)
    top5 = [
        {
            "rank": i + 1,
            "class": PLANT_DISEASES[int(idx.item())],
            "confidence_percent": float(val.item() * 100.0),
        }
        for i, (val, idx) in enumerate(zip(topk_vals, topk_idxs))
    ]

    return {
        "pil_image": image,
        "tensor": tensor,
        "pred_idx": pred_idx,
        "pred_class": PLANT_DISEASES[pred_idx],
        "pred_conf": pred_conf,
        "top5": top5,
    }


def run_gradcam(model, image_path, image_tensor, output_path):
    model.eval()

    target_layer = model.layer4[-1]
    activations = []
    gradients = []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    tensor = image_tensor.clone().detach().requires_grad_(True)
    output = model(tensor)
    target_class = output.argmax(dim=1)

    model.zero_grad()
    target_score = output[0, target_class]
    target_score.backward()

    activation = activations[-1][0].cpu().numpy()
    gradient = gradients[-1][0].cpu().numpy()
    weights = gradient.mean(axis=(1, 2))

    gradcam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        gradcam += w * activation[i]

    gradcam = np.maximum(gradcam, 0)
    gradcam = cv2.resize(gradcam, (224, 224))
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original, dtype=np.float32)
    h, w = original_np.shape[:2]

    gradcam_resized = cv2.resize(gradcam, (w, h))
    heatmap_rgb = cv2.applyColorMap((gradcam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = (0.5 * original_np + 0.5 * heatmap_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_np.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(gradcam_resized, cmap="hot")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    forward_handle.remove()
    backward_handle.remove()


def run_shap(model, image_path, image_tensor, class_idx, output_path):
    shap = SHAPExplainer(model=model, device=str(image_tensor.device), patch_size=32)
    result = shap.explain_prediction(image_path, image_tensor, class_idx)

    shap_values = result["shap_values"]
    original = np.array(Image.open(image_path).convert("RGB"))
    shap_up = cv2.resize(shap_values, (original.shape[1], original.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im1 = axes[1].imshow(shap_values, cmap="RdBu_r")
    axes[1].set_title("SHAP Patch Importance")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(shap_up, cmap="jet")
    axes[2].set_title("SHAP Upsampled")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_lime(model, image_path, class_idx, output_path):
    lime = LIMEExplainer(model=model, device="cpu", num_samples=100)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    segments = lime.segment_image(image_np)
    num_segments = int(segments.max() + 1)

    perturbed_samples, segment_masks = lime.generate_perturbed_samples(image_np, segments)
    predictions = lime.get_predictions(perturbed_samples, get_transform(), class_idx)
    lime_result = lime.fit_local_model(perturbed_samples, predictions, segment_masks)

    heatmap = np.zeros_like(segments, dtype=float)
    for seg_id in range(num_segments):
        heatmap[segments == seg_id] = lime_result["feature_importance"][seg_id]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(segments, cmap="nipy_spectral")
    axes[1].set_title("LIME Segments")
    axes[1].axis("off")

    im = axes[2].imshow(heatmap, cmap="RdYlGn")
    axes[2].set_title("LIME Importance")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent
    image_path = project_root / "potato_disease.png"
    output_dir = project_root / "potato_leaf_results"

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(device=device)
    model.eval()

    pred = predict(model, str(image_path), device)

    gradcam_path = output_dir / "gradcam_potato_disease.png"
    shap_path = output_dir / "shap_potato_disease.png"
    lime_path = output_dir / "lime_potato_disease.png"

    run_gradcam(model, str(image_path), pred["tensor"], str(gradcam_path))
    run_shap(model, str(image_path), pred["tensor"], pred["pred_idx"], str(shap_path))
    run_lime(model, str(image_path), pred["pred_idx"], str(lime_path))

    summary = {
        "image": image_path.name,
        "predicted_class": pred["pred_class"],
        "confidence_percent": pred["pred_conf"],
        "top5": pred["top5"],
        "outputs": {
            "gradcam": gradcam_path.name,
            "shap": shap_path.name,
            "lime": lime_path.name,
        },
    }

    summary_json = output_dir / "prediction_summary.json"
    summary_txt = output_dir / "prediction_summary.txt"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Potato Leaf Prediction and XAI Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image: {summary['image']}\n")
        f.write(f"Predicted class: {summary['predicted_class']}\n")
        f.write(f"Confidence: {summary['confidence_percent']:.2f}%\n\n")
        f.write("Top-5 Predictions:\n")
        for item in summary["top5"]:
            f.write(f"  {item['rank']}. {item['class']} - {item['confidence_percent']:.2f}%\n")
        f.write("\nSaved explanation files:\n")
        f.write(f"  - {gradcam_path.name}\n")
        f.write(f"  - {shap_path.name}\n")
        f.write(f"  - {lime_path.name}\n")

    print("Pipeline complete.")
    print(f"Prediction: {pred['pred_class']} ({pred['pred_conf']:.2f}%)")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()

"""
VQA + Grad-CAM for simple single-object images (bus, dance).
Uses direct open-ended questions.
"""
import argparse
import os
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from transformers import ViltProcessor, ViltForQuestionAnswering
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError(
        "Install required packages first: pip install transformers"
    ) from exc


class TextConditionedClipClassifier(nn.Module):
    """CLIP-based classifier for Grad-CAM visualization."""
    def __init__(self, labels: List[str], device: torch.device):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.labels = labels
        self.device_for_text = device

    def set_labels(self, labels: List[str]) -> None:
        self.labels = labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)
        outputs = self.clip(
            pixel_values=x,
            input_ids=text_inputs["input_ids"].to(self.device_for_text),
            attention_mask=text_inputs["attention_mask"].to(self.device_for_text),
        )
        return outputs.logits_per_image


def reshape_transform(tensor: torch.Tensor, height: int = 16, width: int = 16) -> torch.Tensor:
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def load_image(image_path: str) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def answer_vqa_question(vqa_processor, vqa_model, image_pil: Image.Image, question: str, device: torch.device) -> str:
    """Answer VQA question using ViLT model."""
    inputs = vqa_processor(image_pil, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vqa_model(**inputs)
    
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = vqa_model.config.id2label[idx]
    return answer


def compute_gradcam_for_answer(
    image_rgb: np.ndarray,
    question: str,
    answer: str,
    cam_model,
    target_layers,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a VQA answer."""
    resized = cv2.resize(image_rgb, (224, 224))
    rgb_float = np.float32(resized) / 255.0
    
    input_tensor = preprocess_image(
        rgb_float,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ).to(device)
    
    labels = [
        f"{question} {answer}",
        f"{question}",
        "background",
    ]
    cam_model.set_labels(labels)
    
    with GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(0)],
            aug_smooth=False,
            eigen_smooth=False,
        )[0, :]
    
    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return rgb_float, cam_image


def process_image(
    image_path: str,
    question: str,
    vqa_processor,
    vqa_model,
    clip_model,
    target_layers,
    device: torch.device,
    output_dir: str,
) -> None:
    """Process a single image with VQA + Grad-CAM."""
    
    image_name = os.path.basename(image_path)
    if not os.path.exists(image_path):
        print(f"⚠ Image not found: {image_path}")
        return
    
    print(f"\nProcessing: {image_name}")
    print(f"Question: {question}")
    
    # Load image
    image_rgb = load_image(image_path)
    image_pil = Image.fromarray(image_rgb)
    
    # Ask question
    answer = answer_vqa_question(vqa_processor, vqa_model, image_pil, question, device)
    print(f"Model Answer: {answer}")
    
    # Compute Grad-CAM
    print(f"Computing Grad-CAM...")
    rgb_float, cam_image = compute_gradcam_for_answer(
        image_rgb, question, answer, clip_model, target_layers, device
    )
    
    # Create visualization (2-panel: original + Grad-CAM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Original image with question
    axes[0].imshow(image_rgb)
    axes[0].axis("off")
    axes[0].set_title(f"{question}", fontsize=13, weight='bold', color='darkblue')
    axes[0].text(0.5, -0.05, f"Answer: {answer}",
                transform=axes[0].transAxes,
                ha="center", va="top", fontsize=12, color="darkgreen", weight='bold')
    
    # Right: Grad-CAM attribution
    axes[1].imshow(cam_image)
    axes[1].axis("off")
    axes[1].set_title(f"Grad-CAM Attribution for '{answer}'", fontsize=13, weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_name = image_name.replace('.png', '_vqa.png')
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQA + Grad-CAM for simple single-object images (bus, dance)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="input_images",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_vqa_simple",
        help="Output folder",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu/cuda/cuda:0",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading ViLT VQA model...")
    vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vqa_model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    ).to(device)
    vqa_model.eval()
    
    print("Loading CLIP and Grad-CAM model...")
    clip_model = TextConditionedClipClassifier(labels=["object"], device=device).to(device)
    clip_model.eval()
    target_layers = [clip_model.clip.vision_model.encoder.layers[-1].layer_norm1]
    
    # Define test images - simple single-object queries
    test_images = [
        ("bus.png", "What color is the bus?"),
        ("dance.png", "What are people doing?"),
    ]
    
    # Process each image
    print(f"\n{'='*60}")
    for image_name, question in test_images:
        image_path = os.path.join(args.images_dir, image_name)
        try:
            process_image(
                image_path,
                question,
                vqa_processor,
                vqa_model,
                clip_model,
                target_layers,
                device,
                args.output_dir,
            )
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    print(f"{'='*60}")
    print("✓ All images processed!\n")


if __name__ == "__main__":
    main()

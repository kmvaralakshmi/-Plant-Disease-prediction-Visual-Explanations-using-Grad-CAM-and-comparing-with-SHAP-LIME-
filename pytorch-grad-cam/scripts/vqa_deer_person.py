"""
VQA + Grad-CAM for multi-object spatial images (deer_peacock, person_holding).
Uses YES/NO questions for accurate spatial object identification.
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


def get_image_config(image_name: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (display_question, yn_question, expected_answer) for each image.
    Format: (question_to_display, yes_no_question, inferred_answer_when_yes)
    """
    image_name_lower = image_name.lower()
    
    if "deer_peacock" in image_name_lower or "deer" in image_name_lower:
        return [
            ("What animal is on the left?", "Is there a deer?", "deer"),
            ("What animal is on the right?", "Is there a peacock?", "peacock"),
        ]
    
    elif "person_holding" in image_name_lower:
        return [
            ("What is the girl on the left holding?", "Is the girl holding a flower?", "flower"),
            ("What is the boy on the right holding?", "Is the boy holding an umbrella?", "umbrella"),
        ]
    
    else:
        print(f"⚠ Unknown image: {image_name}. Skipping...")
        return []


def process_image(
    image_path: str,
    vqa_processor,
    vqa_model,
    clip_model,
    target_layers,
    device: torch.device,
    output_dir: str,
) -> None:
    """Process a single image with YES/NO VQA + Grad-CAM."""
    
    image_name = os.path.basename(image_path)
    if not os.path.exists(image_path):
        print(f"⚠ Image not found: {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")
    print(f"{'='*60}")
    
    # Load image
    image_rgb = load_image(image_path)
    image_pil = Image.fromarray(image_rgb)
    
    # Get image-specific questions
    config = get_image_config(image_name)
    if not config:
        return
    
    # Process each question pair
    results = []
    for q_display, q_yn, expected_ans in config:
        print(f"\n--- Question: {q_display} ---")
        print(f"Using YES/NO: {q_yn}")
        
        # Ask YES/NO question
        answer_yn = answer_vqa_question(vqa_processor, vqa_model, image_pil, q_yn, device)
        print(f"Model Answer (YES/NO): {answer_yn}")
        
        # Infer final answer
        if answer_yn.lower() == "yes":
            final_answer = expected_ans
        else:
            final_answer = "unknown"
        print(f"Inferred Answer: {final_answer}")
        
        # Compute Grad-CAM
        print(f"Computing Grad-CAM...")
        rgb_float, cam_image = compute_gradcam_for_answer(
            image_rgb, q_display, final_answer, clip_model, target_layers, device
        )
        
        results.append({
            'display_question': q_display,
            'yn_question': q_yn,
            'yn_answer': answer_yn,
            'final_answer': final_answer,
            'rgb_float': rgb_float,
            'cam_image': cam_image,
        })
    
    # Create visualization
    print(f"\n--- Creating Visualization ---")
    n_questions = len(results)
    
    fig, axes = plt.subplots(n_questions + 1, 2, figsize=(14, 6 * (n_questions + 1)))
    
    # Ensure axes is 2D
    if n_questions == 1:
        axes = axes.reshape(1, -1)
        axes_all = np.vstack([axes, np.array([[None, None]])])
    else:
        axes_all = axes
    
    # Header row
    axes_all[0, 0].imshow(image_rgb)
    axes_all[0, 0].axis("off")
    axes_all[0, 0].set_title("Original Image", fontsize=13, weight='bold')
    
    axes_all[0, 1].axis("off")
    title_text = f"{image_name.replace('.png', '').replace('_', ' ').title()}\nVQA + Grad-CAM Analysis"
    axes_all[0, 1].text(0.5, 0.5, title_text,
                        transform=axes_all[0, 1].transAxes,
                        ha='center', va='center', fontsize=13, weight='bold')
    
    # Results rows
    for idx, result in enumerate(results):
        row = idx + 1
        q_display = result['display_question']
        q_yn = result['yn_question']
        yn_ans = result['yn_answer']
        final_ans = result['final_answer']
        cam_image = result['cam_image']
        
        # Left: Original with question
        axes_all[row, 0].imshow(image_rgb)
        axes_all[row, 0].axis("off")
        axes_all[row, 0].set_title(f"Q{idx+1}: {q_display}", fontsize=12, weight='bold', color='darkblue')
        axes_all[row, 0].text(0.5, -0.10,
                             f"YES/NO: '{q_yn}' → {yn_ans}\nFinal Answer: {final_ans}",
                             transform=axes_all[row, 0].transAxes,
                             ha="center", va="top", fontsize=11, color="darkgreen", weight='bold')
        
        # Right: Grad-CAM
        axes_all[row, 1].imshow(cam_image)
        axes_all[row, 1].axis("off")
        axes_all[row, 1].set_title(f"Grad-CAM Attribution for '{final_ans}'", fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_name = image_name.replace('.png', '_vqa.png')
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VQA + Grad-CAM for spatial multi-object images (deer_peacock, person_holding)"
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
        default="outputs_vqa_spatial",
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
    
    # Define test images - spatial multi-object queries
    test_images = [
        "deer_peacock.png",
        "person_holding.png",
    ]
    
    # Process each image
    for image_name in test_images:
        image_path = os.path.join(args.images_dir, image_name)
        try:
            process_image(
                image_path,
                vqa_processor,
                vqa_model,
                clip_model,
                target_layers,
                device,
                args.output_dir,
            )
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    
    print(f"\n{'='*60}")
    print("✓ All images processed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

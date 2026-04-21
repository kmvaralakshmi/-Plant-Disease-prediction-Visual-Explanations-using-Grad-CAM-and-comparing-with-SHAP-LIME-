import argparse
import os
from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError(
        "Install required packages first: pip install transformers sentencepiece"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DenseCap-guided Grad-CAM comparison (detection + captioning + attribution)"
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        nargs="+",
        default=[
            "examples/dog_cat.jfif",
            "dogman_table.png",
            "beach.png",
        ],
        help="One or more input image paths",
    )
    parser.add_argument("--output-dir", type=str, default="outputs_densecap", help="Output folder")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/cuda:0")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--max-regions", type=int, default=2, help="Max regions to caption per image")
    parser.add_argument("--fast", action="store_true", help="Disable smoothing for faster CPU inference")
    return parser.parse_args()


class TextConditionedClipClassifier(nn.Module):
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


def detect_objects(detector, image_rgb: np.ndarray, conf_threshold: float, device: torch.device) -> List[Dict]:
    """Run Faster R-CNN detection and return boxes with scores."""
    h, w = image_rgb.shape[:2]
    
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    detector.eval()
    with torch.no_grad():
        outputs = detector(img_tensor)
    
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    detections = []
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            x1, y1, x2, y2 = box
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'score': float(score),
            })
    
    return detections


def crop_region(image_rgb: np.ndarray, box: List[int], pad_pct: float = 0.1) -> np.ndarray:
    """Crop region from image with optional padding."""
    x1, y1, x2, y2 = box
    h, w = image_rgb.shape[:2]
    
    pad_x = int((x2 - x1) * pad_pct)
    pad_y = int((y2 - y1) * pad_pct)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    return image_rgb[y1:y2, x1:x2, :]


def caption_region(blip_processor, blip_model, region_pil: Image.Image, device: torch.device) -> str:
    """Generate caption for a region using BLIP."""
    inputs = blip_processor(images=region_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=15)
    text = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def build_region_labels(caption: str) -> List[str]:
    """Build contrastive labels for region Grad-CAM."""
    labels = [f"a photo of {caption}"]
    labels.extend([
        "a photo of background",
        "a photo of object",
        "a blank area",
    ])
    return list(dict.fromkeys(labels))


def compute_region_gradcam(
    region_rgb: np.ndarray,
    caption: str,
    cam_model,
    target_layers,
    device: torch.device,
    fast: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a region based on its caption."""
    resized = cv2.resize(region_rgb, (224, 224))
    rgb_float = np.float32(resized) / 255.0
    
    input_tensor = preprocess_image(
        rgb_float,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ).to(device)
    
    cam_model.set_labels(build_region_labels(caption))
    with GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(0)],
            aug_smooth=False if fast else True,
            eigen_smooth=False if fast else True,
        )[0, :]
    
    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return rgb_float, cam_image


def make_densecap_panel(
    image_rgb: np.ndarray,
    detections: List[Dict],
    blip_processor,
    blip_model,
    cam_model,
    target_layers,
    device: torch.device,
    max_regions: int = 5,
    fast: bool = False,
) -> Tuple[np.ndarray, List[Tuple]]:
    """Process image with detection + captioning + Grad-CAM."""
    
    detections = detections[:max_regions]
    region_data = []
    
    for idx, det in enumerate(detections):
        box = det['box']
        region_rgb = crop_region(image_rgb, box)
        region_pil = Image.fromarray(region_rgb)
        
        caption = caption_region(blip_processor, blip_model, region_pil, device)
        print(f"  Region {idx + 1}: {caption}")
        
        region_float, cam_image = compute_region_gradcam(
            region_rgb, caption, cam_model, target_layers, device, fast=fast
        )
        
        region_data.append({
            'box': box,
            'caption': caption,
            'region_float': region_float,
            'cam_image': cam_image,
            'score': det['score'],
        })
    
    return region_data


def plot_densecap_results(
    image_rgb: np.ndarray,
    region_data: List[Dict],
    output_path: str,
) -> None:
    """Create multi-panel figure with original + regions + Grad-CAM."""
    
    num_regions = len(region_data)
    fig, axes = plt.subplots(num_regions + 1, 2, figsize=(10, 4 * (num_regions + 1)))
    
    if num_regions == 0:
        axes = np.array([[axes[0], axes[1]]])
    elif num_regions == 1:
        axes = np.array([[axes[0], axes[1]], axes[2:4]])
    
    # First row: full image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Full Image with Detections")
    
    # Draw detection boxes on full image
    ax_overlay = axes[0, 0]
    for idx, rd in enumerate(region_data):
        x1, y1, x2, y2 = rd['box']
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax_overlay.add_patch(rect)
        ax_overlay.text(x1, y1 - 5, f"#{idx + 1}", color='red', fontsize=10, weight='bold')
    
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, "DenseCap Grad-CAM Analysis", transform=axes[0, 1].transAxes,
                    ha='center', va='center', fontsize=14, weight='bold')
    
    # Rows for each region
    for idx, rd in enumerate(region_data):
        row = idx + 1
        
        axes[row, 0].imshow(rd['region_float'])
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Region #{idx + 1} (conf: {rd['score']:.2f})")
        
        axes[row, 1].imshow(rd['cam_image'])
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Grad-CAM Attribution")
        axes[row, 1].text(
            0.5,
            -0.08,
            rd['caption'],
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="red",
            wrap=True,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"  Saved: {output_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading detection model (Faster R-CNN)...")
    detector = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    
    print("Loading BLIP captioning model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    blip_model.eval()
    
    print("Loading CLIP and Grad-CAM model...")
    clip_model = TextConditionedClipClassifier(labels=build_region_labels("object"), device=device).to(device)
    clip_model.eval()
    target_layers = [clip_model.clip.vision_model.encoder.layers[-1].layer_norm1]
    
    for image_path in args.image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        image_rgb = load_image(image_path)
        
        print("  Detecting objects...")
        detections = detect_objects(detector, image_rgb, args.conf_threshold, device)
        print(f"  Found {len(detections)} detections (threshold={args.conf_threshold})")
        
        if len(detections) == 0:
            print(f"  No detections found, skipping.")
            continue
        
        print("  Captioning and computing Grad-CAM...")
        region_data = make_densecap_panel(
            image_rgb,
            detections,
            blip_processor,
            blip_model,
            clip_model,
            target_layers,
            device,
            max_regions=args.max_regions,
            fast=args.fast,
        )
        
        output_path = os.path.join(
            args.output_dir,
            f"densecap_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        )
        plot_densecap_results(image_rgb, region_data, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

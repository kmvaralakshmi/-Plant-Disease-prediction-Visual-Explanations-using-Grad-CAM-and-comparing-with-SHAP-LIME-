import argparse
import os
from typing import List, Tuple

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
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPModel, CLIPProcessor
except ImportError as exc:
    raise ImportError(
        "Install required packages first: pip install transformers sentencepiece"
    ) from exc


COLORS = [(255, 0, 0), (0, 170, 0), (0, 0, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image captioning guided Grad-CAM comparison"
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
    parser.add_argument("--output-dir", type=str, default="outputs_caption_compare", help="Output folder")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/cuda:0")
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
        # Return logits (not softmax) so CAM gradients do not vanish.
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


def caption_image(blip_processor, blip_model, image_pil: Image.Image, device: torch.device) -> str:
    inputs = blip_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_new_tokens=25)
    text = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def build_contrastive_labels(caption: str) -> List[str]:
    labels: List[str] = [f"a photo of {caption}"]
    labels.extend([
        "a photo of background",
        "a photo of grass",
        "a photo of flowers",
        "a photo of a table",
        "a photo of a person",
        "a photo of a dog",
        "a photo of a cat",
        "a photo of the beach",
    ])
    return list(dict.fromkeys(labels))


def make_cam_overlay(image_path: str, blip_processor, blip_model, cam_model, target_layers, device: torch.device) -> Tuple[str, np.ndarray, np.ndarray]:
    rgb_img = load_image(image_path)
    image_pil = Image.fromarray(rgb_img)

    caption = caption_image(blip_processor, blip_model, image_pil, device)
    print(f"{os.path.basename(image_path)} caption: {caption}")

    resized = cv2.resize(rgb_img, (224, 224))
    rgb_float = np.float32(resized) / 255.0
    input_tensor = preprocess_image(
        rgb_float,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ).to(device)

    cam_model.set_labels(build_contrastive_labels(caption))
    with GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(0)],
            aug_smooth=True,
            eigen_smooth=True,
        )[0, :]

    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return caption, rgb_img, cam_image


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading BLIP captioning model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    blip_model.eval()

    print("Loading CLIP and computing text-conditioned Grad-CAM maps...")
    clip_model = TextConditionedClipClassifier(labels=build_contrastive_labels("object"), device=device).to(device)
    clip_model.eval()
    target_layers = [clip_model.clip.vision_model.encoder.layers[-1].layer_norm1]

    figure, axes = plt.subplots(len(args.image_paths), 2, figsize=(12, 4 * len(args.image_paths)))
    if len(args.image_paths) == 1:
        axes = np.array([axes])

    for row, image_path in enumerate(args.image_paths):
        caption, original_img, cam_img = make_cam_overlay(
            image_path=image_path,
            blip_processor=blip_processor,
            blip_model=blip_model,
            cam_model=clip_model,
            target_layers=target_layers,
            device=device,
        )

        axes[row, 0].imshow(original_img)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Original: {os.path.basename(image_path)}")

        axes[row, 1].imshow(cam_img)
        axes[row, 1].axis("off")
        axes[row, 1].set_title("Grad-CAM from caption")
        axes[row, 1].text(
            0.5,
            -0.08,
            caption,
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="red",
            wrap=True,
        )

    plt.tight_layout()
    figure_path = os.path.join(args.output_dir, "caption_gradcam_compare.png")
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")

    print("Saved outputs:")
    print(f"- {figure_path}")


if __name__ == "__main__":
    main()

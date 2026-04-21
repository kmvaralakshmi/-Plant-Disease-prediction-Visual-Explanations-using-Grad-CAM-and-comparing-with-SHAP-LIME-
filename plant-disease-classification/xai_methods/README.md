# XAI Methods for Plant Disease Classification

This directory contains 4 complementary Explainable AI (XAI) methods for interpreting plant disease predictions:

## Methods Included

### 1. **Grad-CAM** (Existing in `/evaluate/grad_cam.py`)
- **Type:** Visual Saliency Maps
- **Output:** Heatmap showing which image regions influence predictions
- **Best For:** Understanding model attention patterns
- **Run:** `python ../evaluate/grad_cam.py`

### 2. **LIME** (`lime_explainer.py`)
- **Type:** Local Model-Agnostic Explanations
- **Output:** Segment importance scores with feature bar charts
- **Best For:** Understanding which image regions matter most
- **Run:** `python lime_explainer.py`

```bash
python lime_explainer.py
# Outputs to: ../../outputs/lime_results/
```

### 3. **SHAP** (`shap_explainer.py`)
- **Type:** Shapley Value-Based Feature Attribution
- **Output:** Patch importance heatmaps with distribution analysis
- **Best For:** Theoretically sound feature attribution
- **Run:** `python shap_explainer.py`

```bash
python shap_explainer.py
# Outputs to: ../../outputs/shap_results/
```

### 4. **VQA** (`vqa_system.py`)
- **Type:** Visual Question Answering with Spatial Reasoning
- **Questions:** 
  - "Is there damage on the left side?"
  - "Is there damage on the right side?"
  - "Is there damage in the center?"
  - "Is the disease severity high?"
  - "Is the disease widespread across the leaf?"
- **Output:** YES/NO answers with confidence scores + region visualization
- **Best For:** Agronomist-friendly spatial disease localization
- **Run:** `python vqa_system.py`

```bash
python vqa_system.py
# Outputs to: ../../outputs/vqa_results/
```

---

## Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install scikit-image scikit-learn
pip install matplotlib numpy opencv-python pillow
```

### Run All Methods

```bash
cd xai_methods

# Grad-CAM (existing)
python ../evaluate/grad_cam.py

# LIME
python lime_explainer.py

# SHAP
python shap_explainer.py

# VQA
python vqa_system.py
```

---

## Output Structure

```
outputs/
├── gradcam_results/        # Grad-CAM visualizations
├── lime_results/           # LIME explanations
├── shap_results/           # SHAP heatmaps
└── vqa_results/            # VQA spatial analysis
```

---

## Method Comparison

| Aspect | Grad-CAM | LIME | SHAP | VQA |
|--------|----------|------|------|-----|
| **Speed** | Fast | Medium | Slow | Fast |
| **Interpretability** | Visual | Local Rules | Theoretical | Spatial |
| **Model-Agnostic** | No | Yes | Yes | No |
| **Best For** | Visual patterns | Feature importance | Theory | Spatial reasoning |
| **Output** | Heatmap | Segments + Chart | Patches + Hist | Regions + Bars |

---

## Example Output

### Grad-CAM
Shows which pixels triggered the disease prediction with red highlighting important regions.

### LIME
Highlights 100 perturbed samples to find which image regions matter most using a local linear model.

### SHAP
Occlude 49 patches systematically and measure confidence drop to compute importance values.

### VQA
Analyzes 5 regions (left, right, center, top, bottom) and answers spatial questions about disease presence.

---

## Model Configuration

All methods use:
- **Base Model:** ResNet50 (ImageNet pretrained)
- **Classes:** 38 plant diseases
- **Input Size:** 224×224
- **Device:** CPU or GPU (auto-detected)

### Using Custom Trained Model

```python
from shap_explainer import PlantDiseaseExplainer

explainer = PlantDiseaseExplainer(
    model_path='../path/to/trained/model.pth'
)
```

---

## Dataset Requirements

- **Location:** `../PlantVillage_dataset/train/`
- **Structure:** `disease_name/image.JPG`
- **Classes:** 38 plant diseases
- **Supported:** JPG format

---

## Troubleshooting

### ImportError: No module named 'sklearn.segmentation'
```bash
# Fix: Use scikit-image instead
pip install scikit-image
```

### Model not found
Ensure model path is correct or use ImageNet pretrained weights (default).

### Dataset not found
Place PlantVillage dataset in: `plant-disease-classification/PlantVillage_dataset/`

---

## Future Enhancements

- [ ] Attention-based explanations
- [ ] Integrated Gradients
- [ ] DeepLIFT
- [ ] Influence functions
- [ ] Concept activation vectors (TCAV)

---

## References

- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02055)
- LIME: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- VQA: Custom implementation for spatial disease reasoning

---

## Author

Extended XAI methods package for plant disease classification.
Based on: https://github.com/HoFireMan/plant-disease-classification

## License

Same as parent project

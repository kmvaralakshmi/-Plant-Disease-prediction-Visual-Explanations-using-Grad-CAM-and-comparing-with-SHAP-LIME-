# Unified Multi-XAI Explainable Vision System (Jackfruit / Plant Disease)

## 1. Project Overview

### 1.1 Problem Statement and Introduction
Plant disease classification models can achieve good predictive performance, but practical adoption in agriculture depends on whether users can trust and interpret the model decision. This project uses a ResNet50-based classifier and integrates explanation methods to make the prediction process transparent for leaf images.

### 1.2 Why Interpretability Matters
In agriculture, false positives and false negatives can directly affect crop management cost and yield. Interpretability helps in three ways:
1. It validates whether the model attends to lesion regions instead of background artifacts.
2. It supports human experts in decision verification.
3. It reveals failure modes for model improvement.

### 1.3 The Trade-off Challenge
Deep CNN models provide strong representation learning, but they are often opaque. Simpler models are easier to explain but usually less accurate for complex visual disease patterns. The challenge is to keep CNN performance while improving explanation quality.

### 1.4 The Solution Gap
Many disease classification workflows report accuracy only. They do not provide localized, visual, and method-cross-validated evidence for why a class was predicted. This project addresses that gap by combining Grad-CAM, SHAP, and LIME in one pipeline.

### 1.5 Grad-CAM's Innovation (also LIME and SHAP)
Grad-CAM introduced class-specific localization maps using gradients from convolutional layers, showing where a model focuses for a target class. In this project:
1. Grad-CAM provides coarse localization heatmaps.
2. SHAP (patch occlusion) provides patch-level contribution analysis.
3. LIME provides segment-level local surrogate attribution.

### 1.6 Methodology Architecture
Pipeline architecture:
1. Input image preprocessing (resize to 224x224, normalize).
2. ResNet50 forward pass for prediction and top-k confidence.
3. Grad-CAM explanation generation from final conv block.
4. SHAP patch-occlusion attribution map generation.
5. LIME superpixel perturbation explanation generation.
6. Unified output export (images + JSON/TXT summary).

## 2. Implementation Details

### 2.1 Core Model
1. Backbone: ResNet50 (PyTorch).
2. Output classes: 38 PlantVillage classes.
3. Best checkpoint used: models/trained/resnet50_plantvillage_best.pth.
4. Reported subset accuracy: 75.92%.

### 2.2 Single File Execution Script
The end-to-end script is single_potato_xai.py. It performs:
1. Prediction on potato_disease.png.
2. Grad-CAM generation.
3. SHAP generation.
4. LIME generation.
5. Result packaging to potato_leaf_results.

### 2.3 Environment
1. OS: Windows.
2. Python: 3.11.
3. CPU execution supported; CUDA optional.

Install dependencies:

```powershell
pip install torch torchvision numpy pillow matplotlib opencv-python scikit-image scikit-learn
```

Run:

```powershell
cd D:\XAI_Orange_Jacfruit\plant-disease-classification
python single_potato_xai.py
```

## 3. Results and Output, and Comparison with Grad-CAM Paper

### 3.1 Potato Leaf Test Output (potato_disease.png)
From potato_leaf_results/prediction_summary.json:
1. Predicted class: Strawberry___healthy.
2. Confidence: 73.54%.
3. Top-5 includes Potato___Early_blight at rank 4 (4.73%).

Generated output files:
1. potato_leaf_results/gradcam_potato_disease.png
2. potato_leaf_results/shap_potato_disease.png
3. potato_leaf_results/lime_potato_disease.png
4. potato_leaf_results/prediction_summary.json
5. potato_leaf_results/prediction_summary.txt

### 3.2 Comparison with Grad-CAM Paper
Grad-CAM paper (Selvaraju et al., ICCV 2017) highlights class-discriminative localization without requiring bounding box supervision. Compared to that objective:
1. Similarity: Our Grad-CAM also localizes class-relevant regions on the leaf and provides visual evidence of model attention.
2. Similarity: The heatmap is coarse but semantically meaningful, consistent with Grad-CAM behavior in the original paper.
3. Difference: Prediction correctness is not guaranteed by localization quality; in our potato sample, the model attends to disease-like regions but still predicts a non-potato top class.
4. Practical interpretation: XAI verifies model reasoning path, but dataset/domain mismatch can still cause class-level errors.

## 5. Key Findings and Insights

### 5.1 Model Capability Discovery
1. The model can detect lesion patterns and damaged texture regions.
2. Cross-crop confusion exists in custom samples outside ideal class/domain conditions.
3. Presence of Potato___Early_blight in top-5 indicates partial disease-feature recognition despite top-1 mismatch.

### 5.2 Attribution Consistency
1. Grad-CAM highlights broad disease-related zones.
2. SHAP identifies high-impact local patches.
3. LIME marks influential superpixel segments.
4. All three methods indicate attribution concentration on symptomatic regions, increasing confidence in explanation consistency.

### 5.3 Comparison with Paper
1. The observed Grad-CAM map behavior is aligned with the paper's claim of class-discriminative localization.
2. As in the paper's broader observations, explanations should be used with prediction confidence and data context, not as standalone correctness proof.
3. Multi-XAI combination (Grad-CAM + SHAP + LIME) extends beyond the original paper by enabling attribution triangulation.

## 6. Conclusion
This project implements a unified explainability workflow over a ResNet50 disease classifier and demonstrates end-to-end interpretability on a potato leaf sample. The pipeline successfully generates prediction outputs and three complementary explanation views. Results show that explanation quality can be strong even when class prediction is imperfect, emphasizing the importance of both interpretability and domain-specific retraining for jackfruit-focused deployment.

For jackfruit-specific production use, the next required step is fine-tuning on a dedicated jackfruit disease dataset and updating class definitions accordingly.

## Credits
1. Grad-CAM: Selvaraju et al., ICCV 2017.
2. Local project implementations: Grad-CAM, SHAP-style patch occlusion, and LIME-style superpixel perturbation.

"""
Quick Training Script for Plant Disease Classification
Trains ResNet50 on subset of PlantVillage dataset (~1500 images)
Estimated time: 45-60 minutes on CPU
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

# ===== CONFIG =====
DATASET_ROOT = Path(r'D:\XAI_Orange_Jacfruit\PlantVillage_dataset\train')
OUTPUT_DIR = Path(__file__).parent / 'models' / 'trained'
BATCH_SIZE = 16  # Small batch for CPU
EPOCHS = 5  # Quick training: just 5 epochs
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0
SAMPLE_IMAGES_PER_CLASS = 40  # Only ~40 images per disease (~1500 total)

# ===== AUTO-DETECT DISEASE CLASSES =====
PLANT_DISEASES = sorted([d.name for d in DATASET_ROOT.iterdir() if d.is_dir()])

print(f"🔧 Configuration:")
print(f"   Dataset: {DATASET_ROOT}")
print(f"   Detected classes: {len(PLANT_DISEASES)}")
print(f"   Device: {DEVICE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Images per class: {SAMPLE_IMAGES_PER_CLASS}")
print(f"   Estimated time: 45-60 minutes\n")
print(f"   Device: {DEVICE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Images per class: {SAMPLE_IMAGES_PER_CLASS}")
print(f"   Estimated time: 45-60 minutes\n")

# ===== DATASET CLASS =====
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataset_root, disease_classes, sample_limit=None, transform=None):
        self.disease_classes = disease_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(disease_classes)}
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Collect images (with sample limit)
        for disease_idx, disease in enumerate(disease_classes):
            disease_path = Path(dataset_root) / disease
            if not disease_path.exists():
                print(f"⚠️  {disease_path} not found, skipping...")
                continue
            
            images = list(disease_path.glob('*.JPG')) + list(disease_path.glob('*.jpg'))
            
            # Limit samples per class
            if sample_limit:
                images = images[:sample_limit]
            
            self.image_paths.extend(images)
            self.labels.extend([disease_idx] * len(images))
            print(f"✅ {disease}: {len(images)} images")
        
        print(f"\n📊 Total images: {len(self.image_paths)}\n")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# ===== SETUP =====
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

print("📥 Loading dataset...")
dataset = PlantDiseaseDataset(DATASET_ROOT, PLANT_DISEASES, 
                              sample_limit=SAMPLE_IMAGES_PER_CLASS, 
                              transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=NUM_WORKERS)

# ===== MODEL =====
print("\n🏗️  Building model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, len(PLANT_DISEASES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

print(f"✅ Model: ResNet50 | Output Classes: {len(PLANT_DISEASES)} | Device: {DEVICE}\n")

# ===== TRAINING =====
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


print("🚀 Starting training...\n")
start_time = time.time()

best_accuracy = 0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, dataloader, criterion, optimizer, DEVICE)
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"   Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    
    if train_acc > best_accuracy:
        best_accuracy = train_acc
        # Save best model
        best_model_path = OUTPUT_DIR / 'resnet50_plantvillage_best.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': train_acc,
            'num_classes': len(PLANT_DISEASES)
        }, best_model_path)
        print(f"   ✅ Best model saved to {best_model_path}")

# Save final model
final_model_path = OUTPUT_DIR / 'resnet50_plantvillage_final.pth'
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': train_acc,
    'num_classes': len(PLANT_DISEASES)
}, final_model_path)

elapsed = time.time() - start_time
print(f"\n✅ Training complete!")
print(f"   Time: {elapsed/60:.1f} minutes")
print(f"   Best Accuracy: {best_accuracy:.2f}%")
print(f"   Model saved to: {final_model_path}")
print(f"   Best model saved to: {best_model_path}")

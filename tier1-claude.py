# Tier 1 - Improved Deep CNN for Helmet Detection
import os
import cv2
import numpy as np
import pandas as pd
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================================================
# 1. Load images manually (same logic as your TF version)
# =========================================================
def load_data(directory, size=(224, 224)):
    images, labels = [], []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(directory, filename)
            label_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(label_path):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                images.append(img)

                with open(label_path, 'r') as f:
                    label = int(f.readline().split()[0])
                    labels.append(label)

    return np.array(images), np.array(labels)

BASE_PATH = './dataset'
X_train, y_train = load_data(os.path.join(BASE_PATH, 'train'))
X_test, y_test = load_data(os.path.join(BASE_PATH, 'test'))
print("Train:", X_train.shape, " Test:", X_test.shape)

# Create validation split from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
print("After split - Train:", X_train.shape, " Val:", X_val.shape)

# =========================================================
# 2. PyTorch Dataset with Augmentation
# =========================================================
class HelmetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================================================
# 3. Build model with better architecture
# =========================================================
def build_model(model_name='resnet50', num_classes=1, dropout=0.4):
    """
    Build a transfer learning model with customizable backbone.
    Supports: resnet50, resnet101, efficientnet_b0, mobilenet_v2
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace final classifier with custom head
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    return model

# =========================================================
# 4. Progressive unfreezing strategies
# =========================================================
def unfreeze_layers(model, strategy='gradual', stage=1):
    """
    Progressive unfreezing strategies:
    - 'gradual': Unfreeze from top to bottom progressively
    - 'block': Unfreeze entire blocks (ResNet specific)
    - 'all': Unfreeze all layers
    """
    if strategy == 'block' and hasattr(model, 'layer4'):
        # Stage 1: Only classifier
        # Stage 2: layer4 (top block)
        # Stage 3: layer4 + layer3
        if stage == 2:
            for param in model.layer4.parameters():
                param.requires_grad = True
            print("Unfrozen: layer4")
        elif stage == 3:
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True
            print("Unfrozen: layer4 + layer3")
        elif stage == 4:
            for param in model.parameters():
                param.requires_grad = True
            print("Unfrozen: All layers")
    
    elif strategy == 'gradual' and hasattr(model, 'layer4'):
        # Gradually unfreeze layer4 parameters
        if stage == 2:
            layers = list(model.layer4.parameters())
            for p in layers[-10:]:  # Last 10 params
                p.requires_grad = True
            print("Unfrozen: layer4 (partial)")
        elif stage == 3:
            for param in model.layer4.parameters():
                param.requires_grad = True
            print("Unfrozen: layer4 (full)")
    
    elif strategy == 'all':
        for param in model.parameters():
            param.requires_grad = True
        print("Unfrozen: All layers")
    
    return model

# =========================================================
# 5. Enhanced training function with validation
# =========================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, epochs=5, device='cpu', early_stopping_patience=3):
    """
    Train model with validation monitoring and early stopping
    """
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"  Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
    
    return model

# =========================================================
# 6. Prediction function
# =========================================================
def predict(model, data_loader, device='cpu'):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
    
    return np.array(all_probs)

# =========================================================
# 7. MAIN TRAINING LOOP — 20 RUNS
# =========================================================
ITERATIONS = 20
results = []

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

for run in range(ITERATIONS):
    print(f"\n{'='*60}")
    print(f"RUN {run+1}/{ITERATIONS}")
    print(f"{'='*60}")
    start_time = time.time()
    
    # Create datasets
    train_dataset = HelmetDataset(X_train, y_train, transform=train_transform)
    val_dataset = HelmetDataset(X_val, y_val, transform=val_transform)
    test_dataset = HelmetDataset(X_test, y_test, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Build model
    model = build_model(model_name='resnet50', dropout=0.4)
    criterion = nn.BCEWithLogitsLoss()
    
    # ---- STAGE 1: Train classifier only (head) ----
    print("\n[STAGE 1] Training classifier head only...")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=2)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       scheduler, epochs=5, device=device, early_stopping_patience=3)
    
    # ---- STAGE 2: Unfreeze top layers (layer4) ----
    print("\n[STAGE 2] Fine-tuning layer4...")
    model = unfreeze_layers(model, strategy='block', stage=2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=2)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       scheduler, epochs=5, device=device, early_stopping_patience=3)
    
    # ---- STAGE 3: Unfreeze more layers (layer3 + layer4) ----
    print("\n[STAGE 3] Fine-tuning layer3 + layer4...")
    model = unfreeze_layers(model, strategy='block', stage=3)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=2)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                       scheduler, epochs=3, device=device, early_stopping_patience=2)
    
    # ---- Final Predictions on Test Set ----
    print("\n[EVALUATION] Testing on test set...")
    y_prob = predict(model, test_loader, device=device)
    y_pred = (y_prob > 0.5).astype(int)
    
    # ---- Metrics ----
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate BCE loss on test set
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    y_prob_t = torch.tensor(y_prob, dtype=torch.float32)
    loss = criterion(y_prob_t, y_test_t).item()
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0
    
    # ---- System metrics ----
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    disco = psutil.disk_usage('/').percent
    run_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"RESULTS - RUN {run+1}")
    print(f"{'='*60}")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall (Pos): {rec_pos:.4f}")
    print(f"Recall (Neg): {rec_neg:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"Loss:         {loss:.4f}")
    print(f"AUC:          {auc:.4f}")
    print(f"CPU Usage:    {cpu:.1f}%")
    print(f"RAM Usage:    {mem:.1f}%")
    print(f"Disk Usage:   {disco:.1f}%")
    print(f"Exec Time:    {run_time:.2f}s")
    print(f"{'='*60}")
    
    results.append([acc, prec, rec_pos, rec_neg, f1, loss, auc, cpu, mem, disco, run_time])

# =========================================================
# 8. SUMMARY STATISTICS
# =========================================================
results = np.array(results)
mean_vals = results.mean(axis=0)
std_vals = results.std(axis=0)
min_vals = results.min(axis=0)
max_vals = results.max(axis=0)

metrics_labels = [
    "Accuracy", "Precision", "Recall+", "Recall-", "F1-Score",
    "Loss", "AUC", "CPU %", "RAM %", "Disk %", "Exec Time (s)"
]

print("\n" + "="*80)
print("FINAL SUMMARY (20 RUNS)")
print("="*80)
print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-"*80)

for i, metric in enumerate(metrics_labels):
    print(f"{metric:<15} {mean_vals[i]:<12.4f} {std_vals[i]:<12.4f} "
          f"{min_vals[i]:<12.4f} {max_vals[i]:<12.4f}")

print("="*80)

# =========================================================
# 9. Save results to CSV
# =========================================================
df_results = pd.DataFrame(results, columns=metrics_labels)
df_results.to_csv('helmet_detection_results.csv', index=False)
print("\nResults saved to 'helmet_detection_results.csv'")

# Save summary statistics
summary_dict = {
    'Metric': metrics_labels,
    'Mean': mean_vals,
    'Std': std_vals,
    'Min': min_vals,
    'Max': max_vals
}
df_summary = pd.DataFrame(summary_dict)
df_summary.to_csv('helmet_detection_summary.csv', index=False)
print("Summary saved to 'helmet_detection_summary.csv'")
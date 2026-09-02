import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import time
import psutil
import argparse

# ============================================================
# 1. Dataset con YOLO (una sola línea)
# ============================================================
class HelmetDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [f for f in os.listdir(folder) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        image_path = os.path.join(self.folder, filename)
        txt_path = image_path.replace(".jpg", ".txt")
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(txt_path, "r") as f:
            line = f.readline().strip()
        cls = int(line.split()[0])  # 0 casco, 1 sin casco
        return img, torch.tensor(cls, dtype=torch.float32)

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Model definition
class HelmetCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, dense_units=128):
        super(HelmetCNN, self).__init__()
        # Stage 1: 3->32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 2: 32->64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Stage 3: Parallel 3x3 and 5x5, each to 64, concat to 128
        self.stage3_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.stage3_5x5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.stage3_bn = nn.BatchNorm2d(128)
        self.stage3_relu = nn.ReLU()
        self.stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Stage 4: Parallel 3x3 and 5x5, each to 128, concat to 256
        self.stage4_3x3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.stage4_5x5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.stage4_bn = nn.BatchNorm2d(256)
        self.stage4_relu = nn.ReLU()
        self.stage4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # Stage 3
        x3 = self.stage3_3x3(x)
        x5 = self.stage3_5x5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.stage3_bn(x)
        x = self.stage3_relu(x)
        x = self.stage3_pool(x)
        # Stage 4
        x3 = self.stage4_3x3(x)
        x5 = self.stage4_5x5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.stage4_bn(x)
        x = self.stage4_relu(x)
        x = self.stage4_pool(x)
        # GAP
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(1)

# Data augmentations and transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training phase function
def train_phase(model, optimizer, loader_train, loader_val, criterion, epochs, patience, scheduler=None, device='cpu'):
    es = EarlyStopping(patience=patience)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(loader_train.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in loader_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(loader_val.dataset)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        es(val_loss)
        if es.early_stop:
            print("Early stopping triggered")
            break
        if scheduler:
            scheduler.step()
    return val_loss

# Single run training and evaluation
def run_training_once(run, train_folder, val_folder, device="cpu"):
    torch.manual_seed(run)
    # Hyperparameters justification:
    # - Dropout rate: 0.5 (Standard value for classification heads in CNNs to prevent overfitting without excessively slowing convergence)
    # - Learning rate: Phase 1 = 1e-3 (Suitable for training the classification head from scratch; allows quick adaptation)
    # - Phase 2 = 1e-4 (10x smaller to fine-tune the entire model gently, preserving learned features)
    # - Dense layer units: 128 (Provides a gradual reduction from 256 features to 1 output, allowing sufficient capacity for non-linear decision boundary without overparameterization)
    # - Batch size: 32 (Balances memory usage and training speed on CPU; larger batches may exceed memory, smaller slow down training)
    model = HelmetCNN(dropout_rate=0.5, dense_units=128).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Standard for binary classification without class weighting

    # Datasets and Loaders
    train_dataset = HelmetDataset(train_folder, transform=train_transform)
    val_dataset = HelmetDataset(val_folder, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Phase 1: Freeze conv layers, train head
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    print("Phase 1: Training classification head")
    _ = train_phase(model, optimizer, train_loader, val_loader, criterion, epochs=15, patience=5, device=device)

    # Phase 2: Unfreeze all, fine-tune with CosineAnnealingLR
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    print("Phase 2: Fine-tuning entire model")
    final_val_loss = train_phase(model, optimizer, train_loader, val_loader, criterion, epochs=20, patience=7, scheduler=scheduler, device=device)

    # Evaluation on test set
    # L = -(1/N)*sum[ y*log(p) + (1-y)*log(1-p) ]
    model.eval()
    test_criterion = nn.BCELoss(reduction='sum')
    preds, probs, labels_all = [], [], []
    total_test_loss = 0.0
    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            prob = torch.sigmoid(outputs)
            prob_clamped = torch.clamp(prob, 1e-7, 1 - 1e-7)
            labels_dev = labels.to(device)
            total_test_loss += test_criterion(
                prob_clamped.squeeze(), labels_dev
            ).item()
            pred = (prob.squeeze().cpu().numpy() > 0.5).astype(float)
            preds.extend(pred if pred.ndim > 0 else [pred.item()])
            probs.extend(prob.squeeze().cpu().numpy().flatten())
            labels_all.extend(labels.cpu().numpy())
    total_time = time.time() - start_time
    # Divide by N → mean BCE
    test_loss = total_test_loss / len(labels_all)
    num_images = len(labels_all)
    latency_ms = (total_time / num_images) * 1000
    print(f"Inference latency: {latency_ms:.2f} ms/image (Target: <30 ms)")

    acc = accuracy_score(labels_all, preds)
    prec = precision_score(labels_all, preds, zero_division=0)
    recall_pos = recall_score(labels_all, preds, pos_label=1, zero_division=0)
    recall_neg = recall_score(labels_all, preds, pos_label=0, zero_division=0)
    auc = roc_auc_score(labels_all, probs)
    cm = confusion_matrix(labels_all, preds).tolist()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_percent = psutil.virtual_memory().percent

    row = [
        run,
        acc, prec, recall_pos, recall_neg,
        auc, test_loss,
        cm, total_time,
        cpu_percent, ram_percent
    ]
    return row

# ============================================================
# 7. Múltiples experimentos
# ============================================================
def run_experiments(train_folder, val_folder, n_runs=8, device="cpu"):
    results = []
    for r in range(1, n_runs + 1):
        print(f"\n{'='*50}\n RUN {r}/{n_runs}\n{'='*50}")
        row = run_training_once(r, train_folder, val_folder, device)
        results.append(row)
        print(f" → acc={row[1]:.4f} | loss={row[6]:.4f}")
    df = pd.DataFrame(results, columns=[
        "run",
        "accuracy", "precision", "recall_pos", "recall_neg",
        "auc", "loss",
        "confusion_matrix", "time_sec",
        "cpu_percent", "ram_percent"
    ])
    df.to_csv("tier3-grok_results.csv", index=False)
    print("✔ tier3-grok_results.csv generado")
    summary = df.describe().transpose()[["mean", "std", "min", "max"]]
    summary.to_csv("tier3-grok_summary.csv")
    print("✔ tier3-grok_summary.csv generado")
    return df, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train helmet detection model")
    parser.add_argument("--train_folder", type=str, default="dataset/train", help="Path to training data folder")
    parser.add_argument("--val_folder", type=str, default="dataset/test", help="Path to validation data folder")
    parser.add_argument("--n_runs", type=int, default=8, help="Number of runs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    run_experiments(args.train_folder, args.val_folder, args.n_runs, args.device)
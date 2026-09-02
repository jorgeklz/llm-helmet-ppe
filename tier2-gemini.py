# =========================================================
# Helmet Detection – Optimized Pipeline (Gemini Edition)
# PyTorch Dataset/DataLoader approach (Memory Efficient)
# θ = [d=0.5, η1=1e-3, η2=5e-5, b=32, u=512]
# =========================================================

import os
import cv2
import copy
import numpy as np
import pandas as pd
import time
import psutil

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

BASE_PATH  = './dataset'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 1. Data loading (Adaptado para tus subcarpetas)
# =========================================================
def load_data(directory, size=(224, 224)):
    images, labels = [], []
    if not os.path.exists(directory):
        return np.array(images), np.array(labels)
        
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path   = os.path.join(directory, filename)
            label_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(label_path):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                images.append(img)
                with open(label_path, 'r') as f:
                    labels.append(int(f.readline().split()[0]))
    return np.array(images), np.array(labels)

# =========================================================
# 2. PyTorch Dataset Wrapper (Eficiente en RAM)
# =========================================================
class HelmetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# =========================================================
# 3. Model Definition & Training Logic
# =========================================================
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Fase 1: Congelar backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Reemplazar la capa de clasificación (u=512, d=0.5)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1) # Salida binaria
    )
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, patience=8, phase_name=""):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # --- TRAINING ---
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend((probs > 0.5).astype(int))
                val_trues.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_trues, val_preds)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  [{phase_name}] Epoch {epoch+1:03d}/{num_epochs} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | lr: {current_lr:.2e}")
        
        # Scheduler & Early Stopping logic
        if scheduler is not None:
            # ReduceLROnPlateau usa validation loss por defecto para minimizar
            scheduler.step(val_loss)
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{phase_name}] Early stopping triggered.")
                break
                
    model.load_state_dict(best_model_wts)
    return model, best_loss

# =========================================================
# 4. Single Run Execution
# =========================================================
def run_training_once(run_id):
    t_start = time.time()
    
    # Cargar datos desde subcarpetas
    train_dir = os.path.join(BASE_PATH, 'train')
    test_dir = os.path.join(BASE_PATH, 'test')
    
    X_train_full, y_train_full = load_data(train_dir)
    X_test, y_test = load_data(test_dir)
    
    if len(X_train_full) == 0 or len(X_test) == 0:
        raise FileNotFoundError(f"Faltan datos. Verifica las rutas:\n- {train_dir}\n- {test_dir}")
    
    # Crear set de validación a partir del train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=run_id
    )
    
    # Transformaciones con Data Augmentation básico para train
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # DataLoaders (b=32)
    b = 32
    train_loader = DataLoader(HelmetDataset(X_train, y_train, train_transforms), batch_size=b, shuffle=True)
    val_loader   = DataLoader(HelmetDataset(X_val, y_val, val_test_transforms), batch_size=b, shuffle=False)
    test_loader  = DataLoader(HelmetDataset(X_test, y_test, val_test_transforms), batch_size=b, shuffle=False)
    
    model = create_model()
    criterion = nn.BCEWithLogitsLoss()
    
    # --- FASE 1: Entrenar solo la cabeza (LR = 1e-3) ---
    optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=1e-3)
    model, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer_phase1, 
        scheduler=None, num_epochs=10, patience=3, phase_name="P1"
    )
    
    # --- FASE 2: Fine-tuning de todo el modelo (LR = 5e-5) ---
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_phase2 = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_phase2, mode='min', factor=0.5, patience=3)
    model, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer_phase2, 
        scheduler, num_epochs=30, patience=8, phase_name="P2"
    )
    
    # --- EVALUACIÓN EN TEST ---
    # Use BCELoss on sigmoid probabilities to match the paper definition:
    # L = -(1/N)*sum[ y*log(p) + (1-y)*log(1-p) ]
    test_criterion = nn.BCELoss(reduction='sum')  # sum over batch; divide by N below
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            probs_clamped = torch.clamp(probs, 1e-7, 1 - 1e-7)
            loss = test_criterion(probs_clamped, labels)
            test_loss += loss.item()

            preds = (probs.cpu().numpy() > 0.5).astype(int)
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    # Divide by total number of test samples → mean BCE
    test_loss = test_loss / len(test_loader.dataset)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_probs = np.array(y_probs).flatten()
    
    # Cálculo de métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0
    
    # ── Runtime metrics ────────────────────────────────────
    cpu_end  = psutil.cpu_percent(interval=None)
    ram_end  = psutil.virtual_memory().percent
    exec_sec = time.time() - t_start

    return [
        run_id, acc, prec, rec_pos, rec_neg,
        auc, test_loss,
        cpu_end, ram_end, exec_sec
    ]

# =========================================================
# 5. Multi-run experiment + CSV summary
# =========================================================
def run_experiments(n_runs=20):
    results = []
    for r in range(1, n_runs + 1):
        print(f"\n{'='*50}")
        print(f"  RUN {r}/{n_runs}")
        print(f"{'='*50}")
        row = run_training_once(r)
        results.append(row)
        print(f"  → acc={row[1]:.4f} | auc={row[5]:.4f}")

    df = pd.DataFrame(results, columns=[
        "run",
        "accuracy", "precision", "recall_pos", "recall_neg",
        "auc", "loss",
        "cpu_percent", "ram_percent", "time_sec"
    ])

    df.to_csv("tier2-gemini_results.csv", index=False)
    print("\n✔ tier2-gemini_results.csv generated")

    summary = df.describe().transpose()[["mean", "std", "min", "max"]]
    summary.to_csv("tier2-gemini_summary.csv")
    print("✔ tier2-gemini_summary.csv generated")
    
    return df, summary

if __name__ == '__main__':
    # Inicialización de psutil para lecturas precisas
    psutil.cpu_percent(interval=None) 
    df, summary = run_experiments(n_runs=20)
    print("\n── Summary ──")
    print(summary.to_string())
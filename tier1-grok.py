import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import psutil
import time
import os
import cv2
from PIL import Image

# =========================================================
# 1. LOAD IMAGES MANUALLY
# =========================================================
def load_data(directory, size=(224, 224)):
    print(f"Cargando imágenes de: {directory} ...")
    images, labels = [], []
    if not os.path.exists(directory):
        print(f"ERROR: El directorio {directory} no existe.")
        return np.array([]), np.array([])
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(directory, filename)
            label_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(label_path):
                # Leer imagen
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                images.append(img)
                # Leer etiqueta
                with open(label_path, 'r') as f:
                    try:
                        label = int(f.readline().split()[0])
                        labels.append(label)
                    except (ValueError, IndexError):
                        pass
    return np.array(images), np.array(labels)

BASE_PATH = './dataset'
print("--- Iniciando Carga de Datos ---")
X_train, y_train = load_data(os.path.join(BASE_PATH, 'train'))
X_test, y_test = load_data(os.path.join(BASE_PATH, 'test'))
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
if len(X_train) == 0:
    raise ValueError("No se cargaron imágenes de entrenamiento. Revisa la ruta './dataset/train'")

# =========================================================
# 2. CONFIGURACIÓN PYTORCH
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_RUNS = 20
print(f"Usando dispositivo: {DEVICE}")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Personalizado para conectar tus arrays NumPy con PyTorch
class HelmetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # Convertir array RGB a PIL Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Crear test_loader (fijo)
test_dataset = HelmetDataset(X_test, y_test, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 3. FUNCIONES DE ENTRENAMIENTO Y PREDICCIÓN
# =========================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)  # [batch, 1] para BCE
        optimizer.zero_grad()
        outputs = model(inputs)  # Logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def val_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)

def train_progressive(X_train_fold, y_train_fold, X_val_fold, y_val_fold):
    # Datasets y Loaders
    train_dataset = HelmetDataset(X_train_fold, y_train_fold, transform=train_transform)
    val_dataset = HelmetDataset(X_val_fold, y_val_fold, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Salida binaria para BCE
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    # Fase 1: Entrenar solo la cabeza (congelado)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    num_epochs_phase1 = 10
    for epoch in range(num_epochs_phase1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = val_epoch(model, val_loader, criterion)
        # No imprimir por epoch para matching, pero se puede descomentar si se quiere
        # print(f"Phase 1 Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Fase 2: Descongelar y fine-tune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs_phase2 = 15
    for epoch in range(num_epochs_phase2):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = val_epoch(model, val_loader, criterion)
        scheduler.step()
        # No imprimir por epoch
        # print(f"Phase 2 Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            probs.extend(outputs)
    return np.array(probs)

# =========================================================
# 4. BUCLE PRINCIPAL (20 RUNS)
# =========================================================
results = []
metrics_labels = [
    "Accuracy", "Precision", "Recall+", "Recall-", "F1-Score",
    "Loss", "AUC", "CPU %", "RAM %", "Disk %", "Exec Time (s)"
]
test_criterion = nn.BCELoss(reduction='mean')  # L = -(1/N)*sum[y*log(p)+(1-y)*log(1-p)]
print(f"\nIniciando evaluación de {NUM_RUNS} corridas...")
for run in range(NUM_RUNS):
    start_time = time.time()
    # Split Train/Val (Shuffle en cada corrida)
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=run
    )
    print(f"\n[RUN {run+1}/{NUM_RUNS}] Training...")
    model = train_progressive(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    print(f"[RUN {run+1}/{NUM_RUNS}] Testing...")
    y_prob = predict(model, test_loader, device=DEVICE)
    y_pred = (y_prob > 0.5).astype(int)
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # Loss en Test — clamp to (eps, 1-eps) to avoid log(0)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    y_prob_t = torch.clamp(torch.tensor(y_prob, dtype=torch.float32), 1e-7, 1 - 1e-7)
    loss_val = test_criterion(y_prob_t, y_test_t).item()
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0
    # Métricas de Sistema
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    disco = psutil.disk_usage('/').percent
    run_time = time.time() - start_time
    print(f"--> Run {run+1} Result: Acc={acc:.4f} | Loss={loss_val:.4f} | Time={run_time:.2f}s")
    results.append([acc, prec, rec_pos, rec_neg, f1, loss_val, auc, cpu, mem, disco, run_time])

# =========================================================
# 5. ESTADÍSTICAS FINALES Y GUARDADO
# =========================================================
results = np.array(results)
mean_vals = results.mean(axis=0)
std_vals = results.std(axis=0)
min_vals = results.min(axis=0)
max_vals = results.max(axis=0)
print("\n" + "="*80)
print(f"FINAL SUMMARY ({NUM_RUNS} RUNS)")
print("="*80)
print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-"*80)
for i, metric in enumerate(metrics_labels):
    print(f"{metric:<15} {mean_vals[i]:<12.4f} {std_vals[i]:<12.4f} "
          f"{min_vals[i]:<12.4f} {max_vals[i]:<12.4f}")
print("="*80)
# CSV Detallado
df_results = pd.DataFrame(results, columns=metrics_labels)
df_results.to_csv('tier1-grok_results.csv', index=False)
print("\nResults saved to 'tier1-grok_results.csv'")
# CSV Resumen
summary_dict = {
    'Metric': metrics_labels,
    'Mean': mean_vals,
    'Std': std_vals,
    'Min': min_vals,
    'Max': max_vals
}
df_summary = pd.DataFrame(summary_dict)
df_summary.to_csv('tier1-grok_summary.csv', index=False)
print("Summary saved to 'tier1-grok_summary.csv'")
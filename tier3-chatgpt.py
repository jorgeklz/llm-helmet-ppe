import os
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


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


# ============================================================
# 2. Data Augmentation
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# 3. ResNet18 ajustada para BCELoss (1 salida + Sigmoid)
# ============================================================
class HelmetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Necesario para BCELoss
        )

    def forward(self, x):
        return self.resnet(x)


# ============================================================
# 4. Entrenamiento con BCELoss
# ============================================================
def train_model(model, train_loader, class_weights=None, device="cpu"):
    model.to(device)
    criterion = nn.BCELoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = np.inf
    patience = 5
    stop_counter = 0

    print("\n===== FASE DE ENTRENAMIENTO =====\n")

    for epoch in range(20):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # BCELoss requiere shape [batch,1]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("⏸ Early stopping activado")
                break

    return model


# ============================================================
# 5. Evaluación con BCELoss
# ============================================================
def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    criterion = nn.BCELoss()

    total_loss = 0
    count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            probs = outputs.cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            labels_np = labels.cpu().numpy().flatten()

            y_true.extend(labels_np)
            y_pred.extend(preds)
            y_prob.extend(probs)

    avg_loss = total_loss / count
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec_pos, rec_neg, auc, avg_loss, cm


# ============================================================
# 6. Una corrida completa
# ============================================================
def run_training_once(run_id, device="cpu"):
    train_dataset = HelmetDataset("dataset/train", transform=train_transform)
    test_dataset = HelmetDataset("dataset/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = HelmetResNet()

    # Pesos de clase inversos
    labels_list = [label.item() for _, label in train_loader.dataset]
    class_counts = np.bincount(labels_list)
    total = sum(class_counts)

    weight_pos = total / class_counts[1]
    weight_neg = total / class_counts[0]

    # BCELoss solo acepta 1 peso → creamos un vector según etiqueta
    class_weights = torch.tensor([weight_pos], dtype=torch.float32).to(device)

    # Entrenar
    start_time = time.time()
    model = train_model(model, train_loader, class_weights, device)
    end_time = time.time()

    model.load_state_dict(torch.load("best_model.pth"))

    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent

    # Evaluar
    acc, prec, rec_pos, rec_neg, auc, loss, cm = evaluate_model(
        model, test_loader, device
    )

    return [
        run_id, acc, prec, rec_pos, rec_neg,
        auc, loss, cm.tolist(),
        end_time - start_time, cpu_percent, ram_percent
    ]


# ============================================================
# 7. Múltiples experimentos
# ============================================================
def run_experiments(n_runs=20, device="cpu"):
    results = []

    for r in range(1, n_runs + 1):
        print(f"\n{'='*50}\n RUN {r}/{n_runs}\n{'='*50}")
        row = run_training_once(r, device)
        results.append(row)
        print(f" → acc={row[1]:.4f} | loss={row[6]:.4f}")

    df = pd.DataFrame(results, columns=[
        "run",
        "accuracy", "precision", "recall_pos", "recall_neg",
        "auc", "loss",
        "confusion_matrix", "time_sec",
        "cpu_percent", "ram_percent"
    ])

    df.to_csv("tier3-chatgpt_results.csv", index=False)
    print("✔ tier3-chatgpt_results.csv generado")

    summary = df.describe().transpose()[["mean", "std", "min", "max"]]
    summary.to_csv("tier3-chatgpt_summary.csv")
    print("✔ tier3-chatgpt_summary.csv generado")

    return df, summary


# ============================================================
# 8. MAIN
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, summary = run_experiments(20, device)
    print(summary)
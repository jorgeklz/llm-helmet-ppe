# =========================================================
# Helmet Detection – Full Pipeline (Optimized Multi-Run Version)
# =========================================================

import os
import cv2
import numpy as np
import pandas as pd
import time
import psutil

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


# =========================================================
# 1. Load images manually (OpenCV + TXT labels)
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


# =========================================================
# GLOBAL PATH
# =========================================================
BASE_PATH = './dataset'


# =========================================================
# 2. Transforms
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def apply_transform(X):
    return torch.stack([transform(img) for img in X])


# =========================================================
# 3. Build model – Your custom fine-tunable ResNet50
# =========================================================
def build_model(dropout=0.3, units=256):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False  # Phase 1 frozen

    in_feats = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(units, 1),
        nn.Sigmoid()
    )

    return model


# =========================================================
# 4. Metrics
# =========================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    return acc, prec, rec_pos, rec_neg, f1, auc


# =========================================================
# 5. Manual batch generator
# =========================================================
def get_batches(X, y, batch_size=32):
    idx = torch.randperm(len(X))
    X = X[idx]
    y = y[idx]

    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


# =========================================================
# 6. Training loop
# =========================================================
def run_training_once(run_id,
                      dropout=0.3, units=256, batch_size=32,
                      lr1=1e-3, lr2=1e-4,
                      epochs_phase1=5, epochs_phase2=10):

    # -------------------------- LOAD DATA --------------------------
    X_train, y_train = load_data(os.path.join(BASE_PATH, 'train'))
    X_test, y_test   = load_data(os.path.join(BASE_PATH, 'test'))

    X_train_t = apply_transform(X_train)
    X_test_t  = apply_transform(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    # -------------------------- SETUP --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(dropout, units).to(device)
    criterion = nn.BCELoss(reduction='mean')  # L = -(1/N)*sum[y*log(p)+(1-y)*log(1-p)]

    # stats
    cpu_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().used / (1024**2)
    t_start = time.time()

    # -------------------------- PHASE 1 --------------------------
    optimizer = optim.Adam(model.fc.parameters(), lr=lr1)

    for _ in range(epochs_phase1):
        for batch_x, batch_y in get_batches(X_train_t, y_train_t, batch_size):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # -------------------------- PHASE 2 --------------------------
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr2)

    for _ in range(epochs_phase2):
        for batch_x, batch_y in get_batches(X_train_t, y_train_t, batch_size):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # -------------------------- EVALUATION --------------------------
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in get_batches(X_test_t, y_test_t, batch_size):
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()

            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            y_true.extend(batch_y.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    acc, prec, rec_pos, rec_neg, f1, auc = compute_metrics(y_true, y_pred, y_prob)
    # clamp to (eps, 1-eps) to avoid log(0) in BCE
    test_loss = criterion(
        torch.clamp(torch.tensor(y_prob), 1e-7, 1 - 1e-7),
        torch.tensor(y_true)
    ).item()

    # -------------------------- RUNTIME METRICS --------------------------
    cpu_end = psutil.cpu_percent(interval=None)
    ram_end = psutil.virtual_memory().used / (1024**2)
    exec_time = time.time() - t_start

    return [
        run_id, acc, prec, rec_pos, rec_neg,
        f1, auc, test_loss,
        cpu_end, ram_end - ram_start, exec_time
    ]


# =========================================================
# 7. Multi-run experiment + summary
# =========================================================
def run_experiments(n_runs=20):
    results = []

    for r in range(1, n_runs + 1):
        print(f"\n=== RUN {r}/{n_runs} ===")
        row = run_training_once(r)
        results.append(row)

    df = pd.DataFrame(results, columns=[
        "run",
        "accuracy","precision","recall_pos","recall_neg",
        "f1","auc","loss",
        "cpu_percent","ram_MB","time_sec"
    ])

    df.to_csv("tier2-chatgpt_results.csv", index=False)
    print("\n✔ results.csv generado")

    # SUMMARY
    summary = df.describe().transpose()[["mean","std","min","max"]]
    summary.to_csv("tier2-chatgpt_summary.csv")
    print("✔ summary.csv generado")

    return df, summary


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    df, summary = run_experiments(n_runs=20)

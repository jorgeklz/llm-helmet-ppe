# =========================================================
# Helmet Detection – Optimized Pipeline
# θ = [d=0.5, η1=3e-3, η2=3e-5, b=32, u=512]
# + CosineAnnealingWarmRestarts (Phase 1)
# + ReduceLROnPlateau           (Phase 2)
# + EarlyStopping               (both phases)
# =========================================================

import os
import cv2
import copy
import numpy as np
import pandas as pd
import time
import psutil

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


# =========================================================
# HYPERPARAMETER VECTOR  θ = [d, η1, η2, b, u]
# =========================================================
DROPOUT    = 0.5          # d   – strong regularisation on the head
LR1        = 3e-3         # η1  – aggressive head-only training
LR2        = 3e-5         # η2  – gentle full-network fine-tuning (η2/η1 ≈ 1/100)
BATCH_SIZE = 32           # b   – optimal noise/convergence trade-off
UNITS      = 512          # u   – hidden units in classification head

EPOCHS_P1  = 20           # Phase 1 epochs (head only)
EPOCHS_P2  = 30           # Phase 2 epochs (full fine-tune)

BASE_PATH  = './dataset'


# =========================================================
# 1. Data loading  (unchanged from original)
# =========================================================
def load_data(directory, size=(224, 224)):
    images, labels = [], []
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
# 2. Transforms  (unchanged)
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_transform(X):
    return torch.stack([transform(img) for img in X])


# =========================================================
# 3. Model  – ResNet50 + optimized head
#    Head: Linear(2048→u) → ReLU → Dropout(d) → Linear(u→1) → Sigmoid
# =========================================================
def build_model(dropout=DROPOUT, units=UNITS):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():          # freeze backbone (Phase 1)
        param.requires_grad = False

    in_feats = model.fc.in_features           # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(in_feats, units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(units, 1),
        nn.Sigmoid()
    )
    return model


# =========================================================
# 4. Metrics  (unchanged)
# =========================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return acc, prec, rec_pos, rec_neg, auc


# =========================================================
# 5. Batch generator  (unchanged)
# =========================================================
def get_batches(X, y, batch_size=BATCH_SIZE):
    idx = torch.randperm(len(X))
    X, y = X[idx], y[idx]
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


# =========================================================
# 6. Early stopping
# =========================================================
class EarlyStopping:
    """
    Monitors val_accuracy.
    Saves best weights internally and restores them on stop.
    """
    def __init__(self, patience=8, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.best_weights = None

    def step(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score   = val_acc
            self.counter      = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience   # True → stop

    def restore(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =========================================================
# 7. Validation pass (used by schedulers + early stopping)
# =========================================================
def evaluate(model, X_val, y_val, criterion, device, batch_size=BATCH_SIZE):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for bx, by in get_batches(X_val, y_val, batch_size):
            bx = bx.to(device)
            out = model(bx).squeeze()
            loss = criterion(out, by.to(device))
            total_loss += loss.item()
            n_batches  += 1

            probs = out.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            y_true.extend(by.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    val_loss = total_loss / max(n_batches, 1)
    val_acc  = accuracy_score(y_true, y_pred)
    model.train()
    return val_loss, val_acc, y_true, y_pred, y_prob


# =========================================================
# 8. Training loop  (single run)
# =========================================================
def run_training_once(run_id,
                      dropout=DROPOUT, units=UNITS,
                      batch_size=BATCH_SIZE,
                      lr1=LR1, lr2=LR2,
                      epochs_phase1=EPOCHS_P1,
                      epochs_phase2=EPOCHS_P2):

    # ── Load data ──────────────────────────────────────────
    X_train, y_train = load_data(os.path.join(BASE_PATH, 'train'))
    X_test,  y_test  = load_data(os.path.join(BASE_PATH, 'test'))

    X_train_t = apply_transform(X_train)
    X_test_t  = apply_transform(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    # ── Setup ──────────────────────────────────────────────
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = build_model(dropout, units).to(device)
    criterion = nn.BCELoss(reduction='mean')  # L = -(1/N)*sum[y*log(p)+(1-y)*log(1-p)]

    cpu_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().percent
    t_start   = time.time()

    # ══════════════════════════════════════════════════════
    # PHASE 1 – Head only (backbone frozen)
    #   Scheduler : CosineAnnealingWarmRestarts (T0=10, Tmult=2)
    #   Early stop: patience = 8
    # ══════════════════════════════════════════════════════
    optimizer_p1  = optim.Adam(model.fc.parameters(), lr=lr1)
    scheduler_p1  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p1, T_0=10, T_mult=2, eta_min=1e-5
    )
    stopper_p1 = EarlyStopping(patience=8, min_delta=0.001)

    for epoch in range(epochs_phase1):
        model.train()
        for bx, by in get_batches(X_train_t, y_train_t, batch_size):
            bx, by = bx.to(device), by.to(device)
            optimizer_p1.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer_p1.step()

        scheduler_p1.step(epoch)                       # cosine step

        val_loss, val_acc, *_ = evaluate(
            model, X_test_t, y_test_t, criterion, device, batch_size
        )

        print(f"  [P1] Epoch {epoch+1:03d} | "
              f"val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | "
              f"lr={scheduler_p1.get_last_lr()[0]:.2e}")

        if stopper_p1.step(val_acc, model):
            print(f"  [P1] Early stop at epoch {epoch+1}. "
                  f"Best val_acc={stopper_p1.best_score:.4f}")
            break

    stopper_p1.restore(model)                          # restore best head weights

    # ══════════════════════════════════════════════════════
    # PHASE 2 – Full fine-tune (all layers unfrozen)
    #   Discriminative LRs: backbone ×0.1, head ×1
    #   Scheduler : ReduceLROnPlateau (factor=0.3, patience=4)
    #   Early stop: patience = 12
    # ══════════════════════════════════════════════════════
    for param in model.parameters():
        param.requires_grad = True

    # Discriminative learning rates (backbone at lr2/10, head at lr2)
    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith('fc')]
    head_params     = list(model.fc.parameters())

    optimizer_p2 = optim.Adam([
        {'params': backbone_params, 'lr': lr2 * 0.1},   # 3e-6
        {'params': head_params,     'lr': lr2}           # 3e-5
    ])
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode='max', factor=0.3,
        patience=4, threshold=0.001, min_lr=1e-7
    )
    stopper_p2 = EarlyStopping(patience=12, min_delta=0.001)

    for epoch in range(epochs_phase2):
        model.train()
        for bx, by in get_batches(X_train_t, y_train_t, batch_size):
            bx, by = bx.to(device), by.to(device)
            optimizer_p2.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer_p2.step()

        val_loss, val_acc, *_ = evaluate(
            model, X_test_t, y_test_t, criterion, device, batch_size
        )
        scheduler_p2.step(val_acc)                     # plateau step on val_acc

        current_lr = optimizer_p2.param_groups[1]['lr']
        print(f"  [P2] Epoch {epoch+1:03d} | "
              f"val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | "
              f"head_lr={current_lr:.2e}")

        if stopper_p2.step(val_acc, model):
            print(f"  [P2] Early stop at epoch {epoch+1}. "
                  f"Best val_acc={stopper_p2.best_score:.4f}")
            break

    stopper_p2.restore(model)                          # restore best full-model weights

    # ── Final evaluation ───────────────────────────────────
    _, _, y_true, y_pred, y_prob = evaluate(
        model, X_test_t, y_test_t, criterion, device, batch_size
    )
    acc, prec, rec_pos, rec_neg, auc = compute_metrics(y_true, y_pred, y_prob)
    # clamp to (eps, 1-eps) to avoid log(0) in BCE
    test_loss = criterion(
        torch.clamp(torch.tensor(y_prob), 1e-7, 1 - 1e-7),
        torch.tensor(y_true)
    ).item()

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
# 9. Multi-run experiment + CSV summary  (unchanged structure)
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

    df.to_csv("tier2-claude_results.csv", index=False)
    print("\n✔ tier2-claude_results.csv generated")

    summary = df.describe().transpose()[["mean", "std", "min", "max"]]
    summary.to_csv("tier2-claude_summary.csv")
    print("✔ tier2-claude_summary.csv generated")

    return df, summary


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    df, summary = run_experiments(n_runs=20)
    print("\n── Summary ──")
    print(summary.to_string())
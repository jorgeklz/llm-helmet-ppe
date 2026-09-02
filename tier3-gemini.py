import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
import time
import pandas as pd
import numpy as np
import psutil
from PIL import Image as PILImage 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    train_dir          = "dataset/train",
    test_dir           = "dataset/test",
    img_size           = 224,
    batch_size         = 16,
    head_hidden        = 256,
    dropout_rate       = 0.4,
    lr_phase1          = 1e-3,
    epochs_phase1      = 15,
    patience_phase1    = 5,
    lr_phase2          = 1e-4, 
    epochs_phase2      = 20,
    patience_phase2    = 7,
    n_runs             = 20,
    device             = "cuda" if torch.cuda.is_available() else "cpu",
    deterministic      = True,
    decision_threshold = 0.50 
)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if CFG["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. Architecture: Parallel-Stage CNN
# ─────────────────────────────────────────────────────────────────────────────
class ParallelBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 2
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, mid, 3, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, mid, 5, padding=2),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x)], dim=1)

class HelmetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = self._stage(3, 32)
        self.s2 = self._stage(32, 64)
        self.s3 = nn.Sequential(ParallelBlock(64, 128), nn.MaxPool2d(2))
        self.s4 = nn.Sequential(ParallelBlock(128, 256), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(CFG["dropout_rate"]),
            nn.Linear(256, CFG["head_hidden"]),
            nn.ReLU(inplace=True),
            nn.Linear(CFG["head_hidden"], 1)
        )

    def _stage(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.classifier(self.gap(self.s4(self.s3(self.s2(self.s1(x))))))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset & Augmentation
# ─────────────────────────────────────────────────────────────────────────────
class HelmetDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder, self.transform = folder, transform
        self.samples = self._scan(folder)

    @staticmethod
    def _scan(folder):
        items = []
        if not os.path.exists(folder): return items
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_p = os.path.join(folder, f)
                txt_p = os.path.splitext(img_p)[0] + ".txt"
                if os.path.exists(txt_p): items.append((img_p, txt_p))
        return items

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, txt_p = self.samples[idx]
        img = PILImage.open(img_p).convert("RGB")
        if self.transform: img = self.transform(img)
        with open(txt_p, "r") as f:
            cls = int(f.readline().strip().split()[0])
        return img, torch.tensor(cls, dtype=torch.float32)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────────────────────────────────────
# 3. Training & Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(model, loader, val_loader, optimizer, criterion, epochs, patience, scheduler=None):
    best_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(CFG["device"])).squeeze(), y.to(CFG["device"]))
            loss.backward()
            optimizer.step()
        if scheduler: scheduler.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                v_loss += criterion(model(x.to(CFG["device"])).squeeze(), y.to(CFG["device"])).item()
        v_loss /= len(val_loader)
        if v_loss < best_loss:
            best_loss = v_loss; counter = 0
        else:
            counter += 1
            if counter >= patience: break

def run_training_once(run_id):
    set_seed(42 + run_id)
    model = HelmetNet().to(CFG["device"])
    criterion = nn.BCEWithLogitsLoss()
    
    # Splits
    full_ds = HelmetDataset(CFG["train_dir"], train_transforms)
    train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.15)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=CFG["batch_size"])
    test_ds = HelmetDataset(CFG["test_dir"], test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    # Phase 1
    for name, p in model.named_parameters(): 
        if "classifier" not in name: p.requires_grad = False
    opt1 = optim.Adam(model.classifier.parameters(), lr=CFG["lr_phase1"])
    train_loop(model, train_loader, val_loader, opt1, criterion, CFG["epochs_phase1"], CFG["patience_phase1"])

    # Phase 2
    for p in model.parameters(): p.requires_grad = True
    opt2 = optim.Adam(model.parameters(), lr=CFG["lr_phase2"])
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=CFG["epochs_phase2"])
    train_loop(model, train_loader, val_loader, opt2, criterion, CFG["epochs_phase2"], CFG["patience_phase2"], sched2)

    # Final Inference (CPU)
    # Evaluate test loss using BCELoss on sigmoid probabilities:
    # L = -(1/N)*sum[ y*log(p) + (1-y)*log(1-p) ]
    model.eval().to("cpu")
    test_criterion = nn.BCELoss(reduction='sum')
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    start_time = time.perf_counter()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            prob = torch.sigmoid(out)
            prob_clamped = torch.clamp(prob, 1e-7, 1 - 1e-7)
            # accumulate sum BCE over this sample
            total_loss += test_criterion(prob_clamped.squeeze(0), y).item()

            prob_val = prob.item()
            y_prob.append(prob_val)
            y_pred.append(1 if prob_val > CFG["decision_threshold"] else 0)
            y_true.append(int(y.item()))

    exec_time = time.perf_counter() - start_time
    # Divide by N to get mean BCE
    avg_loss = total_loss / len(test_ds)
    
    # Metrics
    metrics = [
        run_id, accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        roc_auc_score(y_true, y_prob), avg_loss, exec_time, cpu_usage, ram_usage
    ]

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.savefig(f"cm_run{run_id:02d}.png"); plt.close()

    return metrics

def run_experiments(n_runs: int = 20):
    results = []
    for r in range(1, n_runs + 1):
        print(f"Run {r}/{n_runs}...")
        results.append(run_training_once(r))
        
    cols = ["run", "accuracy", "precision", "recall_pos", "recall_neg", "auc", "loss", "execution_time", "cpu_percent", "ram_percent"]
    df = pd.DataFrame(results, columns=cols)
    summary = df.drop(columns=["run"]).describe().transpose()[["mean", "std", "min", "max"]]
    
    print("\n" + "="*80)
    print(f"{'MÉTRICA':<15} | {'MEDIA':>10} | {'STD':>10} | {'MIN':>10} | {'MAX':>10}")
    print("-" * 80)
    for idx, row in summary.iterrows():
        print(f"{idx:<15} | {row['mean']:>10.4f} | {row['std']:>10.4f} | {row['min']:>10.4f} | {row['max']:>10.4f}")
    print("="*80)
    return df, summary

if __name__ == "__main__":
    run_experiments(CFG["n_runs"])
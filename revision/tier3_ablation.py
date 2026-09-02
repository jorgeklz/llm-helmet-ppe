"""
Reviewers 2, 3, 4: ablation to isolate WHICH Tier-3 constraint drives the
Gemini/Grok non-convergence. The revised manuscript argues the culprit is the
interaction of (a) a from-scratch multi-scale architecture with (b) the
prescribed CosineAnnealingLR two-phase schedule, not prompt length. This script
tests that directly by a 2x2 factorial:

    backbone in {pretrained_resnet18, fromscratch_multiscale}
    scheduler in {cosine_annealing, reduce_on_plateau}

All four cells share identical data, augmentation, optimizer family, epochs and
early stopping, so only the two factors vary. Reports test accuracy per cell,
averaged over --runs seeds. The prediction: fromscratch + cosine is the cell
that collapses to ~chance.

Usage: python revision/tier3_ablation.py --runs 5 --epochs 25
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__))
from common_data import load_data, HelmetDataset, train_transform, eval_transform, metrics_from_logits
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

class ParallelBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__(); mid = oc // 2
        self.p1 = nn.Sequential(nn.Conv2d(ic, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.p2 = nn.Sequential(nn.Conv2d(ic, mid, 5, padding=2), nn.BatchNorm2d(mid), nn.ReLU(True))
    def forward(self, x): return torch.cat([self.p1(x), self.p2(x)], 1)

class MultiScaleCNN(nn.Module):
    """The prescribed Tier-3 from-scratch architecture (32->64->128->256)."""
    def __init__(self):
        super().__init__()
        def stage(ic, oc): return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.ReLU(True), nn.MaxPool2d(2))
        self.s1, self.s2 = stage(3, 32), stage(32, 64)
        self.s3 = nn.Sequential(ParallelBlock(64, 128), nn.MaxPool2d(2))
        self.s4 = nn.Sequential(ParallelBlock(128, 256), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 1))
    def forward(self, x): return self.head(self.gap(self.s4(self.s3(self.s2(self.s1(x))))))

def make_model(backbone):
    if backbone == "pretrained_resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 128), nn.ReLU(True),
                             nn.Dropout(0.4), nn.Linear(128, 1))
        return m
    return MultiScaleCNN()

def run_cell(backbone, scheduler, X, y, Xte, yte, dev, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
    m = make_model(backbone).to(dev); crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if scheduler == "cosine_annealing"
             else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3))
    tl = DataLoader(HelmetDataset(Xtr, ytr, train_transform), batch_size=32, shuffle=True)
    vl = DataLoader(HelmetDataset(Xva, yva, eval_transform), batch_size=32)
    tel = DataLoader(HelmetDataset(Xte, yte, eval_transform), batch_size=32)
    for _ in range(epochs):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(m(xb).squeeze(1), yb).backward(); opt.step()
        if scheduler == "cosine_annealing": sched.step()
        else:
            m.eval(); tot = n = 0
            with torch.no_grad():
                for xb, yb in vl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    tot += crit(m(xb).squeeze(1), yb).item()*len(yb); n += len(yb)
            sched.step(tot/max(1, n))
    m.eval(); logits = []
    with torch.no_grad():
        for xb, _ in tel: logits.append(m(xb.to(dev)).squeeze(1).cpu())
    return metrics_from_logits(torch.cat(logits), yte)["accuracy"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset"); ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(os.path.join(a.data, "train")); Xte, yte = load_data(os.path.join(a.data, "test"))
    for backbone in ["pretrained_resnet18", "fromscratch_multiscale"]:
        for scheduler in ["cosine_annealing", "reduce_on_plateau"]:
            accs = [run_cell(backbone, scheduler, X, y, Xte, yte, dev, a.epochs, s) for s in range(a.runs)]
            accs = np.asarray(accs)
            print(f"[{backbone:24s} | {scheduler:18s}] acc mean={accs.mean():.4f} sd={accs.std(ddof=1):.4f}")

if __name__ == "__main__": main()

"""
Reviewers 2, 4, 5: stratified k-fold cross-validation with confidence intervals,
to confirm the reported accuracy is not tied to one particular 49/51 split.

Pools ./dataset/train + ./dataset/test, runs Stratified K-Fold (default k=5)
with a ResNet50 transfer-learning pipeline (frozen backbone + head warm-up,
then full fine-tune), and reports per-fold Accuracy/AUC plus mean, SD and a
95% t-confidence interval across folds.

Usage:  python revision/kfold_cv.py --k 5 --epochs_head 5 --epochs_ft 5
Run from the project root. Requires torch, torchvision, scikit-learn.
"""
import argparse, math, sys, os, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(__file__))
from common_data import load_data, HelmetDataset, train_transform, eval_transform, metrics_from_logits

def build_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, 1))
    return m

def train_eval(Xtr, ytr, Xte, yte, dev, eh, ef, bs=32):
    m = build_resnet50().to(dev)
    crit = nn.BCEWithLogitsLoss()
    tl = DataLoader(HelmetDataset(Xtr, ytr, train_transform), batch_size=bs, shuffle=True)
    vl = DataLoader(HelmetDataset(Xte, yte, eval_transform), batch_size=bs)
    def run_epochs(opt, n):
        for _ in range(n):
            m.train()
            for xb, yb in tl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); out = m(xb).squeeze(1)
                loss = crit(out, yb); loss.backward(); opt.step()
    run_epochs(torch.optim.Adam(m.fc.parameters(), lr=1e-3), eh)     # Phase 1: head
    for p in m.parameters(): p.requires_grad = True
    run_epochs(torch.optim.Adam(m.parameters(), lr=1e-4), ef)        # Phase 2: fine-tune
    m.eval(); logits = []
    with torch.no_grad():
        for xb, _ in vl: logits.append(m(xb.to(dev)).squeeze(1).cpu())
    return metrics_from_logits(torch.cat(logits), yte)

def ci95(a):
    a = np.asarray(a); n = len(a); sd = a.std(ddof=1)
    tcrit = {2:12.71,3:4.30,4:3.18,5:2.78,6:2.57,7:2.45,8:2.36,9:2.31,10:2.26}.get(n,2.09)
    h = tcrit*sd/math.sqrt(n); return a.mean(), sd, a.mean()-h, a.mean()+h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset"); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs_head", type=int, default=5); ap.add_argument("--epochs_ft", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xa, ya = load_data(os.path.join(a.data, "train")); Xb, yb = load_data(os.path.join(a.data, "test"))
    X = np.concatenate([Xa, Xb]); y = np.concatenate([ya, yb])
    print(f"Pooled dataset: {len(y)} images, class balance = {np.bincount(y)}")
    skf = StratifiedKFold(n_splits=a.k, shuffle=True, random_state=a.seed)
    accs, aucs = [], []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        r = train_eval(X[tr], y[tr], X[te], y[te], dev, a.epochs_head, a.epochs_ft)
        accs.append(r["accuracy"]); aucs.append(r["auc"])
        print(f"Fold {i}: acc={r['accuracy']:.4f} auc={r['auc']:.4f}")
    for name, arr in [("Accuracy", accs), ("AUC", aucs)]:
        m, sd, lo, hi = ci95(arr)
        print(f"{name}: mean={m:.4f} sd={sd:.4f} 95%CI=[{lo:.4f},{hi:.4f}]")

if __name__ == "__main__": main()

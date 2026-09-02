"""
Reviewers 2 & 4: an expert-designed, standardized deep-learning baseline with a
FIXED, identical protocol, so that any advantage of the LLM-generated pipelines
can be attributed to the pipelines rather than to differing preprocessing /
augmentation / optimization. Same data, same transforms, same 30-run protocol.

Protocol (held constant): ResNet50 (ImageNet) backbone; Phase 1 head warm-up
(Adam lr=1e-3) then Phase 2 full fine-tune (Adam lr=1e-4) with
CosineAnnealingLR; class-weighted BCE; early stopping on a 15% validation split;
identical augmentation for all runs. Reports mean +/- SD over --runs seeds.

Usage: python revision/expert_baseline.py --runs 30 --epochs_head 5 --epochs_ft 15
"""
import argparse, os, sys, math, numpy as np, torch, torch.nn as nn
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

def build():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, 1))
    return m

def one_run(Xtr, ytr, Xva, yva, Xte, yte, dev, eh, ef, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build().to(dev)
    pw = torch.tensor([ (ytr == 0).sum() / max(1, (ytr == 1).sum()) ], dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    tl = DataLoader(HelmetDataset(Xtr, ytr, train_transform), batch_size=32, shuffle=True)
    vl = DataLoader(HelmetDataset(Xva, yva, eval_transform), batch_size=32)
    tel = DataLoader(HelmetDataset(Xte, yte, eval_transform), batch_size=32)
    def epoch(opt):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = crit(m(xb).squeeze(1), yb); loss.backward(); opt.step()
    def val_loss():
        m.eval(); tot = n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                tot += crit(m(xb).squeeze(1), yb).item()*len(yb); n += len(yb)
        return tot/max(1, n)
    for _ in range(eh): epoch(torch.optim.Adam(m.fc.parameters(), lr=1e-3))
    for p in m.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ef)
    best, bad, patience = 1e9, 0, 7
    for _ in range(ef):
        epoch(opt); sched.step(); vlss = val_loss()
        if vlss < best - 1e-4: best, bad = vlss, 0
        else:
            bad += 1
            if bad >= patience: break
    m.eval(); logits = []
    with torch.no_grad():
        for xb, _ in tel: logits.append(m(xb.to(dev)).squeeze(1).cpu())
    return metrics_from_logits(torch.cat(logits), yte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./dataset"); ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--epochs_head", type=int, default=5); ap.add_argument("--epochs_ft", type=int, default=15)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr0, ytr0 = load_data(os.path.join(a.data, "train")); Xte, yte = load_data(os.path.join(a.data, "test"))
    accs, aucs, losses = [], [], []
    for s in range(a.runs):
        Xtr, Xva, ytr, yva = train_test_split(Xtr0, ytr0, test_size=0.15, random_state=s, stratify=ytr0)
        r = one_run(Xtr, ytr, Xva, yva, Xte, yte, dev, a.epochs_head, a.epochs_ft, s)
        accs.append(r["accuracy"]); aucs.append(r["auc"])
        print(f"run {s+1}/{a.runs}: acc={r['accuracy']:.4f} auc={r['auc']:.4f}")
    for name, arr in [("Accuracy", accs), ("AUC", aucs)]:
        arr = np.asarray(arr); print(f"{name}: mean={arr.mean():.4f} sd={arr.std(ddof=1):.4f}")

if __name__ == "__main__": main()

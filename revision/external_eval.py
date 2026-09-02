"""
Reviewers 2, 3, 4, 5: evaluate a trained model on an EXTERNAL dataset to test
generalization beyond the purpose-built dataset.

Point --external at a folder of images with the same sibling-.txt label
convention (first token = class id; 1 = no-helmet, 0 = helmet). By default it
rebuilds a ResNet50 1-logit head and loads --ckpt (best_model.pth). If your
checkpoint uses a different architecture, adjust build_model() accordingly.

Usage: python revision/external_eval.py --external /path/to/external --ckpt best_model.pth
"""
import argparse, os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
sys.path.insert(0, os.path.dirname(__file__))
from common_data import load_data, HelmetDataset, eval_transform, metrics_from_logits

def build_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, 1))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True); ap.add_argument("--ckpt", default="best_model.pth")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_data(a.external)
    print(f"External set: {len(y)} images, class balance = {np.bincount(y)}")
    m = build_model().to(dev)
    try:
        sd = torch.load(a.ckpt, map_location=dev)
        m.load_state_dict(sd.get("state_dict", sd), strict=False)
        print(f"Loaded checkpoint {a.ckpt}")
    except Exception as e:
        print(f"WARNING: could not load {a.ckpt} ({e}); evaluating ImageNet-init head (edit build_model to match your checkpoint).")
    m.eval(); dl = DataLoader(HelmetDataset(X, y, eval_transform), batch_size=32)
    logits = []
    with torch.no_grad():
        for xb, _ in dl: logits.append(m(xb.to(dev)).squeeze(1).cpu())
    r = metrics_from_logits(torch.cat(logits), y)
    print("External-set metrics:")
    for k in ["accuracy","precision","recall_pos","recall_neg","auc","tp","tn","fp","fn"]:
        print(f"  {k}: {r[k]}")

if __name__ == "__main__": main()

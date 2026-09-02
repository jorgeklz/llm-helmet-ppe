#!/usr/bin/env python3
"""Parse the experiment logs into a single results file ready to paste."""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "revision", "logs")
OUT = os.path.join(ROOT, "revision", "experiment_results.txt")

def read(name):
    p = os.path.join(LOG, name)
    return open(p).read() if os.path.exists(p) else ""

lines = []
def w(s=""): lines.append(s)

w("="*70); w("CONSOLIDATED EXPERIMENT RESULTS (paste these into the response letter)"); w("="*70); w("")

# k-fold
kf = read("kfold.log")
w("1) STRATIFIED K-FOLD CROSS-VALIDATION (Reviewers 2, 4, 5)")
for ln in kf.splitlines():
    if re.search(r"Fold \d|Accuracy:|AUC:|Pooled", ln): w("   " + ln.strip())
w("")

# baseline
bl = read("baseline.log")
w("2) EXPERT-OPTIMIZED STANDARDIZED BASELINE (Reviewers 2, 4)")
for ln in bl.splitlines():
    if re.search(r"Accuracy:|AUC:", ln): w("   " + ln.strip())
w("")

# ablation
ab = read("ablation.log")
w("3) TIER-3 FACTORIAL ABLATION (Reviewers 2, 3, 4, 5)")
w("   backbone x scheduler -> test accuracy (mean +/- sd). Expect the")
w("   from-scratch + cosine_annealing cell to collapse toward chance.")
for ln in ab.splitlines():
    if "acc mean=" in ln: w("   " + ln.strip())
w("")

# external
ex = read("external.log")
if ex.strip():
    w("4) EXTERNAL-DATASET EVALUATION (Reviewers 2, 3, 4, 5)")
    for ln in ex.splitlines():
        if re.search(r"External set:|accuracy:|precision:|recall_pos:|recall_neg:|auc:", ln):
            w("   " + ln.strip())
    w("")

open(OUT, "w").write("\n".join(lines))
print("WROTE", OUT)
print("\n".join(lines))

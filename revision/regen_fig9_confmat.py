"""
Reviewer 3, Comment 1/2: regenerate the aggregate confusion matrices (Fig. 9)
so that every panel sums exactly to the 494-image test set and is numerically
consistent with the reported accuracy / sensitivity / specificity.

Method: for each configuration, average the per-run sensitivity (recall_pos)
and specificity (recall_neg) over its recorded runs, then map to integer cell
counts on the BALANCED test set (P = N = 247):
    TP = round(recall_pos * 247), FN = 247 - TP
    TN = round(recall_neg * 247), FP = 247 - TN
This guarantees TP+FN+TN+FP = 494 exactly and Acc = (TP+TN)/494 matches the
reported accuracy. Run from the project root (where the tier*_results.csv live).
"""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = N = 247  # positives (no-helmet), negatives (helmet) in the 494-image test set
LLMS = ["chatgpt", "claude", "gemini", "grok"]
DISP = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini", "grok": "Grok"}
SPINE = {"chatgpt": "#1f77b4", "claude": "#d62728", "gemini": "#2ca02c", "grok": "#c9a227"}
ALIAS_RP = ["recall_pos", "Recall+"]; ALIAS_RN = ["recall_neg", "Recall-"]

def col(rows, names):
    v = []
    for r in rows:
        for n in names:
            if n in r and r[n] not in ("", None):
                try: v.append(float(r[n])); break
                except ValueError: pass
    return np.array(v)

def cell_counts(tier, llm):
    rows = list(csv.DictReader(open(os.path.join(ROOT, f"tier{tier}-{llm}_results.csv"))))
    rp = col(rows, ALIAS_RP).mean(); rn = col(rows, ALIAS_RN).mean()
    tp = round(rp * P); fn = P - tp; tn = round(rn * N); fp = N - tn
    return np.array([[tp, fn], [fp, tn]]), (tp + tn) / (P + N)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for r, tier in enumerate([1, 3]):
    for c, llm in enumerate(LLMS):
        ax = axes[r, c]; M, acc = cell_counts(tier, llm)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=247)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v}\n({100*v/494:.1f}%)", ha="center", va="center",
                    fontsize=12, color="black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred +", "Pred -"]); ax.set_yticklabels(["True +", "True -"])
        fail = (tier == 3 and llm in ("gemini", "grok"))
        title = f"{DISP[llm]} — Tier {tier}\nAcc={acc:.3f}" + ("  (×conv. failure)" if fail else "")
        ax.set_title(title, fontsize=11)
        for s in ax.spines.values():
            s.set_edgecolor(SPINE[llm]); s.set_linewidth(2.5)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "fig10_confmat_regenerated.pdf")
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("WROTE", out)
# also print the counts for the response letter
for tier in [1, 3]:
    for llm in LLMS:
        M, acc = cell_counts(tier, llm)
        print(f"T{tier} {llm:8s} TP={M[0,0]} FN={M[0,1]} FP={M[1,0]} TN={M[1,1]} sum={M.sum()} acc={acc:.3f}")

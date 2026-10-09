"""
Build docs/data.js for the results website (docs/index.html).

Reads ONLY files already in the repository (raw per-run logs, the recomputed
statistics and the consolidated revision results) and writes a single
JavaScript file that the page loads. No training, no network.

Usage (from the repository root):
    python docs/build_data.py
"""
import base64, csv, io, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data.js")
LLMS = ["chatgpt", "claude", "gemini", "grok"]
TIERS = [1, 2, 3]

# The twelve raw logs use three different column conventions; map them all
# to one set of keys.
ALIASES = {
    "accuracy": ["accuracy", "Accuracy"],
    "precision": ["precision", "Precision"],
    "recall_pos": ["recall_pos", "Recall+"],
    "recall_neg": ["recall_neg", "Recall-"],
    "auc": ["auc", "AUC"],
    "loss": ["loss", "Loss"],
    "time_sec": ["time_sec", "Exec Time (s)"],
}


def read_runs(tier, llm):
    path = os.path.join(ROOT, f"tier{tier}-{llm}_results.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key, names in ALIASES.items():
        col = next((n for n in names if n in rows[0]), None)
        out[key] = [round(float(r[col]), 6) for r in rows] if col else None
    return out


def read_summary():
    stats = {}
    with open(os.path.join(ROOT, "revision", "stats_summary.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["mean"] == "NA":
                continue
            k = f"{r['tier']}-{r['llm']}"
            stats.setdefault(k, {})[r["metric"]] = {
                "n": int(r["n"]), "mean": float(r["mean"]), "sd": float(r["sd"]),
                "lo": float(r["ci95_lo"]), "hi": float(r["ci95_hi"]),
            }
    return stats


def read_revision():
    txt = open(os.path.join(ROOT, "revision", "experiment_results.txt")).read()
    folds = [{"acc": float(a), "auc": float(b)}
             for a, b in re.findall(r"Fold \d+: acc=([\d.]+) auc=([\d.]+)", txt)]
    kf = re.search(r"Accuracy: mean=([\d.]+) sd=([\d.]+) 95%CI=\[([\d.]+),([\d.]+)\]", txt)
    kauc = re.search(r"AUC: mean=([\d.]+) sd=([\d.]+) 95%CI", txt)
    base = txt.split("2) EXPERT")[1]
    bacc = re.search(r"Accuracy: mean=([\d.]+) sd=([\d.]+)", base)
    bauc = re.search(r"AUC: mean=([\d.]+) sd=([\d.]+)", base)
    abl = [{"backbone": b, "scheduler": s, "mean": float(m), "sd": float(d)}
           for b, s, m, d in re.findall(
               r"\[(\w+)\s*\|\s*(\w+)\s*\] acc mean=([\d.]+) sd=([\d.]+)", txt)]
    return {
        "kfold": {"folds": folds, "mean": float(kf[1]), "sd": float(kf[2]),
                  "lo": float(kf[3]), "hi": float(kf[4]),
                  "auc": float(kauc[1]), "auc_sd": float(kauc[2])},
        "baseline": {"acc": float(bacc[1]), "acc_sd": float(bacc[2]),
                     "auc": float(bauc[1]), "auc_sd": float(bauc[2])},
        "ablation": abl,
    }


def read_anova():
    txt = open(os.path.join(ROOT, "revision", "stats_tests.txt")).read()
    rows = re.findall(r"^(LLM x Tier|LLM|Tier)\s+SS=([\d.]+)\s+df=(\d+)\s+MS=([\d.]+)\s+F=([\d.]+)\s+p=([\d.e+-]+)",
                      txt, re.M)
    return [{"factor": f, "ss": float(ss), "df": int(df), "F": float(F), "p": p}
            for f, ss, df, ms, F, p in rows]


def script_info(tier, llm):
    """Backbone and size of each LLM-generated script, read from its source."""
    src = open(os.path.join(ROOT, f"tier{tier}-{llm}.py")).read()
    if "resnet50" in src:
        backbone = "ResNet50 (ImageNet)"
    elif "resnet18" in src:
        backbone = "ResNet18 (ImageNet)"
    else:
        backbone = "Multi-scale CNN from scratch"
    return {"file": f"tier{tier}-{llm}.py", "backbone": backbone, "lines": src.count("\n")}


def samples(k=3, size=360):
    """A few test images (with their YOLO-style box) to animate in the hero."""
    from PIL import Image
    test = os.path.join(ROOT, "dataset", "test")
    picks = []
    for tag, label in (("CON_Casco", 0), ("SIN_Casco", 1)):
        names = sorted(n for n in os.listdir(test) if tag in n and n.endswith(".jpg"))
        for n in names[5:5 + k * 7:7]:
            lab = open(os.path.join(test, n[:-4] + ".txt")).read().split()
            box = [float(v) for v in lab[1:5]] if len(lab) >= 5 else None
            im = Image.open(os.path.join(test, n)).convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=68, optimize=True)
            picks.append({"label": label, "box": box,
                          "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
    # interleave helmet / no-helmet
    half = len(picks) // 2
    return [p for pair in zip(picks[:half], picks[half:]) for p in pair]


def main():
    runs = {f"{t}-{m}": read_runs(t, m) for t in TIERS for m in LLMS}
    count = lambda d: sum(1 for n in os.listdir(os.path.join(ROOT, "dataset", d)) if n.endswith(".jpg"))
    data = {
        "dataset": {"train": count("train"), "test": count("test")},
        "llms": LLMS, "tiers": TIERS,
        "runs": runs, "summary": read_summary(),
        "scripts": {f"{t}-{m}": script_info(t, m) for t in TIERS for m in LLMS},
        "anova": read_anova(), "samples": samples(),
        **read_revision(),
    }
    with open(OUT, "w") as fh:
        fh.write("// Generated by docs/build_data.py from the repository logs. Do not edit by hand.\n")
        fh.write("window.STUDY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

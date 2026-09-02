#!/usr/bin/env python3
"""
Statistical re-analysis of the LLM x Tier helmet-classification study.
Reads the raw per-run *_results.csv files (real n, not assumed 30).
Pure numpy + pure-python p-values (no scipy required).

Outputs:
  - revision/stats_summary.csv         (mean, sd, 95% CI, n per metric/config)
  - revision/stats_tests.txt           (Welch t, Mann-Whitney U, Cohen d, ANOVA)
"""
import csv, math, os
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")

# ---- column-name harmonisation across the heterogeneous CSVs ----
ALIASES = {
    "accuracy": ["accuracy", "Accuracy"],
    "precision": ["precision", "Precision"],
    "recall_pos": ["recall_pos", "Recall+"],
    "recall_neg": ["recall_neg", "Recall-"],
    "auc": ["auc", "AUC"],
    "loss": ["loss", "Loss"],
    "cpu": ["cpu_percent", "CPU %"],
    "ram": ["ram_percent", "RAM %"],
    "time": ["time_sec", "Exec Time (s)", "execution_time"],
}
LLMS = ["chatgpt", "claude", "gemini", "grok"]
TIERS = [1, 2, 3]

def load(tier, llm):
    path = os.path.join(BASE, f"tier{tier}-{llm}_results.csv")
    rows = list(csv.DictReader(open(path)))
    out = {}
    for canon, names in ALIASES.items():
        vals = []
        for r in rows:
            for nm in names:
                if nm in r and r[nm] not in ("", None):
                    try:
                        vals.append(float(r[nm])); break
                    except ValueError:
                        pass
        out[canon] = np.array(vals, dtype=float) if vals else np.array([])
    out["n"] = len(rows)
    return out

DATA = {(t, l): load(t, l) for t in TIERS for l in LLMS}

# ---------- pure-python special functions ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a+b, a+1.0, a-1.0
    c = 1.0; d = 1.0 - qab*x/qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0/d; h = d
    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1.0+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0/d; de = d*c; h *= de
        if abs(de-1.0) < EPS: break
    return h

def betainc(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
    front = math.exp(lbeta + a*math.log(x) + b*math.log(1.0-x))
    if x < (a+1.0)/(a+b+2.0):
        return front*_betacf(a, b, x)/a
    else:
        return 1.0 - front*_betacf(b, a, 1.0-x)/b

def t_sf_two_sided(t, df):
    if df <= 0: return float('nan')
    x = df/(df + t*t)
    return betainc(df/2.0, 0.5, x)          # = P(|T| > |t|)

def f_sf(F, d1, d2):
    if F <= 0: return 1.0
    x = d2/(d2 + d1*F)
    return betainc(d2/2.0, d1/2.0, x)

def t_ppf(p, df):                            # inverse CDF via bisection
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        c = 1.0 - 0.5*t_sf_two_sided(mid, df) if mid > 0 else 0.5*t_sf_two_sided(mid, df)
        if c < p: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) if na > 1 else 0.0, b.var(ddof=1) if nb > 1 else 0.0
    se2 = va/na + vb/nb
    if se2 == 0:
        return dict(t=float('inf'), df=na+nb-2, p=0.0, ma=ma, mb=mb)
    t = (ma-mb)/math.sqrt(se2)
    df = se2**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1)) if (na > 1 and nb > 1) else na+nb-2
    return dict(t=t, df=df, p=t_sf_two_sided(t, df), ma=ma, mb=mb)

def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp2 = ((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1))/(na+nb-2)
    if sp2 == 0: return float('inf')
    d = (a.mean()-b.mean())/math.sqrt(sp2)
    J = 1 - 3/(4*(na+nb)-9)                  # Hedges correction
    return d, d*J

def mannwhitney(a, b):
    na, nb = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    # tie correction
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i in range(len(allv)):
        grp = np.where(inv == inv[i])[0]
        ranks[i] = ranks[grp].mean()
    Ra = ranks[:na].sum()
    Ua = Ra - na*(na+1)/2.0
    U = min(Ua, na*nb - Ua)
    mu = na*nb/2.0
    tie = sum(c**3 - c for c in cnt)
    sigma = math.sqrt(na*nb/12.0 * ((na+nb+1) - tie/((na+nb)*(na+nb-1))))
    if sigma == 0: return dict(U=U, p=float('nan'))
    z = (U - mu + 0.5)/sigma
    p = 2*(0.5*math.erfc(abs(z)/math.sqrt(2)))
    return dict(U=U, z=z, p=min(1.0, p))

# ---------------- 1. descriptive table ----------------
metrics = ["accuracy", "precision", "recall_pos", "recall_neg", "auc", "loss"]
os.makedirs(os.path.join(BASE, "revision"), exist_ok=True)
with open(os.path.join(BASE, "revision", "stats_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tier", "llm", "metric", "n", "mean", "sd", "ci95_lo", "ci95_hi"])
    for t in TIERS:
        for l in LLMS:
            d = DATA[(t, l)]
            for m in metrics:
                v = d[m]
                if len(v) == 0:
                    w.writerow([t, l, m, 0, "NA", "NA", "NA", "NA"]); continue
                n = len(v); mean = v.mean(); sd = v.std(ddof=1) if n > 1 else 0.0
                tc = t_ppf(0.975, n-1) if n > 1 else 0.0
                half = tc*sd/math.sqrt(n)
                w.writerow([t, l, m, n, f"{mean:.4f}", f"{sd:.4f}",
                            f"{mean-half:.4f}", f"{mean+half:.4f}"])

# ---------------- 2. key inferential contrasts ----------------
out = []
def line(s=""): out.append(s)

line("="*78)
line("INFERENTIAL STATISTICS  (independent-samples, runs not paired across configs)")
line("="*78)
line("Note: runs across configurations are INDEPENDENT (different seeds, separate")
line("executions), so Welch's unequal-variance t-test + Mann-Whitney U are used")
line("instead of paired tests. Effect size = Cohen's d (pooled) / Hedges' g.")
line("")

def report(name, a, b):
    if len(a) < 2 or len(b) < 2:
        line(f"{name}: insufficient data"); return
    w = welch(a, b); d, g = cohen_d(a, b); mw = mannwhitney(a, b)
    line(f"{name}")
    line(f"   mean A={w['ma']:.4f}  mean B={w['mb']:.4f}  diff={w['ma']-w['mb']:+.4f}")
    line(f"   Welch t={w['t']:.3f}  df={w['df']:.1f}  p={w['p']:.2e}")
    line(f"   Mann-Whitney U={mw['U']:.1f}  p={mw['p']:.2e}")
    dd = 'inf' if math.isinf(d) else f"{d:.2f}"
    gg = 'inf' if math.isinf(g) else f"{g:.2f}"
    line(f"   Cohen d={dd}  Hedges g={gg}")
    line("")

line("--- Tier 1: pairwise accuracy contrasts among the four LLMs ---")
for i in range(len(LLMS)):
    for j in range(i+1, len(LLMS)):
        a, b = DATA[(1, LLMS[i])]["accuracy"], DATA[(1, LLMS[j])]["accuracy"]
        report(f"T1 {LLMS[i]} vs {LLMS[j]} (accuracy)", a, b)

line("--- Across tiers, converging models (accuracy) ---")
for l in ["chatgpt", "claude"]:
    report(f"{l} T1 vs T3 (accuracy)", DATA[(1, l)]["accuracy"], DATA[(3, l)]["accuracy"])
    report(f"{l} T1 vs T2 (accuracy)", DATA[(1, l)]["accuracy"], DATA[(2, l)]["accuracy"])

line("--- Convergence-failure contrasts (accuracy) ---")
report("chatgpt T3 vs gemini T3", DATA[(3, "chatgpt")]["accuracy"], DATA[(3, "gemini")]["accuracy"])
report("gemini T2 vs gemini T3", DATA[(2, "gemini")]["accuracy"], DATA[(3, "gemini")]["accuracy"])
report("grok T2 vs grok T3", DATA[(2, "grok")]["accuracy"], DATA[(3, "grok")]["accuracy"])

line("--- Claude loss trajectory (calibration) ---")
for (ta, tb) in [(1, 2), (2, 3), (1, 3)]:
    la, lb = DATA[(ta, "claude")]["loss"], DATA[(tb, "claude")]["loss"]
    report(f"claude T{ta} vs T{tb} (loss)", la, lb)

# ---------------- 3. two-way unweighted-means ANOVA on accuracy ----------------
line("="*78)
line("TWO-WAY ANOVA on accuracy  (factors: LLM x Tier) -- tests H3 interaction")
line("unweighted-means method (handles unequal cell n; grok-T3 n=8, others n=20)")
line("="*78)
cells = {(t, l): DATA[(t, l)]["accuracy"] for t in TIERS for l in LLMS}
cell_means = {k: v.mean() for k, v in cells.items()}
ns = {k: len(v) for k, v in cells.items()}
a_lv, b_lv = len(LLMS), len(TIERS)
nh = a_lv*b_lv / sum(1.0/ns[k] for k in cells)          # harmonic mean of cell n
grand = np.mean([cell_means[k] for k in cells])
llm_means = {l: np.mean([cell_means[(t, l)] for t in TIERS]) for l in LLMS}
tier_means = {t: np.mean([cell_means[(t, l)] for l in LLMS]) for t in TIERS}
SS_A = nh*b_lv*sum((llm_means[l]-grand)**2 for l in LLMS)
SS_B = nh*a_lv*sum((tier_means[t]-grand)**2 for t in TIERS)
SS_AB = nh*sum((cell_means[(t, l)]-llm_means[l]-tier_means[t]+grand)**2
               for t in TIERS for l in LLMS)
SS_within = sum(((v-v.mean())**2).sum() for v in cells.values())
N = sum(ns.values())
df_A, df_B, df_AB = a_lv-1, b_lv-1, (a_lv-1)*(b_lv-1)
df_within = N - a_lv*b_lv
MS_within = SS_within/df_within
for nm, SS, df in [("LLM", SS_A, df_A), ("Tier", SS_B, df_B), ("LLM x Tier", SS_AB, df_AB)]:
    MS = SS/df; F = MS/MS_within; p = f_sf(F, df, df_within)
    line(f"{nm:12s} SS={SS:.4f}  df={df}  MS={MS:.4f}  F={F:.2f}  p={p:.2e}")
line(f"{'Within':12s} SS={SS_within:.4f}  df={df_within}  MS={MS_within:.5f}")
line(f"harmonic-mean cell n = {nh:.2f}")

open(os.path.join(BASE, "revision", "stats_tests.txt"), "w").write("\n".join(out))
print("\n".join(out))
print("\nWROTE revision/stats_summary.csv and revision/stats_tests.txt")

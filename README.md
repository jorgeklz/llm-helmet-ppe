# LLM-Generated CNNs for Helmet-Wearing Compliance Classification

Code and data supporting the study submitted to *Discover Artificial Intelligence*.
This repository contains only the material requested by the reviewers for
reproducibility (the manuscript itself is not included here).

## Contents
- `dataset/` — 972 balanced images (486 helmet / 486 no-helmet) in `train/` and `test/`,
  each image with a sibling `.txt` label (first token = class id; 1 = no-helmet, 0 = helmet).
- `tier{1,2,3}-{chatgpt,claude,gemini,grok}.py` — the twelve LLM-generated training scripts.
- `tier*-*_results.csv`, `tier*-*_summary.csv`, `tier*-*-output.txt` — raw per-run logs and summaries.
- `revision/`
  - `stats_analysis.py`, `stats_summary.csv`, `stats_tests.txt` — statistics recomputed from the raw logs (n=20; n=8 for Grok Tier 3).
  - `regen_fig9_confmat.py`, `fig10_confmat_regenerated.*` — regenerated, internally consistent confusion-matrix figure.
  - `common_data.py`, `kfold_cv.py`, `expert_baseline.py`, `tier3_ablation.py`, `external_eval.py`, `collect_results.py` — reproducibility scripts.
  - `experiment_results.txt`, `kfold_result.txt` — consolidated cross-validation, baseline and ablation results.
  - `README.md`, `RUN_INSTRUCTIONS.md` — how to run everything.
- `run_experiments.sh`, `rerun_rest.sh` — one-command runners.
- `requirements.txt` — Python dependencies.

## Reproducing the experiments
From the repository root (PyTorch required):
```
nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &
```
Writes `revision/experiment_results.txt`. See `revision/RUN_INSTRUCTIONS.md` for details.

## Key results
- 5-fold cross-validation: accuracy 0.989 ± 0.007 (95% CI [0.980, 0.997]), AUC 0.9995.
- Expert-optimized standardized baseline: accuracy 0.968 ± 0.005, AUC 0.992.
- Tier-3 ablation: the from-scratch multi-scale architecture fails under both schedulers
  (0.53–0.57), while a pretrained ResNet18 backbone converges under both (0.945–0.947).

## Contact
Jorge Parraga-Alava — jorge.parraga@utm.edu.ec

# Revision package - Discover Artificial Intelligence (Round 1)

All manuscript changes are in `../manuscrito/main.tex`, highlighted in **blue**
via the new `\rr{...}` command (round-1 `\rev{...}` remains black). A backup of
the pre-revision manuscript is at `../manuscrito/main_backup_*.tex`.

## Deliverables
- `Response_to_Reviewers.docx` - point-by-point response (Editor + Reviewers 1-5), B2 English.
- `stats_summary.csv` - mean, SD and 95% CI per metric/config, recomputed from the raw logs (n=20; n=8 for Grok Tier 3).
- `stats_tests.txt` - Welch t-tests, Mann-Whitney U, Cohen d / Hedges g, and the two-way LLM x Tier ANOVA.
- `fig10_confmat_regenerated.pdf` / `.png` - the corrected Figure 9 (every panel sums to 494 and matches the reported accuracy).

## Data re-audit (important)
The raw `*_results.csv` contain **20 runs** per configuration (**8** for Grok Tier 3), not 30.
All tables, SDs, CIs, tests and the confusion-matrix figure were recomputed from these logs.
Not recovered from the logs (shown as `--`): Gemini Tier 3 per-run loss; ChatGPT Tier 2 process RAM.

## Runnable scripts (PyTorch; run from the project root, `python revision/<script>.py`)
- `regen_fig9_confmat.py` - regenerate the confusion-matrix figure (no training).
- `kfold_cv.py` - stratified k-fold cross-validation with 95% CIs (Reviewers 2,4,5).
- `external_eval.py --external <dir> --ckpt best_model.pth` - external-dataset evaluation (Reviewers 2,3,4,5).
- `expert_baseline.py --runs 30` - expert-optimized standardized baseline, fixed protocol (Reviewers 2,4).
- `tier3_ablation.py` - 2x2 factorial (pretrained vs from-scratch) x (CosineAnnealingLR vs ReduceLROnPlateau) to isolate the Tier-3 failure cause (Reviewers 2,3,4,5).
- `common_data.py` - shared loader/dataset/metrics (matches your `dataset/` .txt-label convention).

## TODO before submission
1. Run `kfold_cv.py`, `expert_baseline.py`, `tier3_ablation.py` (and `external_eval.py` if you have an external set); insert the results where the response letter marks them.
2. Insert the real repository URL/DOI in the Data and Code Availability statement.
3. If you accept the regenerated Figure 9, copy `fig10_confmat_regenerated.pdf` over `../manuscrito/fig10_confmat.pdf`.
4. Compile `main.tex` in your usual environment (this machine lacks `algorithm.sty`; the manuscript otherwise compiles cleanly).

## Provenance (edit helpers, not deliverables)
`apply_*.py`, `_ereg.py`, `_tables.py`, `_res.py`, `stats_analysis.py` were used to compute values and apply the blue edits.

# Revision package (Discover Artificial Intelligence, round 1)

Everything here answers a reviewer request. All numbers are recomputed from the raw
`../tier*-*_results.csv` logs or produced by the scripts below.

## Data re-audit
The raw logs contain **20 runs** per configuration (**8** for Grok Tier 3), not 30.
All tables, SDs, CIs, tests and the confusion-matrix figure were recomputed from these logs.
Two values could not be recovered and are reported as missing: Gemini Tier 3 per-run loss
and ChatGPT Tier 2 process RAM.

## Results files
| File | Content |
|---|---|
| `stats_summary.csv` | mean, SD and 95% CI per metric and configuration |
| `stats_tests.txt` | Welch t-tests, Mann-Whitney U, Cohen d / Hedges g, two-way LLM × Tier ANOVA |
| `experiment_results.txt` | consolidated k-fold, expert baseline and Tier-3 ablation results |
| `kfold_result.txt` | per-fold k-fold output |
| `fig10_confmat_regenerated.{pdf,png}` | corrected confusion-matrix figure (every panel sums to 494) |

## Scripts (run from the repository root: `python revision/<script>.py`)
| Script | Purpose | Reviewers |
|---|---|---|
| `stats_analysis.py` | descriptive and inferential statistics from the raw logs | all |
| `regen_fig9_confmat.py` | regenerate the confusion-matrix figure (no training) | 1 |
| `kfold_cv.py` | stratified k-fold cross-validation with 95% CIs | 2, 4, 5 |
| `expert_baseline.py --runs 30` | expert-optimized baseline with a fixed protocol | 2, 4 |
| `tier3_ablation.py` | 2×2 factorial: backbone (pretrained / from scratch) × scheduler (cosine / plateau) | 2, 3, 4, 5 |
| `external_eval.py --external <dir> --ckpt best_model.pth` | evaluation on an external dataset | 2, 3, 4, 5 |
| `collect_results.py` | gather the logs into `experiment_results.txt` | |
| `common_data.py` | shared loader, dataset, transforms and metrics | |

See `RUN_INSTRUCTIONS.md` for the one-command background runner.

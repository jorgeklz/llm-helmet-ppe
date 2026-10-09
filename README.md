# LLM-Generated CNNs for Helmet-Wearing Compliance Classification

Code, data and raw logs supporting the study submitted to *Discover Artificial Intelligence*.
Four large language models (ChatGPT, Claude, Gemini, Grok) wrote the PyTorch training
pipeline for a binary helmet / no-helmet classifier under three levels of prompt
instruction. Every generated script was run repeatedly on the same images and every run
was logged. The manuscript itself is not included.

**Interactive results:** [`docs/index.html`](docs/index.html) is an animated, bilingual (ES/EN)
page built only from the files in this repository. Once GitHub Pages is enabled
(Settings → Pages → *Deploy from a branch* → `main` / `/docs`) it is served at
<https://jorgeklz.github.io/llm-helmet-ppe/>.

## Study design

| | |
|---|---|
| Dataset | 972 images, balanced (486 helmet / 486 no helmet). 478 train, 494 test. 640×640 JPEG with a sibling `.txt` label whose first token is the class (`0` = helmet, `1` = no helmet). |
| Tier 1 | Basic instruction. |
| Tier 2 | Tier 1 plus the hyperparameter vector θ (dropout, two learning rates, batch size, hidden units). |
| Tier 3 | Tier 2 plus a prescribed architecture (from-scratch multi-scale CNN) and learning-rate schedule. |
| Runs | 20 per configuration (8 for Grok Tier 3). 228 runs in total. |

## Key results

Mean test accuracy over the logged runs (`revision/stats_summary.csv`):

| Tier | ChatGPT | Claude | Gemini | Grok |
|---|---|---|---|---|
| 1 | 0.958 | **0.967** | 0.964 | 0.957 |
| 2 | 0.963 | 0.958 | 0.963 | 0.957 |
| 3 | 0.963 | 0.962 | 0.576 ✗ | 0.591 ✗ |

- **Tier 3 collapse.** Gemini and Grok followed the prescribed from-scratch architecture
  and stayed near chance. ChatGPT and Claude replaced it with a pretrained ResNet18 and converged.
- **Ablation (2×2).** Pretrained ResNet18 converges under both schedulers (0.945–0.947);
  the from-scratch multi-scale network fails under both (0.53–0.57). The backbone, not the
  scheduler, explains the failure.
- **5-fold cross-validation:** accuracy 0.989 ± 0.007 (95% CI [0.980, 0.997]), AUC 0.9995.
- **Expert-optimized baseline** (fixed protocol, 30 runs): accuracy 0.968 ± 0.005, AUC 0.992.
- **Two-way ANOVA** (LLM × Tier) on accuracy: both main effects and the interaction are
  significant (all p < 10⁻¹⁵⁰).

## Repository layout

```
dataset/{train,test}/            images + .txt labels
tier{1,2,3}-{llm}.py             the twelve LLM-generated training scripts, unedited
tier{1,2,3}-{llm}_results.csv    one row per run
tier{1,2,3}-{llm}_summary.csv    mean / SD / min / max per metric
tier{1,2,3}-{llm}-output.txt     console log of every run
revision/                        statistics and reviewer-requested experiments (see revision/README.md)
docs/                            results website (index.html, data.js, build_data.py)
run_experiments.sh               runs k-fold CV, expert baseline and Tier-3 ablation
rerun_rest.sh                    re-runs only the baseline and the ablation
requirements.txt                 pinned Python dependencies
```

The raw logs use three column conventions (`Accuracy`/`accuracy`, `Recall+`/`recall_pos`, …).
`docs/build_data.py` shows the mapping. Loss values are reported as each script computed them,
so they are comparable within a script and not across LLMs.

## Reproducing

```bash
pip install -r requirements.txt

# statistics from the raw logs (no training, seconds)
python revision/stats_analysis.py

# reviewer-requested experiments (PyTorch; several hours on CPU)
mkdir -p revision/logs
nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &

# a single LLM-generated pipeline
python tier1-claude.py

# regenerate the website data after any change to the logs
python docs/build_data.py
```

Run everything from the repository root. See `revision/RUN_INSTRUCTIONS.md` for run sizes
and the optional external-dataset evaluation.

## Contact

Jorge Párraga-Álava · jorge.parraga@utm.edu.ec · Universidad Técnica de Manabí

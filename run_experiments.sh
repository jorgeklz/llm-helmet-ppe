#!/usr/bin/env bash
# =============================================================================
# One-command background runner for the Round-1 revision experiments.
# Run this on your Mac's Terminal (where PyTorch is available), NOT inside any
# sandbox. It runs the four reviewer-requested experiments and writes a single
# consolidated results file at revision/experiment_results.txt.
#
# LAUNCH (background, survives closing the terminal):
#     nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &
#
# CHECK PROGRESS:
#     tail -f revision/logs/run_all.log
#     cat revision/experiment_results.txt        # once finished
#
# Tunable run sizes (override on the command line, e.g. BASELINE_RUNS=10 nohup ...):
KFOLD_K=${KFOLD_K:-5}
KF_EH=${KF_EH:-3}; KF_EF=${KF_EF:-4}
BASELINE_RUNS=${BASELINE_RUNS:-30}
ABL_RUNS=${ABL_RUNS:-5}; ABL_EPOCHS=${ABL_EPOCHS:-20}
EXTERNAL_DIR=${EXTERNAL_DIR:-}        # set to a folder to also run external_eval
# =============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p revision/logs
LOG=revision/logs

echo "[$(date)] starting revision experiments"

# --- 1. locate or build a Python with torch -------------------------------
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch,torchvision,sklearn,cv2" >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ] && [ -x ".venv/bin/python" ] && ".venv/bin/python" -c "import torch" >/dev/null 2>&1; then
  PY=".venv/bin/python"
fi
if [ -z "$PY" ]; then
  echo "[setup] No Python with torch found. Creating .venv_torch and installing dependencies (needs internet)..."
  python3 -m venv .venv_torch || { echo "ERROR: could not create venv"; exit 1; }
  ./.venv_torch/bin/pip install --upgrade pip >/dev/null 2>&1
  ./.venv_torch/bin/pip install torch torchvision scikit-learn opencv-python numpy || {
    echo "ERROR: dependency install failed. Install torch manually, then re-run."; exit 1; }
  PY="./.venv_torch/bin/python"
fi
echo "[setup] using interpreter: $PY"
"$PY" -c "import torch;print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

run () {  # name  logfile  command...
  local name="$1"; local lf="$2"; shift 2
  echo "[$(date)] >>> $name"
  ( "$@" ) < /dev/null > "$lf" 2>&1
  echo "[$(date)] <<< $name (exit $?) -> $lf"
}

# --- 2. run the four experiments (sequential; each logged) ----------------
run "k-fold CV"          "$LOG/kfold.log"     "$PY" -u revision/kfold_cv.py --k "$KFOLD_K" --epochs_head "$KF_EH" --epochs_ft "$KF_EF"
run "expert baseline"    "$LOG/baseline.log"  "$PY" -u revision/expert_baseline.py --runs "$BASELINE_RUNS"
run "tier-3 ablation"    "$LOG/ablation.log"  "$PY" -u revision/tier3_ablation.py --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS"
if [ -n "$EXTERNAL_DIR" ]; then
  run "external eval"    "$LOG/external.log"  "$PY" -u revision/external_eval.py --external "$EXTERNAL_DIR" --ckpt best_model.pth
fi

# --- 3. collect results ---------------------------------------------------
"$PY" revision/collect_results.py
echo "[$(date)] ALL DONE. See revision/experiment_results.txt"

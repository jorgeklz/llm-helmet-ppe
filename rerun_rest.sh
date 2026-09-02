#!/usr/bin/env bash
# Re-run ONLY the two phases that crashed at Python startup (baseline, ablation).
# Fixes the "init_sys_streams: Bad file descriptor" crash by giving each process
# a clean stdin (< /dev/null) and unbuffered output (-u). k-fold already succeeded
# and is NOT repeated.
#
# LAUNCH:
#   nohup bash rerun_rest.sh > revision/logs/rerun.log 2>&1 &
# WATCH:
#   tail -F revision/logs/rerun.log revision/logs/baseline.log revision/logs/ablation.log
#
BASELINE_RUNS=${BASELINE_RUNS:-30}
ABL_RUNS=${ABL_RUNS:-5}; ABL_EPOCHS=${ABL_EPOCHS:-20}
cd "$(dirname "$0")"
PY="./.venv_torch/bin/python"; [ -x "$PY" ] || PY="python3"
echo "[$(date)] using $PY"; "$PY" -c "import torch;print('torch',torch.__version__)" < /dev/null

echo "[$(date)] >>> expert baseline ($BASELINE_RUNS runs)"
"$PY" -u revision/expert_baseline.py --runs "$BASELINE_RUNS" < /dev/null > revision/logs/baseline.log 2>&1
echo "[$(date)] <<< expert baseline (exit $?)"

echo "[$(date)] >>> tier-3 ablation ($ABL_RUNS runs x $ABL_EPOCHS epochs)"
"$PY" -u revision/tier3_ablation.py --runs "$ABL_RUNS" --epochs "$ABL_EPOCHS" < /dev/null > revision/logs/ablation.log 2>&1
echo "[$(date)] <<< tier-3 ablation (exit $?)"

"$PY" -u revision/collect_results.py < /dev/null
echo "[$(date)] ALL DONE -> revision/experiment_results.txt"

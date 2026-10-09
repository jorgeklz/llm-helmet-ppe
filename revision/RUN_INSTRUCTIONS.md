# How to run the revision experiments in the background

Open a terminal in the repository root and launch one command. PyTorch must be
available (the same environment where you ran the original tier*.py scripts). If
no torch is found, the runner creates a local .venv_torch and installs it (needs
internet).

## Launch (background; keeps running if you close the terminal)
```
mkdir -p revision/logs
nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &
```

## Watch progress
```
tail -f revision/logs/run_all.log
```

## When it finishes, the consolidated numbers are here
```
cat revision/experiment_results.txt
```
After a re-run, refresh the website data with `python docs/build_data.py`.

## Run sizes (optional overrides)
Defaults: k-fold k=5; expert baseline 30 runs; ablation 5 runs x 20 epochs.
The from-scratch ablation cells and the 30-run baseline are the slow parts on
CPU (possibly several hours). To do a quick first pass, launch with smaller sizes:
```
BASELINE_RUNS=10 ABL_RUNS=3 ABL_EPOCHS=15 nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &
```

## External dataset (optional, Reviewers 2/3/4/5)
If you have an independent test folder (same image + .txt label convention):
```
EXTERNAL_DIR="/path/to/external_images" nohup bash run_experiments.sh > revision/logs/run_all.log 2>&1 &
```

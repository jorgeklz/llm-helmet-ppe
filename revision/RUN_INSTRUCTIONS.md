# How to run the four experiments in the background (on your Mac)

Open Terminal, go to the project folder, and launch one command. PyTorch must be
available (the same environment where you ran the original tier*.py scripts). If
no torch is found, the runner creates a local .venv_torch and installs it (needs
internet).

## Launch (background; keeps running if you close Terminal)
```
cd "/Users/jorge/Library/CloudStorage/Dropbox/Papers/2026 - LLM helmet"
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
Send me that file (or paste its contents) and I will insert the numbers into the
response letter and the manuscript, and change the wording from "provided as a
reproducible script" to the actual results.

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

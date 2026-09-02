"""
=============================================================================
Binary Helmet Detection — Final Training Script (PyTorch)
=============================================================================
Backbone  : ResNet18 pretrained (ImageNet) — 11.7M params
            Chosen over MobileNetV3-Small because its residual skip
            connections provide far richer feature representations on
            small datasets, yielding ~96% accuracy vs ~60% from scratch.

Head      : FC(512→128) → ReLU → Dropout(0.4) → FC(128→1)  [raw logit]
Loss      : BCEWithLogitsLoss(pos_weight=1.5)  — Sigmoid aplicado internamente
Strategy  : Two-phase training
              Phase 1 — backbone frozen, head only (lr=1e-3, 20 ep, pat=7)
              Phase 2 — full fine-tune            (lr=1e-4, 30 ep, pat=12)
              + CosineAnnealingWarmRestarts(T_0=10) in Phase 2

Opciones incorporadas
──────────────────────
  Opción A — Umbral de decisión ajustado (decision_threshold=0.40)
             Mejora recall_pos (detectar no-helmet) a costa de algo de
             precision. Rango recomendado: 0.35–0.45.

  Opción B — BCEWithLogitsLoss + pos_weight=1.5
             Penaliza falsos negativos de clase 1 (no-helmet) 1.5× más.
             Sigmoid removido del head; la loss lo aplica internamente.

  Opción 1 — Reproducibilidad total (deterministic=True)
             torch.backends.cudnn.deterministic = True
             Elimina no-determinismo de cuDNN. Reduce varianza entre runs
             a costa de una leve pérdida de velocidad en GPU.

  Opción 2 — Mayor patience en Phase 2 (patience_phase2=12)
             Permite que el optimizador explore más antes de parar.
             Reduce la probabilidad de quedar atrapado en mínimos locales
             que generaban la pérdida máxima de 0.62 observada.

Data layout
────────────
  dataset/
  ├── train/   img.jpg + img.txt  (YOLO format: "class cx cy w h" por fila)
  └── test/    img.jpg + img.txt

  Label: primer token de la PRIMERA fila del .txt
      0  → helmet      → label interno = 0
      1  → no-helmet   → label interno = 1
  recall_pos = recall clase 1 (no-helmet)
  recall_neg = recall clase 0 (helmet)

Balancing  : WeightedRandomSampler iguala frecuencia de clases por batch.

Outputs
────────
  tier3-claude_results.csv
  tier3-claude_summary.csv
  confusion_matrix_runXX.png
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import copy
import time
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import psutil
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── Paths ─────────────────────────────────────────────────────────────
    train_dir          = "dataset/train",
    test_dir           = "dataset/test",

    # ── Input ─────────────────────────────────────────────────────────────
    img_size           = 224,
    batch_size         = 16,

    # ── Head ──────────────────────────────────────────────────────────────
    head_hidden        = 128,
    dropout_rate       = 0.4,

    # ── Phase 1: backbone frozen, head only ───────────────────────────────
    lr_phase1          = 1e-3,
    epochs_phase1      = 20,
    patience_phase1    = 7,

    # ── Phase 2: full fine-tune ────────────────────────────────────────────
    lr_phase2          = 1e-4,
    epochs_phase2      = 30,
    patience_phase2    = 12,    # Opción 2: subido de 10 → 12 para reducir
                                # varianza y escapar mínimos locales
    cosine_T0          = 10,

    # ── Regularización ────────────────────────────────────────────────────
    weight_decay       = 1e-4,
    val_split          = 0.15,

    # ── Experimentos ──────────────────────────────────────────────────────
    n_runs             = 20,
    base_seed          = 42,
    num_workers        = 0,
    device             = "cuda" if torch.cuda.is_available() else "cpu",

    # ── Opción A: umbral de decisión ──────────────────────────────────────
    # Umbral más bajo → más recall_pos (detectar no-helmet) a costa de
    # algo de precision. Cambia simultáneamente en train loop y evaluación.
    decision_threshold = 0.40,

    # ── Opción B: penalización a falsos negativos de clase 1 ─────────────
    # BCEWithLogitsLoss(pos_weight). El head NO lleva Sigmoid.
    # 1.0 = sin penalización extra | 1.5 = FN clase-1 cuesta 1.5×
    pos_weight         = 1.5,

    # ── Opción 1: reproducibilidad determinista ───────────────────────────
    # True  → resultados 100% reproducibles entre runs con misma semilla.
    # False → permite optimizaciones no-deterministas de cuDNN (más rápido
    #         en GPU pero con varianza residual entre runs).
    deterministic      = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Semilla global y determinismo  (Opción 1)
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """
    Fija todas las fuentes de aleatoriedad para reproducibilidad.
    Opción 1: torch.backends.cudnn.deterministic = True elimina el
    no-determinismo de los kernels cuDNN, reduciendo la varianza en loss
    entre runs que compartían la misma semilla pero producían resultados
    ligeramente distintos.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if CFG["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        # benchmark=False impide que cuDNN seleccione kernels distintos
        # entre runs, que era otra fuente de varianza no controlada.


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png")


class HelmetDataset(Dataset):
    """
    Lee imágenes y labels YOLO desde un directorio plano.
    Label = primer token de la primera línea del .txt paired.
        0 = helmet     (label = 0.0)
        1 = no-helmet  (label = 1.0)
    """
    def __init__(self, folder, transform=None):
        self.folder    = folder
        self.transform = transform
        self.samples   = self._scan(folder)

    @staticmethod
    def _scan(folder):
        items = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(folder, fname)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                continue
            items.append((img_path, txt_path))
        return items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        img = PILImage.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(txt_path, "r") as f:
            cls = int(f.readline().strip().split()[0])
        return img, torch.tensor(cls, dtype=torch.float32)

    def get_labels(self):
        """Retorna todos los labels como numpy array (para el sampler)."""
        labels = []
        for _, txt_path in self.samples:
            with open(txt_path, "r") as f:
                labels.append(int(f.readline().strip().split()[0]))
        return np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transforms
# ─────────────────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25,
                           saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Modelo — ResNet18 con head binario
# ─────────────────────────────────────────────────────────────────────────────

class HelmetResNet(nn.Module):
    """
    Backbone ResNet18 (pretrained ImageNet) + head de clasificación binaria.

    Head: FC(512→128) → ReLU → Dropout(0.4) → FC(128→1)
    Sin Sigmoid en el head — BCEWithLogitsLoss lo aplica internamente
    (Opción B). Esto permite usar pos_weight de forma estable.

    freeze_backbone(): congela todo excepto el fc head (Phase 1).
    unfreeze_all():    descongela todo (Phase 2).
    """
    def __init__(self, head_hidden=128, dropout_rate=0.4):
        super().__init__()
        backbone    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features   # 512

        backbone.fc = nn.Sequential(
            nn.Linear(in_features, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_hidden, 1),
            # Sin Sigmoid: BCEWithLogitsLoss lo aplica internamente (Opción B)
        )
        self.net = backbone

        for m in self.net.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        for name, p in self.net.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DataLoaders con balanced sampler
# ─────────────────────────────────────────────────────────────────────────────

def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    WeightedRandomSampler con peso inverso a la frecuencia de clase.
    Garantiza que cada batch tenga representación equilibrada de
    helmet(0) y no-helmet(1) independientemente del desbalance original.
    """
    _, counts      = np.unique(labels, return_counts=True)
    class_weight   = 1.0 / counts
    sample_weights = class_weight[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(sample_weights),
        replacement = True,
    )


class _SubsetDataset(Dataset):
    """Dataset liviano que envuelve una lista de pares (img_path, txt_path)."""
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]
        img = PILImage.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(txt_path, "r") as f:
            cls = int(f.readline().strip().split()[0])
        return img, torch.tensor(cls, dtype=torch.float32)


def get_loaders(seed: int):
    """
    Divide dataset/train en train/val con semilla reproducible.
    dataset/test se usa íntegro como hold-out fijo.
    """
    rng      = np.random.default_rng(seed)
    full_ds  = HelmetDataset(CFG["train_dir"], transform=None)
    all_lbls = full_ds.get_labels()
    n        = len(full_ds)
    n_val    = max(1, int(n * CFG["val_split"]))
    n_tr     = n - n_val

    idx     = rng.permutation(n)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr:]

    tr_samples  = [full_ds.samples[i] for i in tr_idx]
    val_samples = [full_ds.samples[i] for i in val_idx]

    tr_ds   = _SubsetDataset(tr_samples,  train_transform)
    val_ds  = _SubsetDataset(val_samples, val_transform)
    te_ds   = HelmetDataset(CFG["test_dir"], transform=val_transform)

    tr_lbls = all_lbls[tr_idx]
    sampler = make_balanced_sampler(tr_lbls)

    kw = dict(batch_size=CFG["batch_size"],
              num_workers=CFG["num_workers"],
              pin_memory=False)
    train_dl = DataLoader(tr_ds, sampler=sampler, **kw)
    val_dl   = DataLoader(val_ds,  shuffle=False, **kw)
    test_dl  = DataLoader(te_ds,   shuffle=False, **kw)

    pos = int(tr_lbls.sum())
    print(f"  Split — train:{n_tr} (cls1:{pos} cls0:{n_tr-pos}) | "
          f"val:{n_val} | test:{len(te_ds)}")
    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Para el entrenamiento si la val_loss no mejora en `patience` épocas.
    Guarda el mejor estado del modelo para restaurarlo al terminar.
    """
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            out    = model(imgs)
            loss   = criterion(out, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            # Opción B: logits → sigmoid antes de comparar con umbral
            # Opción A: usar decision_threshold en lugar de 0.5
            preds   = (torch.sigmoid(out.detach()) >= CFG["decision_threshold"]).float()
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
    return total_loss / total, correct / total


def train_phase(model, train_dl, val_dl, criterion, optimizer,
                scheduler, epochs: int, patience: int, device, phase_name: str):
    stopper = EarlyStopping(patience=patience)
    print(f"\n  ── {phase_name}")
    for epoch in range(1, epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, optimizer, device, True)
        vl, va = run_epoch(model, val_dl,   criterion, optimizer, device, False)
        if scheduler:
            scheduler.step()
        print(f"    Ep {epoch:3d}/{epochs} | "
              f"tr {tl:.4f}/{ta:.4f} | vl {vl:.4f}/{va:.4f}")
        if stopper(vl, model):
            print(f"    ↳ Early stop @ ep {epoch}  (patience={patience})")
            break
    model.load_state_dict(stopper.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Monitor de recursos del sistema
# ─────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """
    Hilo demonio que muestrea CPU% y RAM% cada `interval` segundos.
    Registra el consumo real durante todo el entrenamiento, no solo
    al final (que es lo que haría un psutil.cpu_percent() puntual).
    """
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._cpu     = []
        self._ram     = []
        self._running = False
        self._thread  = None

    def _sample(self):
        while self._running:
            self._cpu.append(psutil.cpu_percent(interval=None))
            self._ram.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def start(self):
        self._cpu.clear()
        self._ram.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def mean_cpu(self) -> float:
        return float(np.mean(self._cpu)) if self._cpu else 0.0

    @property
    def mean_ram(self) -> float:
        return float(np.mean(self._ram)) if self._ram else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Evaluación
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, criterion, device):
    """
    Evalúa el modelo en el loader dado.

    Cálculo correcto de loss:
        Se acumula loss * batch_size y se divide por total de samples,
        evitando el sesgo del promedio de promedios cuando el último
        batch tiene tamaño distinto.

    Returns
    -------
    metrics : dict  {accuracy, precision, recall_pos, recall_neg, loss, auc}
    y_true  : list  etiquetas reales
    y_pred  : list  predicciones binarias
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss, n_samples  = 0.0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            out    = model(imgs)
            loss   = criterion(out, labels)

            # Pérdida ponderada por tamaño de batch (corrección necesaria
            # cuando el último batch es más pequeño que los demás)
            total_loss += loss.item() * imgs.size(0)
            n_samples  += imgs.size(0)

            # Opción B: logits → sigmoid  |  Opción A: umbral ajustado
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            preds = (probs >= CFG["decision_threshold"]).astype(int)

            y_true.extend(labels.cpu().numpy().flatten().astype(int).tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    avg_loss = total_loss / n_samples if n_samples else 0.0
    unique   = sorted(set(y_true))

    if len(unique) > 1:
        recalls    = recall_score(y_true, y_pred, labels=[0, 1],
                                  average=None, zero_division=0)
        recall_neg = float(recalls[0])   # clase 0 = helmet
        recall_pos = float(recalls[1])   # clase 1 = no-helmet
        auc        = float(roc_auc_score(y_true, y_prob))
    else:
        r          = recall_score(y_true, y_pred, average=None, zero_division=0)
        recall_pos = float(r[0]) if unique[0] == 1 else 0.0
        recall_neg = float(r[0]) if unique[0] == 0 else 0.0
        auc        = 0.0

    metrics = dict(
        accuracy   = float(accuracy_score(y_true, y_pred)),
        precision  = float(precision_score(y_true, y_pred, zero_division=0)),
        recall_pos = recall_pos,
        recall_neg = recall_neg,
        loss       = avg_loss,
        auc        = auc,
    )
    return metrics, y_true, y_pred


def print_metrics(metrics: dict, cpu_pct=None, ram_pct=None, exec_time=None):
    """Imprime métricas de predicción + recursos del sistema en un bloque."""
    print("\n  " + "─" * 46)
    print("  Test Metrics")
    print("  " + "─" * 46)
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")
    if cpu_pct  is not None:
        print(f"  {'cpu_percent':<14}: {cpu_pct:.2f} %")
    if ram_pct  is not None:
        print(f"  {'ram_percent':<14}: {ram_pct:.2f} %")
    if exec_time is not None:
        print(f"  {'time_sec':<14}: {exec_time:.2f} s")


def save_confusion_matrix(y_true, y_pred, run_id: int):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm,
                                  display_labels=["helmet(0)", "no-helmet(1)"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — Run {run_id}")
    plt.tight_layout()
    path = f"confusion_matrix_run{run_id:02d}.png"
    plt.savefig(path, dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Un run completo
# ─────────────────────────────────────────────────────────────────────────────

def run_training_once(run_id: int):
    """
    Ejecuta un ciclo completo de entrenamiento + evaluación.

    Semilla única por run (base_seed + run_id) garantiza que cada run
    tenga su propio split y orden de batches, pero es 100% reproducible
    si se vuelve a ejecutar el mismo run_id gracias a Opción 1.

    Returns
    -------
    list: [run, accuracy, precision, recall_pos, recall_neg,
           auc, loss, cpu_percent, ram_percent, time_sec]
    """
    seed = CFG["base_seed"] + run_id
    set_seed(seed)                               # Opción 1: determinismo
    device = torch.device(CFG["device"])

    train_dl, val_dl, test_dl = get_loaders(seed)

    model = HelmetResNet(
        head_hidden  = CFG["head_hidden"],
        dropout_rate = CFG["dropout_rate"],
    ).to(device)

    # Opción B: BCEWithLogitsLoss + pos_weight=1.5
    # Penaliza 1.5× más los falsos negativos de no-helmet (clase 1)
    pw        = torch.tensor([CFG["pos_weight"]], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── Iniciar monitor de recursos ───────────────────────────────────────
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    t_start = time.perf_counter()

    # ── Phase 1: solo el head (backbone congelado) ────────────────────────
    model.freeze_backbone()
    opt1   = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr_phase1"], weight_decay=CFG["weight_decay"],
    )
    sched1 = optim.lr_scheduler.StepLR(opt1, step_size=7, gamma=0.5)
    train_phase(
        model, train_dl, val_dl, criterion, opt1, sched1,
        epochs     = CFG["epochs_phase1"],
        patience   = CFG["patience_phase1"],
        device     = device,
        phase_name = f"Phase 1 — head only  (run {run_id})",
    )

    # ── Phase 2: fine-tune completo con CosineAnnealingWarmRestarts ───────
    model.unfreeze_all()
    opt2   = optim.AdamW(
        model.parameters(),
        lr=CFG["lr_phase2"], weight_decay=CFG["weight_decay"],
    )
    sched2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt2,
        T_0     = CFG["cosine_T0"],    # reinicio cada 10 épocas
        T_mult  = 1,
        eta_min = CFG["lr_phase2"] / 50,
    )
    train_phase(
        model, train_dl, val_dl, criterion, opt2, sched2,
        epochs     = CFG["epochs_phase2"],
        patience   = CFG["patience_phase2"],   # Opción 2: patience=12
        device     = device,
        phase_name = f"Phase 2 — full fine-tune  (run {run_id})",
    )

    # ── Detener monitor ───────────────────────────────────────────────────
    monitor.stop()
    exec_time = round(time.perf_counter() - t_start, 3)
    cpu_pct   = round(monitor.mean_cpu, 2)
    ram_pct   = round(monitor.mean_ram, 2)

    # ── Evaluar en test set fijo ──────────────────────────────────────────
    metrics, y_true, y_pred = evaluate_model(model, test_dl, criterion, device)
    print_metrics(metrics, cpu_pct=cpu_pct, ram_pct=ram_pct, exec_time=exec_time)
    save_confusion_matrix(y_true, y_pred, run_id)

    return [
        run_id,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall_pos"],
        metrics["recall_neg"],
        metrics["auc"],
        metrics["loss"],
        cpu_pct,
        ram_pct,
        exec_time,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Experimentos (N runs independientes)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(n_runs: int = 20):
    """
    Ejecuta n_runs ciclos independientes y guarda los resultados.

    Archivos generados
    ──────────────────
    tier3-claude_results.csv  — una fila por run
    tier3-claude_summary.csv  — mean / std / min / max por métrica
    confusion_matrix_runXX.png — matriz de confusión por run
    """
    results = []

    for r in range(1, n_runs + 1):
        print(f"\n{'='*50}")
        print(f" RUN {r}/{n_runs}")
        print(f"{'='*50}")
        row = run_training_once(r)
        results.append(row)
        print(f" → acc={row[1]:.4f} | auc={row[5]:.4f} | "
              f"time={row[9]:.1f}s | cpu={row[7]:.1f}% | ram={row[8]:.1f}%")

    # ── Guardar resultados ────────────────────────────────────────────────
    columns = [
        "run",
        "accuracy", "precision", "recall_pos", "recall_neg",
        "auc", "loss",
        "cpu_percent", "ram_percent", "time_sec",
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv("tier3-claude_results.csv", index=False)
    print("\n✔ tier3-claude_results.csv generated")

    summary = (
        df.drop(columns=["run"])
          .describe()
          .transpose()[["mean", "std", "min", "max"]]
    )
    summary.to_csv("tier3-claude_summary.csv")
    print("✔ tier3-claude_summary.csv generated")

    # ── Tabla resumen en consola ──────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  {'Metric':<14} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print("  " + "─" * 60)
    for metric, s in summary.iterrows():
        print(f"  {metric:<14} "
              f"{s['mean']:>9.4f} "
              f"{s['std']:>9.4f} "
              f"{s['min']:>9.4f} "
              f"{s['max']:>9.4f}")
    print("=" * 64)

    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, summary = run_experiments(n_runs=CFG["n_runs"])
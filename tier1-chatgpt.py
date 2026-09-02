# Tier 1 - ChatGPT
import os
import cv2
import numpy as np
import pandas as pd
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# =========================================================
# 1. Load images manually
# =========================================================
def load_data(directory, size=(224, 224)):
    images, labels = [], []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(directory, filename)
            label_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(label_path):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                images.append(img)

                with open(label_path, 'r') as f:
                    label = int(f.readline().split()[0])
                    labels.append(label)

    return np.array(images), np.array(labels)

BASE_PATH = './dataset'
X_train, y_train = load_data(os.path.join(BASE_PATH, 'train'))
X_test, y_test = load_data(os.path.join(BASE_PATH, 'test'))
print("Train:", X_train.shape, " Test:", X_test.shape)

# =========================================================
# 2. Transforms
# =========================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def apply_transform(X):
    return torch.stack([transform(img) for img in X])

X_train_t = apply_transform(X_train)
X_test_t = apply_transform(X_test)

y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# =========================================================
# 3. Build model: ResNet50 binary classifier
# =========================================================
def build_model():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1)
    )
    return model

# =========================================================
# 4. Unfreeze top layers
# =========================================================
def unfreeze_top(model, layers_to_unfreeze=20):
    layers = list(model.layer4.parameters())
    for p in layers[-layers_to_unfreeze:]:
        p.requires_grad = True
    return model

# =========================================================
# 5. Train model
# =========================================================
def train_model(model, X, y, lr=1e-4, epochs=5, batch=32):
    device = "cpu"
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()
    N = X.shape[0]

    for ep in range(epochs):
        for i in range(0, N, batch):
            xb = X[i:i+batch].to(device)
            yb = y[i:i+batch].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model

# =========================================================
# 6. Predict
# =========================================================
def predict(model, X, batch=32):
    device = "cpu"
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch].to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    return np.array(preds)

# =========================================================
# Loss
# =========================================================
# BCELoss with default mean reduction implements exactly:
# L = -(1/N) * sum[ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
# where p_i are predicted probabilities in (0,1).
bce_loss = nn.BCELoss(reduction='mean')

# =========================================================
# 7. TIER 1 - ChatGPT FUNCTION
# =========================================================
def tier1_chatgpt(X_train_t, y_train_t, X_test_t, y_test):
    start_time = time.time()

    # Phase 1
    model = build_model()
    model = train_model(model, X_train_t, y_train_t, lr=1e-4, epochs=5)

    # Phase 2
    model = unfreeze_top(model, layers_to_unfreeze=20)
    model = train_model(model, X_train_t, y_train_t, lr=1e-5, epochs=5)

    # Predictions
    y_prob = predict(model, X_test_t)
    y_pred = (y_prob > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_pos = recall_score(y_test, y_pred, pos_label=1)
    rec_neg = recall_score(y_test, y_pred, pos_label=0)

    # Correct BCE
    y_test_t2 = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    y_prob_t2 = torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1)
    loss = bce_loss(y_prob_t2, y_test_t2).item()

    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0

    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    disco = psutil.disk_usage('/').percent
    run_time = time.time() - start_time

    return [acc, prec, rec_pos, rec_neg, loss, auc, cpu, mem, disco, run_time]

# =========================================================
# 8. MAIN LOOP — 20 RUNS
# =========================================================
ITERATIONS = 1
results = []

metrics_labels = [
    "Accuracy", "Precision", "Recall+",
    "Recall-", "Loss", "AUC",
    "CPU %", "RAM %", "Disk %", "Exec Time (s)"
]

for run in range(ITERATIONS):
    print(f"\n================ RUN {run+1}/{ITERATIONS} ================")

    r = tier1_chatgpt(X_train_t, y_train_t, X_test_t, y_test)

    print(
        f"ACC={r[0]:.4f}  PREC={r[1]:.4f}  R+={r[2]:.4f}  "
        f"R-={r[3]:.4f}  LOSS={r[4]:.4f}  AUC={r[5]:.4f}  "
        f"CPU={r[6]:.1f}% MEM={r[7]:.1f}% DISK={r[8]:.1f}% "
        f"TIME={r[9]:.2f}s"
    )

    results.append(r)

# =========================================================
# 9. SUMMARY
# =========================================================
results = np.array(results)
mean_vals = results.mean(axis=0)
std_vals = results.std(axis=0)
min_vals = results.min(axis=0)
max_vals = results.max(axis=0)

print("\n================ SUMMARY (20 RUNS) ================\n")
for i, m in enumerate(metrics_labels):
    print(f"{m:15}: Mean={mean_vals[i]:.4f}  |  Std={std_vals[i]:.4f}")

# =========================================================
# 10. Save results to CSV
# =========================================================
df_results = pd.DataFrame(results, columns=metrics_labels)
df_results.to_csv('tier1-chatgpt_results.csv', index=False)
print("\nResults saved to 'tier1-chatgpt_results.csv'")

summary_dict = {
    'Metric': metrics_labels,
    'Mean': mean_vals,
    'Std': std_vals,
    'Min': min_vals,
    'Max': max_vals
}

df_summary = pd.DataFrame(summary_dict)
df_summary.to_csv('tier1-chatgpt_summary.csv', index=False)
print("Summary saved to 'tier1-chatgpt_summary.csv'")

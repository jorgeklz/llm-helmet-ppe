"""
Shared data utilities for the revision experiments.
Matches the data convention of the original pipeline scripts:
  ./dataset/train and ./dataset/test, each containing image files with a
  sibling YOLO-style .txt whose first token is the class id
  (label convention: 1 = no-helmet / non-compliance = positive class,
   0 = helmet / compliance = negative class).
"""
import os, numpy as np, cv2, torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = ('.jpg', '.png', '.jpeg')
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

def load_data(directory, size=(224, 224)):
    images, labels = [], []
    for fn in os.listdir(directory):
        if fn.lower().endswith(IMG_EXT):
            ip = os.path.join(directory, fn)
            lp = os.path.splitext(ip)[0] + '.txt'
            if os.path.exists(lp):
                img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                images.append(img)
                with open(lp) as f:
                    labels.append(int(f.readline().split()[0]))
    return np.array(images), np.array(labels)

class HelmetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i]
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32)

train_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def metrics_from_logits(logits, y_true):
    """logits: 1-D tensor of raw scores; returns dict of metrics."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    p = torch.sigmoid(torch.as_tensor(logits)).numpy().ravel()
    yhat = (p >= 0.5).astype(int); y = np.asarray(y_true).astype(int).ravel()
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y))
    prec = tp / max(1, tp + fp); rec_pos = tp / max(1, tp + fn); rec_neg = tn / max(1, tn + fp)
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float('nan')
    return dict(accuracy=acc, precision=prec, recall_pos=rec_pos, recall_neg=rec_neg,
                auc=auc, tp=tp, tn=tn, fp=fp, fn=fn)

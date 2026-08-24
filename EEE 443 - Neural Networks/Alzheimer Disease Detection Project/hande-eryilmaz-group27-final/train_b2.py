import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = "data"

RAW_CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

# 4 → 3 CLASS MERGE
MERGE_MAP = {
    "Mild Dementia": "Dementia",
    "Moderate Dementia": "Dementia",
    "Non Demented": "NonDemented",
    "Very mild Dementia": "VeryMild"
}

CLASS_NAMES = ["Dementia", "NonDemented", "VeryMild"]


# =====================================================
# PATIENT ID PARSER
# =====================================================
def extract_pid(filename):
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return parts[0]


# =====================================================
# LOAD PATIENT FILE STRUCTURE
# =====================================================
def load_patient_index():
    patients = {}

    for raw_cls in RAW_CLASS_NAMES:
        merged = MERGE_MAP[raw_cls]
        cls_idx = CLASS_NAMES.index(merged)

        folder = os.path.join(DATA_DIR, raw_cls)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".jpg"):
                continue

            pid = extract_pid(fname)
            path = os.path.join(folder, fname)

            if pid not in patients:
                patients[pid] = []

            patients[pid].append((path, cls_idx))

    return patients


# =====================================================
# DATASET CLASS
# =====================================================
class SliceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =====================================================
# MAIN
# =====================================================
def main():
    print(f"🔥 Using device: {DEVICE}")

    # Load patient → slice map
    patient_files = load_patient_index()

    # Count merged classes
    merged_counts = {c: 0 for c in CLASS_NAMES}
    for pid, lst in patient_files.items():
        merged_counts[CLASS_NAMES[lst[0][1]]] += 1

    print("\n📌 Patient-level distribution:")
    for c in CLASS_NAMES:
        print(f"{c}: {merged_counts[c]} patients")

    patients = list(patient_files.keys())
    total = len(patients)

    # =====================================================
    # PATIENT SPLIT (70/15/15)
    # =====================================================
    random.shuffle(patients)
    n_train = int(total * 0.70)
    n_val = int(total * 0.15)

    train_p = patients[:n_train]
    val_p = patients[n_train:n_train+n_val]
    test_p = patients[n_train+n_val:]

    print(f"\n📌 PATIENT SPLIT:")
    print(f"Train: {len(train_p)}")
    print(f"Val:   {len(val_p)}")
    print(f"Test:  {len(test_p)}")

    # -----------------------------------------------------
    # COLLECT SLICES
    # -----------------------------------------------------
    def collect(p_list):
        s = []
        for pid in p_list:
            s.extend(patient_files[pid])
        return s

    train_s = collect(train_p)
    val_s = collect(val_p)
    test_s = collect(test_p)

    print(f"→ Train slices: {len(train_s)}")
    print(f"→ Val slices:   {len(val_s)}")
    print(f"→ Test slices:  {len(test_s)}\n")

    # Show TEST SET PATIENT distribution
    test_dist = {c: 0 for c in CLASS_NAMES}
    for pid in test_p:
        test_dist[CLASS_NAMES[patient_files[pid][0][1]]] += 1

    print("📌 TEST SET PATIENT DISTRIBUTION")
    for c in CLASS_NAMES:
        print(f"{c}: {test_dist[c]}")
    print()

    # =====================================================
    # TRANSFORMS
    # =====================================================
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = SliceDataset(train_s, train_tf)
    val_ds = SliceDataset(val_s, test_tf)
    test_ds = SliceDataset(test_s, test_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================
    class_counts = np.bincount([c for _, c in train_s], minlength=len(CLASS_NAMES))
    weights = 1.0 / (class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Class weights:", weights, "\n")

    # =====================================================
    # MODEL
    # =====================================================
    model = timm.create_model("efficientnet_b2", pretrained=True, num_classes=3)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    print("🚀 STARTING TRAINING...\n")

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    def evaluate(loader):
        model.eval()
        total, correct, totloss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                totloss += loss.item()

                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return correct / total, totloss / len(loader)

    for epoch in range(1, 11):
        model.train()
        correct, total, totloss = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            totloss += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        train_loss = totloss / len(train_loader)
        val_acc, val_loss = evaluate(val_loader)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch {epoch}/10 | "
              f"TrainAcc={train_acc:.4f} Loss={train_loss:.4f} | "
              f"ValAcc={val_acc:.4f} Loss={val_loss:.4f}")

    # =====================================================
    # PATIENT-LEVEL TEST (majority vote)
    # =====================================================
    print("\n🎯 PATIENT-LEVEL TESTING...\n")

    patient_preds = {pid: [] for pid in test_p}
    patient_trues = {}

    model.eval()
    with torch.no_grad():
        for pid in test_p:
            slices = patient_files[pid]
            true_lbl = slices[0][1]
            patient_trues[pid] = true_lbl

            for (fpath, lbl) in slices:
                img = Image.open(fpath).convert("RGB")
                img = test_tf(img).unsqueeze(0).to(DEVICE)

                out = model(img)
                pred = out.argmax(1).item()
                patient_preds[pid].append(pred)

    final_preds, final_trues = [], []

    for pid in test_p:
        votes = patient_preds[pid]
        maj = max(set(votes), key=votes.count)
        final_preds.append(maj)
        final_trues.append(patient_trues[pid])

    # classification report
    rep = classification_report(final_trues, final_preds, target_names=CLASS_NAMES, zero_division=0)
    print(rep)

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    os.makedirs("results/b2", exist_ok=True)

    with open("results/b2/classification_report.txt", "w") as f:
        f.write(rep)

    # CONFUSION MATRIX
    cm = confusion_matrix(final_trues, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("EfficientNet-B2 Confusion Matrix (Patient-Level)")
    plt.savefig("results/b2/confusion_matrix.png")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("results/b2/accuracy_curve.png")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("results/b2/loss_curve.png")
    plt.close()

    print("\n✅ DONE! All results saved to results/b2/\n")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()
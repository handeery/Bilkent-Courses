import os
import torch
import timm
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================
# CLASS MAPPING (4 → 3)
# ============================================================
RAW_CLASSES = [
    "Mild Dementia",       # → Dementia
    "Moderate Dementia",   # → Dementia
    "Non Demented",        # → NonDemented
    "Very mild Dementia"   # → VeryMild
]

MAPPED_CLASSES = ["Dementia", "NonDemented", "VeryMild"]
CLASS_MAP = {
    0: 0,   # Mild       -> Dementia
    1: 0,   # Moderate   -> Dementia
    2: 1,   # NonDemented
    3: 2    # VeryMild
}

# ============================================================
# PATIENT ID EXTRACTION
# ============================================================
def extract_pid(path):
    fname = os.path.basename(path)
    parts = fname.split("_")
    return parts[0] + "_" + parts[1]

# ============================================================
# LOAD ALL SLICES → pid → mapped label
# ============================================================
def load_patient_index(data_dir="data"):
    patient_to_slices = defaultdict(list)

    for raw_idx, folder in enumerate(RAW_CLASSES):
        full_folder = os.path.join(data_dir, folder)
        if not os.path.isdir(full_folder):
            continue

        mapped_label = CLASS_MAP[raw_idx]

        for fname in os.listdir(full_folder):
            if not fname.lower().endswith(".jpg"):
                continue

            path = os.path.join(full_folder, fname)
            pid = extract_pid(fname)

            patient_to_slices[pid].append((path, mapped_label))

    return patient_to_slices

# ============================================================
# SLICE DATASET
# ============================================================
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
        return img, label, path

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"🔥 Using device: {DEVICE}")

    # LOAD DATA
    patient_files = load_patient_index("data")
    patients = list(patient_files.keys())
    total_patients = len(patients)

    print(f"📌 Found {total_patients} patients")

    # SPLIT (70/15/15)
    np.random.shuffle(patients)

    n_train = int(0.70 * total_patients)
    n_val = int(0.15 * total_patients)

    train_p = patients[:n_train]
    val_p = patients[n_train:n_train + n_val]
    test_p = patients[n_train + n_val:]

    print(f"Train patients: {len(train_p)}")
    print(f"Val patients:   {len(val_p)}")
    print(f"Test patients:  {len(test_p)}")

    # Gather slices
    def collect(pids):
        s = []
        for pid in pids:
            s.extend(patient_files[pid])
        return s

    train_s = collect(train_p)
    val_s = collect(val_p)

    print(f"→ Train slices: {len(train_s)}")
    print(f"→ Val slices:   {len(val_s)}")

    # ============================================================
    # TRANSFORMS
    # ============================================================
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

    # DATASETS
    train_ds = SliceDataset(train_s, train_tf)
    val_ds = SliceDataset(val_s, test_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # ============================================================
    # MODEL
    # ============================================================
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=3)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # ============================================================
    # TRAINING
    # ============================================================
    print("\n🚀 STARTING TRAINING...\n")

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    def evaluate(loader):
        model.eval()
        correct, total, total_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                out = model(imgs)

                loss = criterion(out, labels)
                total_loss += loss.item()

                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / total, total_loss / len(loader)

    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        val_acc, val_loss = evaluate(val_loader)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} | TrainAcc={train_acc:.4f} Loss={train_loss:.4f} | "
              f"ValAcc={val_acc:.4f} Loss={val_loss:.4f}")

    # ============================================================
    # PATIENT-LEVEL TESTING
    # ============================================================
    print("\n🎯 PATIENT-LEVEL TESTING...\n")

    # Build test dataset with path outputs
    test_samples = []
    for pid in test_p:
        test_samples.extend(patient_files[pid])

    test_ds = SliceDataset(test_samples, test_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    pid_probs = defaultdict(list)
    pid_labels = {}

    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1).cpu().numpy()

            for p, lbl, pr in zip(paths, labels, probs):
                pid = extract_pid(p)
                pid_labels[pid] = lbl
                pid_probs[pid].append(pr)

    # Aggregate slice → patient
    final_preds, final_trues = [], []

    for pid in pid_probs.keys():
        avg_prob = np.mean(pid_probs[pid], axis=0)
        pred_class = np.argmax(avg_prob)

        final_preds.append(pred_class)
        final_trues.append(pid_labels[pid])

    # REPORT
    rep = classification_report(final_trues, final_preds, target_names=MAPPED_CLASSES, zero_division=0)
    print(rep)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    os.makedirs("results/vit", exist_ok=True)

    with open("results/vit/classification_report.txt", "w") as f:
        f.write(rep)

    cm = confusion_matrix(final_trues, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=MAPPED_CLASSES, yticklabels=MAPPED_CLASSES)
    plt.title("ViT Patient-Level Confusion Matrix")
    plt.savefig("results/vit/confusion_matrix.png")
    plt.close()

    # Curves
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.title("ViT Accuracy Curve")
    plt.legend()
    plt.savefig("results/vit/accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.title("ViT Loss Curve")
    plt.legend()
    plt.savefig("results/vit/loss_curve.png")
    plt.close()

    print("🎉 ALL DONE! Patient-level ViT results saved → results/vit/")


# RUN
if __name__ == "__main__":
    main()
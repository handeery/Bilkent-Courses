import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10

CLASSES = ["Dementia", "NonDemented", "VeryMild"]
CLASS_TO_IDX = {
    "Mild Dementia": 0,
    "Moderate Dementia": 0,
    "Non Demented": 1,
    "Very mild Dementia": 2
}

print(f"🔥 Using device: {DEVICE}")


# ------------------------------------------------------------
# PATIENT-ID PARSER
# ------------------------------------------------------------
def get_patient_id(fname):
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


# ------------------------------------------------------------
# LOAD ALL SLICES
# ------------------------------------------------------------
def load_dataset():
    patient_files = {}

    for folder in os.listdir(DATA_DIR):
        cls_folder = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(cls_folder):
            continue
        if folder not in CLASS_TO_IDX:
            continue

        label = CLASS_TO_IDX[folder]

        for fname in os.listdir(cls_folder):
            if not fname.lower().endswith(".jpg"):
                continue

            full = os.path.join(cls_folder, fname)
            pid = get_patient_id(fname)

            if pid not in patient_files:
                patient_files[pid] = []

            patient_files[pid].append((full, label))

    return patient_files


# ------------------------------------------------------------
# DATASET CLASS
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    patient_files = load_dataset()

    # Count patients per class BEFORE split
    class_patient_count = {0: 0, 1: 0, 2: 0}
    for pid, slices in patient_files.items():
        lbl = slices[0][1]
        class_patient_count[lbl] += 1

    print("\n📌 Patient-level distribution:")
    print(f"Dementia: {class_patient_count[0]} patients")
    print(f"NonDemented: {class_patient_count[1]} patients")
    print(f"VeryMild: {class_patient_count[2]} patients\n")

    patients = list(patient_files.keys())
    y = [patient_files[pid][0][1] for pid in patients]

    # Stratified split
    train_pids, temp_pids, y_train, y_temp = train_test_split(
        patients, y, test_size=0.30, random_state=42, stratify=y
    )
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("📌 PATIENT SPLIT:")
    print(f"Train: {len(train_pids)}")
    print(f"Val:   {len(val_pids)}")
    print(f"Test:  {len(test_pids)}")

    # Build slice sets
    def build_samples(pid_list):
        samples = []
        for pid in pid_list:
            samples.extend(patient_files[pid])
        return samples

    train_samples = build_samples(train_pids)
    val_samples = build_samples(val_pids)
    test_samples = build_samples(test_pids)

    print(f"→ Train slices: {len(train_samples)}")
    print(f"→ Val slices:   {len(val_samples)}")
    print(f"→ Test slices:  {len(test_samples)}\n")

    # Test set patient distribution
    test_lbls = [patient_files[pid][0][1] for pid in test_pids]
    print("📌 TEST SET CLASS DISTRIBUTION")
    print(f"Dementia: {test_lbls.count(0)}")
    print(f"NonDemented: {test_lbls.count(1)}")
    print(f"VeryMild: {test_lbls.count(2)}\n")

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])
    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = SliceDataset(train_samples, tf_train)
    val_ds = SliceDataset(val_samples, tf_test)
    test_ds = SliceDataset(test_samples, tf_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(DEVICE)

    # Weighted CE
    class_counts = np.array([
        class_patient_count[0],
        class_patient_count[1],
        class_patient_count[2]
    ], dtype=float)

    weights = 1.0 / class_counts
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {weights}\n")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    print("🚀 STARTING TRAINING...\n")

    # TRAIN LOOP
    for epoch in range(1, EPOCHS + 1):

        # ---- TRAIN ----
        model.train()
        correct = 0
        total = 0
        total_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)

                out = model(imgs)
                loss = criterion(out, labels)
                total_loss += loss.item()

                pred = out.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss = total_loss / len(val_loader)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"TrainAcc={train_acc:.4f} Loss={train_loss:.4f} | "
            f"ValAcc={val_acc:.4f} Loss={val_loss:.4f}"
        )

    # ------------------------------------------------
    # PATIENT-LEVEL EVALUATION
    # ------------------------------------------------
    patient_probs = {pid: [] for pid in test_pids}
    patient_true = {pid: None for pid in test_pids}

    model.eval()
    sample_pointer = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1).cpu().numpy()

            batch_size = len(labels)
            batch_files = test_samples[sample_pointer: sample_pointer + batch_size]
            sample_pointer += batch_size

            for i, (path, lbl) in enumerate(batch_files):
                pid = get_patient_id(os.path.basename(path))
                patient_probs[pid].append(probs[i])
                patient_true[pid] = lbl

    final_preds = []
    final_trues = []

    for pid in test_pids:
        preds = patient_probs[pid]
        if len(preds) == 0:
            continue
        avg_prob = np.mean(preds, axis=0)
        final_preds.append(np.argmax(avg_prob))
        final_trues.append(patient_true[pid])

    report = classification_report(
        final_trues,
        final_preds,
        target_names=CLASSES
    )
    print("\n🎯 Patient-level evaluation:\n")
    print(report)

    # Save outputs
    os.makedirs("results/resnet", exist_ok=True)

    # Save classification_report.txt
    with open("results/resnet/classification_report.txt", "w") as f:
        f.write(report)

    # Accuracy plot
    plt.plot(train_acc_hist, label="Train")
    plt.plot(val_acc_hist, label="Val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("results/resnet/accuracy.png")
    plt.close()

    # Loss plot
    plt.plot(train_loss_hist, label="Train")
    plt.plot(val_loss_hist, label="Val")
    plt.legend()
    plt.title("Loss")
    plt.savefig("results/resnet/loss.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(final_trues, final_preds)
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix")
    plt.savefig("results/resnet/confusion_matrix.png")
    plt.close()

    print("\n🖼 Saved all outputs in results/resnet/\n")


if __name__ == "__main__":
    main()
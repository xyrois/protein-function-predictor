from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    MODEL_SAVE_PATH,
)

from model import ProteinClassifier


# Paths
CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = CURRENT_DIR.parent / "models"
RESULTS_DIR = CURRENT_DIR.parent / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# Label Mapping
LABEL_MAPPING_PATH = (
    MODELS_DIR / "label_mapping.json"
)

with open(
    LABEL_MAPPING_PATH,
    "r",
) as f:

    mapping = json.load(f)


id_to_label = {
    int(k): v
    for k, v in mapping["id_to_label"].items()
}


print("\nLabel mapping:")

for class_id, label in id_to_label.items():

    print(
        f"{class_id} -> {label}"
    )


# Load Dataset
print("\nLoading test dataset...")

dataset = load_from_disk(
    str(PROCESSED_DIR)
)

test_dataset = dataset["test"]

print(
    f"Test samples: {len(test_dataset):,}"
)



# Verify Dataset
print("\nDataset features:")

print(
    test_dataset.features
)

print("\nExample:")

print(
    test_dataset[0]
)


# DataLoader
test_dataset.set_format(
    type="torch",
    columns=[
        "label",
        "input_ids",
        "attention_mask",
    ],
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Model
print("\nLoading model...")

model = ProteinClassifier()

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

criterion = nn.CrossEntropyLoss()



# Evaluation
print("\nEvaluating...\n")

all_predictions = []
all_labels = []

running_loss = 0.0


with torch.no_grad():

    progress = tqdm(
        test_loader,
        desc="Evaluating",
    )

    for batch in progress:

        # Move tensors to device
        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)

        # Forward pass
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Loss
        loss = criterion(
            logits,
            labels,
        )

        running_loss += loss.item()

        # Predictions
        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# Metrics
test_loss = (
    running_loss /
    len(test_loader)
)


accuracy = accuracy_score(
    all_labels,
    all_predictions,
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
)


# Overall Results
print("\n==============================")
print("Evaluation Results")
print("==============================\n")

print(
    f"Test Loss : {test_loss:.4f}"
)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"Macro F1  : {f1:.4f}"
)


# Per-Class Results
class_ids = sorted(
    id_to_label.keys()
)

target_names = [
    id_to_label[class_id]
    for class_id in class_ids
]


print("\n==============================")
print("Per-Class Results")
print("==============================\n")


report = classification_report(
    all_labels,
    all_predictions,
    labels=class_ids,
    target_names=target_names,
    zero_division=0,
)

print(report)


# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=class_ids,
)


print("\nConfusion Matrix:")

print(cm)


# Confusion Matrix Plot
display_labels = [
    id_to_label[class_id]
    for class_id in class_ids
]


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels,
)


fig, ax = plt.subplots(
    figsize=(9, 9)
)


disp.plot(
    ax=ax,
    xticks_rotation=45,
)


plt.title(
    "Protein Function Classification"
)

plt.tight_layout()


confusion_path = (
    RESULTS_DIR /
    "confusion_matrix.png"
)


plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print(
    f"\nConfusion matrix saved to:"
)

print(
    confusion_path
)


# Save Metrics
metrics = {
    "test_loss": float(test_loss),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "num_test_samples": len(test_dataset),
}


metrics_path = (
    RESULTS_DIR /
    "evaluation_metrics.json"
)


with open(
    metrics_path,
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


print(
    f"\nMetrics saved to:"
)

print(
    metrics_path
)


# Done
print(
    "\nEvaluation complete!"
)
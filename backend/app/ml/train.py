from pathlib import Path

import torch
import torch.nn as nn

from datasets import load_from_disk

from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_EPOCHS,
    USE_SUBSET,
    TRAIN_SUBSET,
    VALID_SUBSET,
    MAX_LENGTH,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

from model import ProteinClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent.parent / "data"

PROCESSED_DIR = DATA_DIR / "processed"

MODEL_PATH = CURRENT_DIR.parent / "models"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# Device
if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


print(f"\nUsing device: {DEVICE}")


# Load Dataset
print("\nLoading processed dataset...")

dataset = load_from_disk(
    str(PROCESSED_DIR)
)

train_dataset = dataset["train"]

valid_dataset = dataset["dev"]


# Development mode
if USE_SUBSET:

    train_size = min(
        TRAIN_SUBSET,
        len(train_dataset),
    )

    valid_size = min(
        VALID_SUBSET,
        len(valid_dataset),
    )

    train_dataset = train_dataset.select(
        range(train_size)
    )

    valid_dataset = valid_dataset.select(
        range(valid_size)
    )

    print("\nRunning DEVELOPMENT mode")

else:

    print("\nRunning FULL training")


print(
    f"Training samples   : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation samples : "
    f"{len(valid_dataset):,}"
)


# Dataset truncation
def truncate_batch(batch):

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    return {
        "input_ids": input_ids[:, :MAX_LENGTH],
        "attention_mask": attention_mask[:, :MAX_LENGTH],
        "label": batch["label"],
    }


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=False,
)


# Calculate class weights
print("\nCalculating class weights...")

class_counts = torch.zeros(
    6,
    dtype=torch.float32,
)


for example in train_dataset:

    label = int(example["label"])

    class_counts[label] += 1


print("\nClass counts:")

class_names = [
    "Oxidoreductase",
    "Transferase",
    "Hydrolase",
    "Lyase",
    "Isomerase",
    "Ligase",
]

for i, count in enumerate(class_counts):

    print(
        f"{i}: "
        f"{class_names[i]:<15}"
        f"{int(count):,}"
    )


# Inverse-frequency class weights
total_samples = class_counts.sum()

class_weights = total_samples / (
    len(class_counts) * class_counts
)

class_weights = class_weights.to(
    DEVICE
)

print("\nClass weights:")

for i, weight in enumerate(class_weights):

    print(
        f"{class_names[i]:<15}"
        f"{weight.item():.4f}"
    )


# Model
print("\nLoading model...")

model = ProteinClassifier()

model.to(DEVICE)


# Trainable parameter count
total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTotal parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# Loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# Optimizer
optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters(),
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# Scheduler
total_training_steps = (
    len(train_loader) * NUM_EPOCHS
)

warmup_steps = int(
    0.1 * total_training_steps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)


# Best model tracking
best_val_loss = float("inf")

epochs_without_improvement = 0


# Training
for epoch in range(NUM_EPOCHS):

    print(
        f"\n========== "
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        f" ==========\n"
    )

    # Training mode
    model.train()

    # Keep ESM in evaluation mode because it is frozen.
    model.esm.eval()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc="Training",
    )

    for batch in progress:

        input_ids = batch["input_ids"].to(
            DEVICE
        )

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch["label"].to(
            DEVICE
        )

        # Truncate to MAX_LENGTH
        input_ids = input_ids[:, :MAX_LENGTH]

        attention_mask = attention_mask[
            :, :MAX_LENGTH
        ]

        # Zero gradients
        optimizer.zero_grad(
            set_to_none=True
        )

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

        # Backpropagation
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        scheduler.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=(
                f"{scheduler.get_last_lr()[0]:.2e}"
            ),
        )

    train_loss = (
        running_loss / len(train_loader)
    )


    # Validation
    model.eval()

    validation_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        validation_progress = tqdm(
            valid_loader,
            desc="Validation",
        )

        for batch in validation_progress:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            # Truncate
            input_ids = input_ids[
                :, :MAX_LENGTH
            ]

            attention_mask = attention_mask[
                :, :MAX_LENGTH
            ]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            validation_loss += (
                loss.item()
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    validation_loss /= len(
        valid_loader
    )

    accuracy = correct / total


    # Results
    print("\nResults")

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation Loss : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{accuracy * 100:.2f}%"
    )


    # Save best model
    if validation_loss < best_val_loss:

        best_val_loss = validation_loss

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation_loss": (
                validation_loss
            ),
            "accuracy": accuracy,
        }

        torch.save(
            checkpoint,
            MODEL_SAVE_PATH,
        )

        print(
            "\n✓ Saved new best model."
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


print("\nTraining complete!")

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Model saved to:\n"
    f"{MODEL_SAVE_PATH}"
)
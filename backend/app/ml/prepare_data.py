from pathlib import Path
import ast

from datasets import load_dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH


# Paths
CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Load Dataset
print("\nDownloading SwissProt-EC...")

dataset = load_dataset(
    "DanielHesslow/SwissProt-EC"
)

print(dataset)


# Create EC Class Labels
print("\nCreating labels...")


def extract_ec_class(example):
    """
    Extract the top-level EC class from labels_str.

    Example:

        "['EC:6.-.-.-',
          'EC:6.1.-.-',
          'EC:6.1.1.-',
          'EC:6.1.1.20']"

    becomes:

        5

    because EC class 6 is converted to zero-based
    class index 5.

    Classes:

        0 -> Oxidoreductase
        1 -> Transferase
        2 -> Hydrolase
        3 -> Lyase
        4 -> Isomerase
        5 -> Ligase
    """

    labels_str = example["labels_str"]

    # --------------------------------------------------------
    # labels_str is stored as a string representation of a
    # Python list, so convert it back into a list.
    #
    # Example:
    #
    # "['EC:6.-.-.-', 'EC:6.1.-.-']"
    #
    # becomes:
    #
    # ['EC:6.-.-.-', 'EC:6.1.-.-']
    # --------------------------------------------------------

    if not isinstance(labels_str, str):
        return {"label": -1}

    try:
        labels = ast.literal_eval(labels_str)

    except (ValueError, SyntaxError):
        return {"label": -1}

    if not isinstance(labels, list):
        return {"label": -1}

    # Find first valid EC class
    for label in labels:

        if not isinstance(label, str):
            continue

        label = label.strip()

        if not label.startswith("EC:"):
            continue

        # Remove "EC:"
        ec_number = label[3:]

        # Example:
        #
        # 6.-.-.-
        #
        # 6.1.-.-
        #
        # 6.1.1.20

        parts = ec_number.split(".")

        if not parts:
            continue

        first_component = parts[0]

        if first_component == "-":
            continue

        try:
            ec_class = int(first_component)

        except ValueError:
            continue

        # EC classes 1-6
        if ec_class < 1 or ec_class > 6:
            continue

        # Convert:
        #
        # EC 1 -> class 0
        # EC 2 -> class 1
        # ...
        # EC 6 -> class 5

        return {
            "label": ec_class - 1
        }

    # No valid EC class
    return {
        "label": -1
    }


# Apply label extraction
dataset = dataset.map(
    extract_ec_class
)


# Verify Labels
print("\nChecking generated labels...")

for split in dataset:

    labels = dataset[split]["label"]

    invalid = sum(
        1
        for label in labels
        if label == -1
    )

    print(
        f"{split}: "
        f"{len(labels):,} examples | "
        f"{invalid:,} invalid"
    )


# Remove Invalid Examples
print("\nRemoving invalid examples...")

for split in dataset:

    before = len(dataset[split])

    dataset[split] = dataset[split].filter(
        lambda example: example["label"] != -1
    )

    after = len(dataset[split])

    print(
        f"{split}: "
        f"{before:,} -> {after:,}"
    )


# Class Distribution
print("\nClass distribution:")

class_names = [
    "Oxidoreductase",
    "Transferase",
    "Hydrolase",
    "Lyase",
    "Isomerase",
    "Ligase",
]

for split in dataset:

    print(f"\n{split}:")

    labels = dataset[split]["label"]

    for class_id, class_name in enumerate(class_names):

        count = sum(
            1
            for label in labels
            if label == class_id
        )

        print(
            f"{class_id}: "
            f"{class_name:<15} "
            f"{count:,}"
        )


# Load Tokenizer
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Tokenize Sequences
print("\nTokenizing sequences...")


def tokenize(example):

    return tokenizer(
        example["seq"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


dataset = dataset.map(
    tokenize
)


# Remove Original Dataset Columns
print("\nRemoving unused columns...")

columns_to_remove = [
    column
    for column in [
        "seq",
        "labels",
        "labels_str",
        "id",
    ]
    if column in dataset["train"].column_names
]

dataset = dataset.remove_columns(
    columns_to_remove
)


# Set PyTorch Format
print("\nSetting PyTorch format...")

dataset.set_format(
    type="torch",
    columns=[
        "label",
        "input_ids",
        "attention_mask",
    ],
)


# Save Dataset
print("\nSaving processed dataset...")

dataset.save_to_disk(
    str(PROCESSED_DIR)
)


# Final Summary
print("\n================================")
print("Data preparation complete!")
print("================================\n")

print(dataset)

print("\nFeatures:")

print(
    dataset["train"].features
)

print("\nExample:")

print(
    dataset["train"][0]
)
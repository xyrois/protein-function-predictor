import sys
from pathlib import Path

# Ensure backend/app/ml is directly in sys.path so config and model always resolve
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    MODEL_SAVE_PATH,
    LABEL_MAPPING_PATH,
)
from model import ProteinClassifier

# Device selection: CUDA -> MPS -> CPU
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"\nUsing device: {DEVICE}")

# Label Mapping
with open(LABEL_MAPPING_PATH, "r") as f:
    mapping = json.load(f)

# Handles both {"id_to_label": {"0": ...}} and direct {"0": ...}
raw_labels = mapping.get("id_to_label", mapping)
id_to_label = {int(k): v for k, v in raw_labels.items()}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Model
model = ProteinClassifier()
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# Prediction
def predict(sequence: str):
    sequence = sequence.strip().upper()

    # Basic Validation
    if not sequence:
        raise ValueError("Protein sequence cannot be empty.")

    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(sequence) - valid_amino_acids

    if invalid:
        raise ValueError(
            "Invalid amino acid characters: " + ", ".join(sorted(invalid))
        )

    # Tokenize using MAX_LENGTH from config (512)
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # Model Prediction
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probabilities = F.softmax(logits, dim=1)

    # Top Prediction
    confidence, prediction = torch.max(probabilities, dim=1)
    prediction_id = prediction.item()
    confidence_value = confidence.item()
    predicted_ec = id_to_label[prediction_id]

    # Top 3
    top_k = min(3, probabilities.shape[1])
    top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    # Terminal Output
    print("\n==============================")
    print("Prediction")
    print("==============================\n")
    print(f"Predicted Class : EC:{predicted_ec}.-.-.-")
    print(f"Confidence      : {confidence_value * 100:.2f}%")
    print("\nTop Predictions\n")

    for probability, index in zip(top_probabilities[0], top_indices[0]):
        class_id = index.item()
        ec_class = id_to_label[class_id]
        print(f"EC:{ec_class}.-.-.-{' ':10}{probability.item() * 100:.2f}%")

    return {
        "prediction": predicted_ec,
        "confidence": confidence_value,
        "top_predictions": [
            {
                "ec_class": id_to_label[index.item()],
                "probability": probability.item(),
            }
            for probability, index in zip(top_probabilities[0], top_indices[0])
        ],
    }


# Main
if __name__ == "__main__":
    print("\nEnter a protein sequence.")
    print("Type 'quit' to exit.\n")

    while True:
        sequence = input("> ").strip()
        if sequence.lower() == "quit":
            print("\nGoodbye!")
            break

        try:
            predict(sequence)
        except ValueError as error:
            print(f"\nError: {error}\n")
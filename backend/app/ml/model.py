import torch
import torch.nn as nn
from transformers import EsmModel

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    DROPOUT,
    FREEZE_ESM,
)


class ProteinClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Pretrained ESM model
        self.esm = EsmModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.esm.config.hidden_size

        # Freeze ESM
        if FREEZE_ESM:

            for param in self.esm.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            hidden_size,
            NUM_CLASSES,
        )

    # Get protein embedding
    def get_embedding(
        self,
        input_ids,
        attention_mask,
    ):

        # If ESM is frozen, don't construct a computation
        # graph. This saves memory and computation.
        if FREEZE_ESM:

            with torch.no_grad():

                outputs = self.esm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        else:

            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # ESM's first token embedding.
        embedding = outputs.last_hidden_state[:, 0, :]

        return embedding

    # Forward
    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        embedding = self.get_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        embedding = self.dropout(
            embedding
        )

        logits = self.classifier(
            embedding
        )

        return logits


# Test
if __name__ == "__main__":

    model = ProteinClassifier()

    input_ids = torch.randint(
        0,
        20,
        (2, 512),
    )

    attention_mask = torch.ones(
        (2, 512),
        dtype=torch.long,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    print("Output shape:", outputs.shape)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,} / {total:,}"
    )
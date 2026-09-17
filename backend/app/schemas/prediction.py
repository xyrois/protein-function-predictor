from pydantic import BaseModel, Field, field_validator
import re

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

class PredictionRequest(BaseModel):
    sequence: str = Field(
        ...,
        description="Single-letter amino acid sequence",
        examples=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    )

    @field_validator("sequence")
    @classmethod
    def clean_sequence(cls, v: str) -> str:
        # Strip whitespace, newlines, and FASTA headers
        clean_seq = re.sub(r"\s+", "", v).upper()
        if clean_seq.startswith(">"):
            lines = v.strip().splitlines()
            clean_seq = "".join(l.strip() for l in lines[1:]).upper()

        if len(clean_seq) < 10:
            raise ValueError("Protein sequence must be at least 10 amino acids long.")
        if len(clean_seq) > 2000:
            raise ValueError("Protein sequence exceeds maximum supported length (2000 residues).")

        invalid_chars = set(clean_seq) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(f"Sequence contains invalid amino acid characters: {', '.join(sorted(invalid_chars))}")

        return clean_seq


class TopPrediction(BaseModel):
    ec_class: str
    probability: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: list[TopPrediction]
    sequence_length: int
    disclaimer: str = "Experimental model (~55% accuracy). For research/prototyping purposes only."
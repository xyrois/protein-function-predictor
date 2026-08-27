# Protein Function Predictor

An ML-based protein function prediction application using ESM-2 and the SwissProt-EC dataset.

## Current Status

### Machine Learning Pipeline

- Dataset: SwissProt-EC
- Model: ESM-2 (`facebook/esm2_t6_8M_UR50D`)
- Task: 6-class enzyme classification
- Classes:
  - Oxidoreductase
  - Transferase
  - Hydrolase
  - Lyase
  - Isomerase
  - Ligase

### Dataset

After preprocessing:

- Training: 200,465
- Validation: 25,747
- Test: 24,886

Invalid/unusable labels were removed during preprocessing.

### Model Evaluation

Current baseline:

| Metric | Score |
|---|---:|
| Accuracy | 55.12% |
| Precision | 0.5187 |
| Recall | 0.5680 |
| Macro F1 | 0.5251 |
| Weighted F1 | 0.56 |

### Per-Class F1

| Class | F1 |
|---|---:|
| Oxidoreductase | 0.58 |
| Transferase | 0.57 |
| Hydrolase | 0.60 |
| Lyase | 0.42 |
| Isomerase | 0.39 |
| Ligase | 0.58 |

## ML Pipeline

```text
SwissProt-EC
      ↓
prepare_data.py
      ↓
Tokenized Dataset
      ↓
ESM-2 Fine-Tuning
      ↓
best_model.pt
      ↓
Evaluation
      ↓
inference.py
      ↓
Protein Function Prediction
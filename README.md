# Protein Function Predictor

An ongoing machine learning and bioinformatics project for predicting the functional class of proteins from their amino acid sequences.

The project uses **ESM-2**, a protein language model, fine-tuned on the **SwissProt-EC** dataset to classify proteins into six major enzyme classes.

> 🚧 **Project Status:** The machine learning pipeline is currently complete. The backend API and frontend are still in development.

---

## Overview

The goal of this project is to build an end-to-end protein function prediction application.

Given a protein's amino acid sequence, the eventual application will:

1. Accept a protein sequence or FASTA file
2. Process and validate the sequence
3. Use a fine-tuned ESM-2 model to predict its enzyme class
4. Return the predicted class and confidence scores
5. Display the results through a web interface

The project is being developed incrementally, starting with the machine learning pipeline before building the application layer.

---

## Current Progress

### ✅ Machine Learning Pipeline — Complete

The initial ML pipeline has been implemented, including:

- SwissProt-EC dataset loading
- EC enzyme class extraction
- Invalid data filtering
- ESM-2 tokenization
- Dataset preparation
- ESM-2 fine-tuning
- Model checkpointing
- Test-set evaluation
- Confusion matrix generation
- Single-sequence inference

### 🚧 Backend API — Coming Next

Planned backend functionality includes:

- FastAPI prediction endpoint
- Protein sequence validation
- Connection between the API and trained ML model
- FASTA file processing
- Batch predictions

### 🚧 Frontend — Coming Soon

A React-based web interface is planned for:

- Protein sequence input
- FASTA file uploads
- Prediction results
- Confidence scores
- Top predictions
- Batch prediction results

### 🔮 Future ML Improvements

The current model serves as a baseline. Future experiments may include:

- Class-weighted loss
- Improved protein embedding pooling
- Hyperparameter tuning
- Additional evaluation metrics
- Addressing class imbalance
- Model performance improvements

---

## Machine Learning

### Model

**ESM-2**

```text
facebook/esm2_t6_8M_UR50D
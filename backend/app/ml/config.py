from pathlib import Path


# Model
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

NUM_CLASSES = 6

MAX_LENGTH = 512


# Training
BATCH_SIZE = 16

LEARNING_RATE = 1e-3

NUM_EPOCHS = 2

WEIGHT_DECAY = 1e-4

DROPOUT = 0.3

# Freeze the pretrained ESM backbone.
FREEZE_ESM = True

# Number of epochs with no validation improvement
# before stopping.
EARLY_STOPPING_PATIENCE = 1


# Development Mode
USE_SUBSET = False

TRAIN_SUBSET = 5000

VALID_SUBSET = 1000


# Paths
CURRENT_DIR = Path(__file__).resolve().parent

MODEL_SAVE_PATH = (
    CURRENT_DIR.parent / "models" / "best_model.pt"
)

LABEL_MAPPING_PATH = (
    CURRENT_DIR.parent / "models" / "label_mapping.json"
)
import os
from pathlib import Path

# Label constants
CLASS_DIRS = ["Cat", "Dog"]

# Paths - workspace root is 3 levels up from this file
BASE_DIR = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw/PetImages")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Image size
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['accuracy']

# Model saving
MODEL_FILENAME = 'baseline_model.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# MLflow tracking
MLFLOW_EXPERIMENT_NAME = 'Cats_vs_Dogs'

import os
from pathlib import Path
from common.base import *

# Paths - workspace root is 3 levels up from this file
BASE_DIR = str(Path(__file__).resolve().parents[3])
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw/PetImages")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

INPUT_SHAPE = (224, 224, 3)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['accuracy']

# GPU Configuration
USE_MIXED_PRECISION = False  # Enable for faster training on GPUs (Metal/CUDA)

# Model saving
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

MLFLOW_RUN_NAME_PREFIX = "pipeline"
VALIDATION_METRIC_NAME = 'test_accuracy'  # Metric to use for validation gate comparison


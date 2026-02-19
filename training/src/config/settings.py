import os

# Label constants
CLASS_DIRS = ["Cat", "Dog"]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw/PetImages")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Image size
IMG_SIZE = (224, 224)

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 5

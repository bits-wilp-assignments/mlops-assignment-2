import os

# Label constants
CLASS_DIRS = ["Cat", "Dog"]

# Image size
IMG_SIZE = (224, 224)

# MLflow tracking - read from environment variable with fallback to localhost
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'https://localhost:5050')
MLFLOW_EXPERIMENT_NAME = 'pet_adoptation_classification'

# Model Registry
REGISTERED_MODEL_NAME = 'pet-classification-model'

# Model filename
MODEL_FILENAME = 'baseline_model.h5'
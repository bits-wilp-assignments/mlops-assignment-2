import numpy as np
import argparse
import mlflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from serving.src.config.settings import IMG_SIZE, CLASS_DIRS
from common.base import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from common.logger import get_logger

logger = get_logger(__name__)

class Predictor:
    def __init__(self, model_alias="champion"):
        """
        Initialize predictor with model from MLflow registry.

        Args:
            model_alias: MLflow model alias to load (default: "champion" for production)
        """
        self.img_size = IMG_SIZE
        self.classes = CLASS_DIRS
        self.model_alias = model_alias
        self.model_name = REGISTERED_MODEL_NAME

        # Load model from MLflow registry
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{model_alias}"
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        
        self.model = mlflow.keras.load_model(model_uri)
        logger.info(f"Successfully loaded model '{REGISTERED_MODEL_NAME}' with alias '{model_alias}'")

    def predict(self, image_path):
        img = load_img(image_path, target_size=self.img_size)
        x = img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        prob = self.model.predict(x)[0][0]

        # Determine the predicted class
        predicted_class_idx = int(round(prob))
        label = self.classes[predicted_class_idx]

        # Get the probability for the predicted class
        # prob is the probability of class 1 (Dog)
        # so for class 0 (Cat), the probability is 1 - prob
        confidence = prob if predicted_class_idx == 1 else 1 - prob

        logger.info("Predicted %s with probability %.3f", label, confidence)
        return {"label": label, "probability": float(confidence)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cat or dog from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(args.image_path)
    print(f"Prediction: {result['label']} (probability: {result['probability']:.3f})")

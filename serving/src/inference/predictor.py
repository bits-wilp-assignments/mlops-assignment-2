import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from common.base import IMG_SIZE
from common.logger import get_logger

logger = get_logger(__name__)
MODEL_PATH = "models/baseline_model.h5"

class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)
        self.img_size = IMG_SIZE
        self.classes = ["cat","dog"]

    def predict(self, image_path):
        img = load_img(image_path, target_size=self.img_size)
        x = img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        prob = self.model.predict(x)[0][0]
        label = self.classes[int(round(prob))]
        logger.info("Predicted %s with probability %.3f", label, prob)
        return {"label": label, "probability": float(prob)}

if __name__ == "__main__":
    predictor = Predictor()
    print(predictor.predict("data/raw/cats_and_dogs/cats/cat.0.jpg"))

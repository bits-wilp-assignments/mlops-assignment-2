from tensorflow.keras.models import load_model # type: ignore
from training.src.data.preprocess import get_generators
from training.src.config.settings import MODEL_PATH
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from common.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    _, _, test_gen = get_generators()
    model = load_model(MODEL_PATH)
    preds = model.predict(test_gen)
    y_pred = np.round(preds)
    y_true = test_gen.classes

    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))
    logger.info("Classification Report:\n%s", classification_report(y_true, y_pred))

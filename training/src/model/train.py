from tensorflow.keras import layers, models # type: ignore
from training.src.data.preprocess import get_generators
from training.src.tracking.mlflow_logger import log_model_metrics
from training.src.config.settings import (
    EPOCHS, INPUT_SHAPE, OPTIMIZER, LOSS_FUNCTION, METRICS,
    MODEL_DIR, MODEL_PATH
)
from common.logger import get_logger
import os

logger = get_logger(__name__)

def build_cnn(input_shape=INPUT_SHAPE):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    return model

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_generators()
    model = build_cnn()
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Saved model at %s", MODEL_PATH)
    log_model_metrics(history)

from tensorflow.keras import layers, models # type: ignore
from training.src.data.preprocess import get_generators
from training.src.tracking.mlflow_logger import log_model_metrics
from config.settings import EPOCHS
from common.logger import get_logger
import os

logger = get_logger(__name__)

def build_cnn(input_shape=(224,224,3)):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_generators()
    model = build_cnn()
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models","baseline_model.h5")
    model.save(model_path)
    logger.info("Saved model at %s", model_path)
    log_model_metrics(history)

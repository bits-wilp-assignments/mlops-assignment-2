from tensorflow.keras import layers, models, Input # type: ignore
import tensorflow as tf
import argparse
from training.src.data.preprocess import get_generators
from training.src.tracking.mlflow_logger import log_training_run, start_pipeline_run
import mlflow
from training.src.config.settings import (
    EPOCHS, INPUT_SHAPE, OPTIMIZER, LOSS_FUNCTION, METRICS,
    MODEL_DIR, MODEL_PATH, USE_MIXED_PRECISION
)
from common.logger import get_logger
import os
import sys

logger = get_logger(__name__)

def setup_device(use_mixed_precision=False):
    """Configure TensorFlow to use Metal (Mac) or CUDA (NVIDIA GPU) if available.
    Args:
        use_mixed_precision: Whether to enable mixed precision training for faster GPU performance
    """
    logger.info("Setting up device for training...")

    # Enable mixed precision if requested and GPU is available
    if use_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision training enabled (float16)")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")

    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Detect device type
            if sys.platform == 'darwin':  # macOS
                logger.info("Detected macOS - using Metal GPU acceleration")
                logger.info(f"Available GPU devices: {len(gpus)}")
            else:  # Linux/Windows with NVIDIA GPU
                logger.info("Detected NVIDIA GPU - using CUDA acceleration")
                logger.info(f"Available GPU devices: {len(gpus)}")

            for i, gpu in enumerate(gpus):
                logger.info(f"GPU {i}: {gpu.name}")

        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.warning("No GPU detected - training will use CPU")
        logger.info("For Mac: Install tensorflow-metal for GPU acceleration")
        logger.info("For NVIDIA: Install CUDA toolkit and cuDNN")

    return gpus

def build_cnn(input_shape=INPUT_SHAPE):
    """Build a simple CNN model for binary image classification.
    Args:
        input_shape: Shape of the input images
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    return model

def train_model(epochs=None, log_mlflow=True, use_mixed_precision=None):
    """Train the CNN model and optionally log training details to MLflow.
    Args:
        epochs: Number of training epochs (default: EPOCHS)
        log_mlflow: Whether to log training details to MLflow (default: True)
        use_mixed_precision: Whether to enable mixed precision training for faster GPU performance
    Returns:
        model: Trained Keras model
        history: Training history object
    """
    logger.info("Starting model training...")

    # Setup device (Metal for Mac, CUDA for NVIDIA)
    mp_enabled = use_mixed_precision if use_mixed_precision is not None else USE_MIXED_PRECISION
    setup_device(use_mixed_precision=mp_enabled)

    train_gen, val_gen, test_gen = get_generators()
    model = build_cnn()

    logger.info("Model architecture:\n%s", model.summary())

    training_epochs = epochs if epochs is not None else EPOCHS
    logger.info("Training for %d epochs", training_epochs)

    history = model.fit(train_gen, validation_data=val_gen, epochs=training_epochs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Saved model at %s", MODEL_PATH)

    if log_mlflow:
        log_training_run(history, model, training_epochs, use_mixed_precision=mp_enabled)

    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for image classification')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training for faster GPU training')
    args = parser.parse_args()

    # Start pipeline run if MLflow logging is enabled
    if not args.no_mlflow:
        pipeline_run = start_pipeline_run()
        logger.info("="*70)
        logger.info(f"PIPELINE RUN ID: {pipeline_run.info.run_id}")
        logger.info("="*70)
        logger.info("To attach evaluation to this run, use:")
        logger.info(f"  python -m training.src.model.evaluate --pipeline-run-id {pipeline_run.info.run_id}")
        logger.info("="*70)
        try:
            # Train model with nested run
            train_model(
                epochs=args.epochs,
                log_mlflow=True,
                use_mixed_precision=args.mixed_precision
            )
        finally:
            # End pipeline run
            mlflow.end_run()
            logger.info("Pipeline run completed")
    else:
        # Train without MLflow
        train_model(
            epochs=args.epochs,
            log_mlflow=False,
            use_mixed_precision=args.mixed_precision
        )

if __name__ == "__main__":
    main()

from tensorflow.keras import layers, models, Input # type: ignore
import tensorflow as tf
import argparse
import mlflow
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from training.src.data.preprocess import get_generators
from training.src.model.evaluate import evaluate_model as evaluate
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
    """Train and evaluate the CNN model, logging all metrics to a single MLflow run.
    Args:
        epochs: Number of training epochs (default: EPOCHS)
        log_mlflow: Whether to log to MLflow (default: True)
        use_mixed_precision: Whether to enable mixed precision training
    Returns:
        model: Trained Keras model
        history: Training history object
    """

    logger.info("Starting model training...")

    mp_enabled = use_mixed_precision if use_mixed_precision is not None else USE_MIXED_PRECISION
    setup_device(use_mixed_precision=mp_enabled)

    train_gen, val_gen, test_gen = get_generators()
    model = build_cnn()

    training_epochs = epochs if epochs is not None else EPOCHS
    logger.info("Training for %d epochs", training_epochs)

    # Start MLflow run
    if log_mlflow:
        from training.src.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, BATCH_SIZE
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Create descriptive run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"pipeline_{timestamp}"
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")

        # Log parameters
        mlflow.log_params({
            "epochs": training_epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "loss_function": LOSS_FUNCTION,
            "mixed_precision": mp_enabled
        })

    # Train
    history = model.fit(train_gen, validation_data=val_gen, epochs=training_epochs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Saved model at %s", MODEL_PATH)

    # Evaluate on test set using evaluate.py
    test_metrics, y_true, y_pred, y_pred_proba = evaluate(model, test_gen)

    # Log all metrics to MLflow
    if log_mlflow:
        # Training metrics
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        # Final training metrics
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])

        # Test metrics from evaluate.py
        for metric_name, metric_value in test_metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        # Generate and log confusion matrix plot
        try:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True, ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            ax.set_xticklabels(['Cat', 'Dog'])
            ax.set_yticklabels(['Cat', 'Dog'])
            plt.tight_layout()
            cm_path = 'confusion_matrix.png'
            plt.savefig(cm_path, dpi=80, bbox_inches='tight')
            plt.close(fig)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            logger.info("Logged confusion matrix plot to MLflow")
        except Exception as e:
            logger.warning(f"Could not log confusion matrix: {e}")

        # Generate and log ROC curve plot
        if test_metrics.get('test_auc_roc') is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_metrics["test_auc_roc"]:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                roc_path = 'roc_curve.png'
                plt.savefig(roc_path, dpi=80, bbox_inches='tight')
                plt.close(fig)
                mlflow.log_artifact(roc_path)
                os.remove(roc_path)
                logger.info("Logged ROC curve plot to MLflow")
            except Exception as e:
                logger.warning(f"Could not log ROC curve: {e}")

        # Log model summary
        try:
            import io
            stringio = io.StringIO()
            model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            model_summary = stringio.getvalue()
            summary_path = 'model_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(model_summary)
            mlflow.log_artifact(summary_path)
            os.remove(summary_path)
            logger.info("Logged model summary to MLflow")
        except Exception as e:
            logger.warning(f"Could not log model summary: {e}")

        # Log model
        mlflow.keras.log_model(model, artifact_path="model")
        mlflow.log_artifact(MODEL_PATH)
        logger.info("Logged model to MLflow")

        mlflow.end_run()
        logger.info("MLflow run completed")

        # Save run_id for DVC pipeline AFTER all artifacts are logged
        run_id_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.mlflow_run_id')
        with open(run_id_file, 'w') as f:
            f.write(run_id)
        logger.info(f"Saved MLflow run_id: {run_id}")

    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate CNN model')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        log_mlflow=not args.no_mlflow,
        use_mixed_precision=args.mixed_precision
    )

if __name__ == "__main__":
    main()

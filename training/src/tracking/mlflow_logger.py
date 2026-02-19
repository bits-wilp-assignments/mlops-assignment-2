import mlflow
from common.logger import get_logger

logger = get_logger(__name__)

def log_model_metrics(history):
    mlflow.set_experiment("Cats_vs_Dogs")
    with mlflow.start_run():
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_artifact("models/baseline_model.h5")
        logger.info("Logged metrics and model to MLflow")

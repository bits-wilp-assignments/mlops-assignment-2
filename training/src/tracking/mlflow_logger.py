import mlflow
from training.src.config.settings import MLFLOW_EXPERIMENT_NAME, MODEL_PATH
from common.logger import get_logger

logger = get_logger(__name__)

def log_model_metrics(history):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_artifact(MODEL_PATH)
        logger.info("Logged metrics and model to MLflow")

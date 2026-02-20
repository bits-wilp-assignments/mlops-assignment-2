from tensorflow.keras.models import load_model # type: ignore
import argparse
import mlflow
from training.src.data.preprocess import get_generators
from training.src.config.settings import MODEL_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from training.src.tracking.mlflow_logger import log_evaluation_metrics, start_pipeline_run
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)
import numpy as np
from common.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model_path=MODEL_PATH, log_mlflow=True):
    """Evaluate the trained model on test data and log metrics to MLflow.
    Args:
        model_path: Path to the trained model file
        log_mlflow: Whether to log metrics to MLflow (default: True)
    Returns:
        y_true: True labels
        y_pred_binary: Predicted binary labels
        y_pred_proba: Predicted probabilities
    """
    logger.info("Starting model evaluation...")
    _, _, test_gen = get_generators()
    model = load_model(model_path)

    logger.info("Generating predictions...")
    preds = model.predict(test_gen)
    y_pred_binary = np.round(preds).flatten()
    y_pred_proba = preds.flatten()
    y_true = test_gen.classes

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_binary)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)

    # Calculate AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        auc_roc = None
        fpr, tpr, thresholds = None, None, None

    # Generate classification report
    class_report = classification_report(y_true, y_pred_binary)

    # Log metrics to console
    logger.info("="*60)
    logger.info("EVALUATION METRICS")
    logger.info("="*60)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    if auc_roc is not None:
        logger.info(f"AUC-ROC:   {auc_roc:.4f}")
    logger.info("="*60)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")
    logger.info("="*60)
    logger.info("Classification Report:")
    logger.info(f"\n{class_report}")
    logger.info("="*60)

    # Log to MLflow
    if log_mlflow:
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        log_evaluation_metrics(
            y_true, y_pred_binary, y_pred_proba,
            conf_matrix, class_report, metrics_dict,
            fpr, tpr, model_path
        )

    return y_true, y_pred_binary, y_pred_proba

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test data')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help=f'Path to the trained model (default: {MODEL_PATH})')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging')
    parser.add_argument('--pipeline-run-id', type=str, default=None,
                        help='MLflow Run ID of existing pipeline run to attach evaluation to')
    args = parser.parse_args()

    # Start/resume pipeline run if MLflow logging is enabled
    if not args.no_mlflow:
        if args.pipeline_run_id:
            # Resume existing pipeline run
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            logger.info(f"Resuming existing pipeline run: {args.pipeline_run_id}")
            mlflow.start_run(run_id=args.pipeline_run_id)
        else:
            # Create new pipeline run
            pipeline_run = start_pipeline_run()

        try:
            # Evaluate model with nested run
            evaluate_model(model_path=args.model_path, log_mlflow=True)
        finally:
            # End pipeline run
            mlflow.end_run()
            logger.info("Pipeline run completed")
    else:
        # Evaluate without MLflow
        evaluate_model(model_path=args.model_path, log_mlflow=False)

if __name__ == "__main__":
    main()

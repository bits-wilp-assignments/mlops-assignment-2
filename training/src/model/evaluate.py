import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from common.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, test_generator):
    """Evaluate model on test data and return metrics.

    Args:
        model: Trained Keras model
        test_generator: Test data generator

    Returns:
        tuple: (metrics_dict, y_true, y_pred, y_pred_proba) for plotting
    """
    logger.info("Evaluating model on test set...")

    predictions = model.predict(test_generator)
    y_pred_proba = predictions.flatten()
    y_pred = np.round(y_pred_proba)
    y_true = test_generator.classes

    metrics = {
        'test_accuracy': accuracy_score(y_true, y_pred),
        'test_precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'test_recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'test_f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

    try:
        metrics['test_auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        metrics['test_auc_roc'] = None

    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"Test Recall: {metrics['test_recall']:.4f}")
    logger.info(f"Test F1: {metrics['test_f1_score']:.4f}")
    if metrics['test_auc_roc']:
        logger.info(f"Test AUC-ROC: {metrics['test_auc_roc']:.4f}")

    return metrics, y_true, y_pred, y_pred_proba

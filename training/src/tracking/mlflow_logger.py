import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from training.src.config.settings import (
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_PATH,
    BATCH_SIZE, OPTIMIZER, LOSS_FUNCTION, IMG_SIZE, INPUT_SHAPE,
    MLFLOW_RUN_NAME_PIPELINE, MLFLOW_RUN_NAME_TRAINING, MLFLOW_RUN_NAME_EVALUATION
)
from common.logger import get_logger

logger = get_logger(__name__)

def plot_confusion_matrix(confusion_mat, filename="confusion_matrix.png"):
    """Create and save confusion matrix plot with smaller file size."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Cat', 'Dog'])
    ax.set_yticklabels(['Cat', 'Dog'])
    
    # Save with lower DPI for smaller file size
    plt.tight_layout()
    plt.savefig(filename, dpi=80, bbox_inches='tight', format='png')
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {filename}")

def plot_roc_curve(fpr, tpr, auc_score, filename="roc_curve.png"):
    """Create and save ROC curve plot with smaller file size."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Save with lower DPI for smaller file size
    plt.tight_layout()
    plt.savefig(filename, dpi=80, bbox_inches='tight', format='png')
    plt.close(fig)
    logger.info(f"ROC curve plot saved to {filename}")

def generate_run_name(prefix="run", **kwargs):
    """Generate a unique run name with timestamp and optional parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = "_".join([f"{k}{v}" for k, v in kwargs.items()]) if kwargs else ""
    run_name = f"{prefix}_{param_str}_{timestamp}" if param_str else f"{prefix}_{timestamp}"
    return run_name

def start_pipeline_run():
    """Start a parent MLflow run for the entire pipeline execution."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    pipeline_name = generate_run_name(prefix=MLFLOW_RUN_NAME_PIPELINE)

    run = mlflow.start_run(run_name=pipeline_name)
    logger.info(f"Started pipeline run: {pipeline_name} (ID: {run.info.run_id})")
    return run

def log_training_run(history, model, epochs, use_mixed_precision=False, run_name=None, parent_run_id=None):
    """Log training metrics, model parameters, and artifacts to MLflow as a nested run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Use simple run name if not provided
    if run_name is None:
        run_name = MLFLOW_RUN_NAME_TRAINING

    # Create nested run if parent_run_id is provided
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Log model parameters
        params = {
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "loss_function": LOSS_FUNCTION,
            "image_size": str(IMG_SIZE),
            "input_shape": str(INPUT_SHAPE),
            "mixed_precision": use_mixed_precision,
            "total_params": model.count_params()
        }
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")
        logger.info(f"MLflow Run Name: {run_name}")

        # Log training metrics for all epochs
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        # Log final metrics
        mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

        # Log model artifact
        mlflow.log_artifact(MODEL_PATH)

        # Log model architecture as text
        try:
            import io
            stringio = io.StringIO()
            model.summary(print_fn=lambda x: stringio.write(x + '\n'))
            model_summary = stringio.getvalue()
            with open("model_summary.txt", "w") as f:
                f.write(model_summary)
            mlflow.log_artifact("model_summary.txt")
            import os
            os.remove("model_summary.txt")
        except Exception as e:
            logger.warning(f"Could not log model summary: {e}")

        logger.info(f"Logged training metrics and model to MLflow.")
        logger.info(f"Run Name: {run_name} | Run ID: {run.info.run_id}")
        return run.info.run_id

def log_evaluation_metrics(y_true, y_pred, y_pred_proba, confusion_mat, classification_rep, metrics_dict, fpr=None, tpr=None, model_path=MODEL_PATH, run_name=None, parent_run_id=None):
    """Log evaluation metrics to MLflow as a nested run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Use simple run name if not provided
    if run_name is None:
        run_name = MLFLOW_RUN_NAME_EVALUATION

    # Create nested run if parent_run_id is provided
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Log all metrics
        mlflow.log_metric("test_accuracy", metrics_dict['accuracy'])
        mlflow.log_metric("test_precision", metrics_dict['precision'])
        mlflow.log_metric("test_recall", metrics_dict['recall'])
        mlflow.log_metric("test_f1_score", metrics_dict['f1_score'])

        if metrics_dict.get('auc_roc') is not None:
            mlflow.log_metric("test_auc_roc", metrics_dict['auc_roc'])

        logger.info(f"Logged evaluation metrics: {metrics_dict}")
        logger.info(f"MLflow Run Name: {run_name}")

        # Log confusion matrix plot
        try:
            cm_filename = "confusion_matrix.png"
            plot_confusion_matrix(confusion_mat, cm_filename)
            mlflow.log_artifact(cm_filename)
            import os
            os.remove(cm_filename)
            logger.info("Logged confusion matrix plot to MLflow")
        except Exception as e:
            logger.warning(f"Could not log confusion matrix plot: {e}")

        # Log ROC curve plot if data is available
        if fpr is not None and tpr is not None and metrics_dict.get('auc_roc') is not None:
            try:
                roc_filename = "roc_curve.png"
                plot_roc_curve(fpr, tpr, metrics_dict['auc_roc'], roc_filename)
                mlflow.log_artifact(roc_filename)
                import os
                os.remove(roc_filename)
                logger.info("Logged ROC curve plot to MLflow")
            except Exception as e:
                logger.warning(f"Could not log ROC curve plot: {e}")

        # Log confusion matrix and classification report as artifact
        try:
            results_text = f"""EVALUATION RESULTS
{'='*60}

Metrics:
{'-'*60}
Accuracy:  {metrics_dict['accuracy']:.4f}
Precision: {metrics_dict['precision']:.4f}
Recall:    {metrics_dict['recall']:.4f}
F1-Score:  {metrics_dict['f1_score']:.4f}"""

            if metrics_dict.get('auc_roc') is not None:
                results_text += f"\nAUC-ROC:   {metrics_dict['auc_roc']:.4f}"

            results_text += f"""\n
Confusion Matrix:
{'-'*60}
{confusion_mat}

Classification Report:
{'-'*60}
{classification_rep}
"""

            with open("evaluation_results.txt", "w") as f:
                f.write(results_text)
            mlflow.log_artifact("evaluation_results.txt")

            # Save ROC curve data if available
            if fpr is not None and tpr is not None:
                roc_data = f"""ROC Curve Data:
{'-'*60}
FPR,TPR
"""
                for f_val, t_val in zip(fpr, tpr):
                    roc_data += f"{f_val:.6f},{t_val:.6f}\n"

                with open("roc_curve_data.csv", "w") as f:
                    f.write(roc_data)
                mlflow.log_artifact("roc_curve_data.csv")
                import os
                os.remove("roc_curve_data.csv")

            import os
            os.remove("evaluation_results.txt")
        except Exception as e:
            logger.warning(f"Could not log evaluation results: {e}")

        # Log model path as parameter
        mlflow.log_param("model_path", model_path)

        logger.info(f"Logged evaluation results to MLflow.")
        logger.info(f"Run Name: {run_name} | Run ID: {run.info.run_id}")
        return run.info.run_id

# Keep backward compatibility
def log_model_metrics(history):
    """Deprecated: Use log_training_run instead."""
    logger.warning("log_model_metrics is deprecated. Use log_training_run instead.")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("train_loss", history.history['loss'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_artifact(MODEL_PATH)
        logger.info("Logged metrics and model to MLflow")

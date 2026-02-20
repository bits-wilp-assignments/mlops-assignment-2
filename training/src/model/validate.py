import argparse
import mlflow
from mlflow.tracking import MlflowClient
from common.logger import get_logger
from training.src.config.settings import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    VALIDATION_METRIC_NAME
)

logger = get_logger(__name__)


def validation_gate(pipeline_run_id, metric_name=None, model_name=None):
    """
    Compare new model with production model and promote if better.
    Uses pipeline run ID to find both model artifact and evaluation metrics.

    Args:
        pipeline_run_id: MLflow parent pipeline run ID
        metric_name: Metric to compare (default: test_accuracy)
        model_name: Registry name (default: pet-classification-model)
    Returns:
        bool: True if model was promoted, False otherwise
    """
    # Setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    metric_name = metric_name or VALIDATION_METRIC_NAME
    model_name = model_name or REGISTERED_MODEL_NAME

    # Find child runs under this pipeline
    pipeline_run = client.get_run(pipeline_run_id)
    child_runs = client.search_runs(
        experiment_ids=[pipeline_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{pipeline_run_id}'"
    )

    # Find training run (has model artifact) and evaluation run (has test metrics)
    training_run_id = None
    evaluation_run_id = None

    for run in child_runs:
        run_name = run.data.tags.get('mlflow.runName', '').lower()
        if 'training' in run_name:
            training_run_id = run.info.run_id
            logger.info(f"Found training run: {training_run_id}")
        elif 'evaluation' in run_name:
            evaluation_run_id = run.info.run_id
            logger.info(f"Found evaluation run: {evaluation_run_id}")

    if not training_run_id or not evaluation_run_id:
        logger.error("Could not find training and evaluation runs under pipeline")
        return False

    # Get current production model metric
    try:
        production_model = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(production_model.run_id)
        champion_metric = champion_run.data.metrics.get(metric_name, 0)
        logger.info(f"Production model {metric_name}: {champion_metric:.4f}")
    except:
        # No production model exists
        champion_metric = 0
        logger.info("No production model found - first model will be promoted")

    # Get new model metric from EVALUATION run
    challenger_run = client.get_run(evaluation_run_id)
    challenger_metric = challenger_run.data.metrics.get(metric_name, 0)
    logger.info(f"New model {metric_name}: {challenger_metric:.4f}")

    # Compare and promote if better
    if challenger_metric > champion_metric:
        improvement = challenger_metric - champion_metric
        logger.info(f"New model is better by {improvement:+.4f} - promoting to production")

        # Register and promote using MLflow model format from training run
        model_uri = f"runs:/{training_run_id}/model"
        logger.info(f"Registering model from: {model_uri}")

        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.set_registered_model_alias(model_name, "champion", result.version)

        logger.info(f"Model promoted: version {result.version}")
        return True
    else:
        logger.info("New model not better - keeping current production model")
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model validation gate')
    parser.add_argument('--pipeline-run-id', type=str, required=True,
                        help='Pipeline run ID containing training and evaluation runs')
    parser.add_argument('--metric', type=str, help='Metric name for comparison')
    parser.add_argument('--model-name', type=str, help='Model registry name')

    args = parser.parse_args()

    validation_gate(
        pipeline_run_id=args.pipeline_run_id,
        metric_name=args.metric,
        model_name=args.model_name
    )

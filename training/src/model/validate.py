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


def validation_gate(run_id, metric_name=None, model_name=None):
    """
    Compare new model with production model and promote if better.
    
    Args:
        run_id: MLflow run ID containing both model and metrics
        metric_name: Metric to compare (default: test_accuracy)
        model_name: Registry name (default: pet-classification-model)
    Returns:
        bool: True if model was promoted, False otherwise
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    metric_name = metric_name or VALIDATION_METRIC_NAME
    model_name = model_name or REGISTERED_MODEL_NAME
    try:
        production_model = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(production_model.run_id)
        champion_metric = champion_run.data.metrics.get(metric_name, 0)
        logger.info(f"Production model {metric_name}: {champion_metric:.4f}")
    except:
        # No production model exists
        champion_metric = 0
        logger.info("No production model found - first model will be promoted")

    # Get new model metric
    challenger_run = client.get_run(run_id)
    challenger_metric = challenger_run.data.metrics.get(metric_name, 0)
    logger.info(f"New model {metric_name}: {challenger_metric:.4f}")

    # Compare and promote if better
    if challenger_metric > champion_metric:
        improvement = challenger_metric - champion_metric
        logger.info(f"New model is better by {improvement:+.4f} - promoting to production")

        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        client.set_registered_model_alias(model_name, "champion", result.version)

        logger.info(f"Model promoted: version {result.version}")
        return True
    else:
        logger.info("New model not better - keeping current production model")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model validation gate')
    parser.add_argument('--run-id', type=str, required=True, 
                        help='MLflow run ID to validate')
    parser.add_argument('--metric', type=str, help='Metric name for comparison')
    parser.add_argument('--model-name', type=str, help='Model registry name')

    args = parser.parse_args()

    validation_gate(
        run_id=args.run_id,
        metric_name=args.metric,
        model_name=args.model_name
    )

"""
Complete ML Pipeline: Training + Evaluation
Runs both training and evaluation as nested runs under a single pipeline execution.
"""
import argparse
import mlflow
from training.src.model.train import train_model
from training.src.model.evaluate import evaluate_model
from training.src.tracking.mlflow_logger import start_pipeline_run
from training.src.config.settings import EPOCHS, USE_MIXED_PRECISION
from common.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(epochs=None, use_mixed_precision=None):
    """
    Execute the complete ML pipeline with training and evaluation as nested runs.

    Args:
        epochs: Number of training epochs
        use_mixed_precision: Enable mixed precision training
    """
    logger.info("=" * 80)
    logger.info("Starting Complete ML Pipeline")
    logger.info("=" * 80)

    # Start parent pipeline run
    pipeline_run = start_pipeline_run()
    
    logger.info("=" * 80)
    logger.info(f"PIPELINE RUN ID: {pipeline_run.info.run_id}")
    logger.info("=" * 80)
    logger.info("To attach evaluation to this run later, use:")
    logger.info(f"  python -m training.src.model.evaluate --pipeline-run-id {pipeline_run.info.run_id}")
    logger.info("=" * 80)

    try:
        # Log pipeline parameters to parent run
        mlflow.log_param("pipeline_stage", "complete")
        mlflow.log_param("includes_training", True)
        mlflow.log_param("includes_evaluation", True)

        # Stage 1: Training (nested run)
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: Model Training")
        logger.info("=" * 80)

        mp_enabled = use_mixed_precision if use_mixed_precision is not None else USE_MIXED_PRECISION
        model, history = train_model(
            epochs=epochs,
            log_mlflow=True,
            use_mixed_precision=mp_enabled
        )

        # Stage 2: Evaluation (nested run)
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Model Evaluation")
        logger.info("=" * 80)

        y_true, y_pred = evaluate_model(log_mlflow=True)

        # Log overall pipeline success
        mlflow.log_param("pipeline_status", "success")
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Completed Successfully!")
        logger.info(f"Pipeline Run: {pipeline_name}")
        logger.info(f"Run ID: {pipeline_run.info.run_id}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        mlflow.log_param("pipeline_status", "failed")
        mlflow.log_param("error_message", str(e))
        raise

    finally:
        # End parent pipeline run
        mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(
        description='Run complete ML pipeline (training + evaluation) with MLflow tracking'
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training for faster GPU training')

    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        use_mixed_precision=args.mixed_precision
    )


if __name__ == "__main__":
    main()

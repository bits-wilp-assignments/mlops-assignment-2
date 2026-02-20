"""
Complete ML Pipeline: Training + Evaluation + Validation Gate
Runs training, evaluation, and validation gate as nested runs under a single pipeline execution.
"""
import argparse
import mlflow
import subprocess
from pathlib import Path
from training.src.model.train import train_model
from training.src.model.evaluate import evaluate_model
from training.src.model.validate import validation_gate
from training.src.tracking.mlflow_logger import start_pipeline_run
from training.src.config.settings import EPOCHS, USE_MIXED_PRECISION, PROCESSED_DIR
from common.logger import get_logger

logger = get_logger(__name__)


def ensure_data_available():
    """
    Ensure DVC-tracked data is available before running the pipeline.
    Checks if processed data exists, and runs 'dvc pull' if necessary.
    """
    processed_path = Path(PROCESSED_DIR)
    
    # Check if processed data directory exists and has content
    if not processed_path.exists():
        logger.warning(f"Processed data directory not found: {PROCESSED_DIR}")
        logger.info("Attempting to pull data from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True, cwd=Path(__file__).resolve().parents[3])
            logger.info("✓ Data pulled successfully from DVC")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull data from DVC: {e}")
            raise RuntimeError("Data not available and DVC pull failed. Please run 'dvc pull' manually.")
        except FileNotFoundError:
            logger.error("DVC command not found. Please install DVC: pip install dvc")
            raise RuntimeError("DVC not installed. Please run: pip install dvc")
    else:
        # Check if directory has train/val/test subdirectories
        expected_dirs = ["train", "val", "test"]
        has_data = all((processed_path / d).exists() for d in expected_dirs)
        
        if not has_data:
            logger.warning("Processed data structure incomplete")
            logger.info("Attempting to pull data from DVC...")
            try:
                subprocess.run(["dvc", "pull"], check=True, cwd=Path(__file__).resolve().parents[3])
                logger.info("✓ Data pulled successfully from DVC")
            except Exception as e:
                logger.error(f"Failed to pull data: {e}")
                raise RuntimeError("Data structure incomplete and DVC pull failed.")
        else:
            logger.info("✓ DVC-tracked data is available")


def run_pipeline(epochs=None, use_mixed_precision=None, enable_validation_gate=True):
    """
    Execute the complete ML pipeline with training, evaluation, and validation gate.

    Args:
        epochs: Number of training epochs
        use_mixed_precision: Enable mixed precision training
        enable_validation_gate: Whether to run validation gate (default: True)
    """
    logger.info("=" * 80)
    logger.info("Starting Complete ML Pipeline")
    logger.info("=" * 80)
    
    # Ensure DVC data is available
    ensure_data_available()

    # Start parent pipeline run
    pipeline_run = start_pipeline_run()

    logger.info("=" * 80)
    logger.info(f"PIPELINE RUN ID: {pipeline_run.info.run_id}")
    logger.info("=" * 80)
    logger.info("To attach evaluation to this run later, use:")
    logger.info(f"  python -m training.src.model.evaluate --pipeline-run-id {pipeline_run.info.run_id}")
    logger.info("=" * 80)

    training_run_id = None
    evaluation_run_id = None

    try:
        # Log pipeline parameters to parent run
        mlflow.log_param("pipeline_stage", "complete")
        mlflow.log_param("includes_training", True)
        mlflow.log_param("includes_evaluation", True)
        mlflow.log_param("includes_validation_gate", enable_validation_gate)

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

        # Get the training run ID from the active MLflow run
        # The training run ID is from the nested run created by train_model
        # We need to retrieve it from MLflow
        client = mlflow.tracking.MlflowClient()
        parent_run = client.get_run(pipeline_run.info.run_id)

        # Get child runs (nested runs)
        query = f"tags.mlflow.parentRunId = '{pipeline_run.info.run_id}'"
        child_runs = client.search_runs(
            experiment_ids=[parent_run.info.experiment_id],
            filter_string=query,
            order_by=["start_time DESC"]
        )

        # The most recent child run should be the training run
        if child_runs:
            training_run_id = child_runs[0].info.run_id
            logger.info(f"Training Run ID: {training_run_id}")

        # Stage 2: Evaluation (nested run)
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: Model Evaluation")
        logger.info("=" * 80)

        y_true, y_pred, y_pred_proba = evaluate_model(log_mlflow=True)

        # Get the evaluation run ID
        child_runs = client.search_runs(
            experiment_ids=[parent_run.info.experiment_id],
            filter_string=query,
            order_by=["start_time DESC"]
        )

        if len(child_runs) > 1:
            evaluation_run_id = child_runs[0].info.run_id
            logger.info(f"Evaluation Run ID: {evaluation_run_id}")

        # Stage 3: Validation Gate (if enabled)
        if enable_validation_gate and training_run_id:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: Validation Gate")
            logger.info("=" * 80)

            # Run validation gate - compares and promotes if better
            promoted = validation_gate(pipeline_run_id=pipeline_run.info.run_id)

            # Log validation result to pipeline run
            mlflow.log_param("model_promoted", promoted)

        # Log overall pipeline success
        mlflow.log_param("pipeline_status", "success")
        logger.info("\n" + "=" * 80)
        logger.info("✓ Pipeline Completed Successfully!")
        logger.info(f"Pipeline Run ID: {pipeline_run.info.run_id}")
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
        description='Run complete ML pipeline (training + evaluation + validation) with MLflow tracking'
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training for faster GPU training')
    parser.add_argument('--no-validation-gate', action='store_true',
                        help='Disable validation gate (skip model comparison and promotion)')

    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        use_mixed_precision=args.mixed_precision,
        enable_validation_gate=not args.no_validation_gate
    )


if __name__ == "__main__":
    main()

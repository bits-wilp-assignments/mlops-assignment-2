import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from unittest.mock import patch, MagicMock
from training.src.model.evaluate import evaluate_model
from training.src.config.settings import MODEL_PATH


def test_classification_metrics():
    """Test classification report generation."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    # Test classification report
    report = classification_report(y_true, y_pred)
    assert "precision" in report, "Report missing precision"
    assert "recall" in report, "Report missing recall"
    assert "f1-score" in report, "Report missing f1-score"

    # Test confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"

    # Test accuracy
    acc = accuracy_score(y_true, y_pred)
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"


@patch('training.src.model.evaluate.get_generators')
@patch('training.src.model.evaluate.load_model')
def test_evaluate_model_without_mlflow(mock_load_model, mock_get_generators):
    """Test model evaluation function without MLflow logging."""
    # Mock the test generator
    mock_test_gen = MagicMock()
    mock_test_gen.classes = np.array([0, 1, 0, 1, 1, 0])
    mock_get_generators.return_value = (None, None, mock_test_gen)

    # Mock the model
    mock_model = MagicMock()
    mock_predictions = np.array([[0.2], [0.8], [0.3], [0.7], [0.9], [0.1]])
    mock_model.predict.return_value = mock_predictions
    mock_load_model.return_value = mock_model

    # Run evaluation without MLflow
    y_true, y_pred_binary, y_pred_proba = evaluate_model(model_path=MODEL_PATH, log_mlflow=False)

    # Verify results
    assert len(y_true) == 6, "Should have 6 true labels"
    assert len(y_pred_binary) == 6, "Should have 6 binary predictions"
    assert len(y_pred_proba) == 6, "Should have 6 probability predictions"
    assert np.all(np.isin(y_pred_binary, [0, 1])), "Predictions should be binary (0 or 1)"
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilities should be between 0 and 1"

    # Verify model was loaded and used
    mock_load_model.assert_called_once_with(MODEL_PATH)
    mock_model.predict.assert_called_once()


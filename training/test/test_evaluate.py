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


def test_evaluate_model_without_mlflow():
    """Test model evaluation function without MLflow logging."""
    # Mock the test generator
    mock_test_gen = MagicMock()
    mock_test_gen.classes = np.array([0, 1, 0, 1, 1, 0])

    # Mock the model
    mock_model = MagicMock()
    mock_predictions = np.array([[0.2], [0.8], [0.3], [0.7], [0.9], [0.1]])
    mock_model.predict.return_value = mock_predictions

    # Run evaluation
    metrics, y_true, y_pred, y_pred_proba = evaluate_model(mock_model, mock_test_gen)

    # Verify results
    assert len(y_true) == 6, "Should have 6 true labels"
    assert len(y_pred) == 6, "Should have 6 binary predictions"
    assert len(y_pred_proba) == 6, "Should have 6 probability predictions"
    assert np.all(np.isin(y_pred, [0, 1])), "Predictions should be binary (0 or 1)"
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilities should be between 0 and 1"
    
    # Verify metrics dictionary contains expected keys
    assert 'test_accuracy' in metrics, "Metrics should contain test_accuracy"
    assert 'test_precision' in metrics, "Metrics should contain test_precision"
    assert 'test_recall' in metrics, "Metrics should contain test_recall"
    assert 'test_f1_score' in metrics, "Metrics should contain test_f1_score"

    # Verify model was used
    mock_model.predict.assert_called_once()


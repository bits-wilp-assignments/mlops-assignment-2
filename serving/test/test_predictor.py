import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Check if we're in CI environment (GitHub Actions sets CI=true)
CI_ENV = os.getenv("CI", "").lower() == "true"

@pytest.fixture
def mock_predictor():
    """Mock predictor for CI environment"""
    mock = Mock()
    mock.model_name = "pet-classification-model"
    mock.model_alias = "champion"
    
    def mock_predict(image_path):
        if "Cat" in image_path or "cat" in image_path:
            return {"label": "Cat", "probability": 0.95}
        else:
            return {"label": "Dog", "probability": 0.92}
    
    mock.predict = mock_predict
    return mock

def test_predictor_initialization():
    """Test predictor initialization with mocked MLflow"""
    with patch('serving.src.inference.predictor.mlflow') as mock_mlflow:
        # Mock MLflow components
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.95]])
        mock_mlflow.keras.load_model.return_value = mock_model
        mock_mlflow.set_tracking_uri.return_value = None
        
        from serving.src.inference.predictor import Predictor
        predictor = Predictor(model_alias="champion")
        
        # Verify initialization
        assert predictor.model_name == "pet-classification-model"
        assert predictor.model_alias == "champion"
        assert predictor.model is not None
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.keras.load_model.assert_called_once()

@pytest.mark.skipif(CI_ENV, reason="Integration test - requires MLflow server and registered model")
def test_predictor_cat_image():
    """Test prediction on a cat image (integration test)"""
    from serving.src.inference.predictor import Predictor
    predictor = Predictor()
    
    result = predictor.predict("data/processed/test/Cat/3.jpg")
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ["Cat", "Dog"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] == "Cat"

@pytest.mark.skipif(CI_ENV, reason="Integration test - requires MLflow server and registered model")
def test_predictor_dog_image():
    """Test prediction on a dog image (integration test)"""
    from serving.src.inference.predictor import Predictor
    predictor = Predictor()
    
    result = predictor.predict("data/processed/test/Dog/9.jpg")
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ["Cat", "Dog"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] == "Dog"

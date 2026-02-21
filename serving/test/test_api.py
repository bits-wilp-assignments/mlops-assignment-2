from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Mock the Predictor before importing main
mock_predictor = Mock()
mock_predictor.model_name = "pet-classification-model"
mock_predictor.model_alias = "champion"
mock_predictor.predict.return_value = {"label": "Cat", "probability": 0.95}

with patch('serving.src.api.main.Predictor', return_value=mock_predictor):
    from serving.src.api.main import app

client = TestClient(app)

def test_health():
    """Test health endpoint returns correct status and model info from MLflow registry"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_name" in data
    assert "model_alias" in data
    assert data["model_name"] == "pet-classification-model"
    assert data["model_alias"] == "champion"

def test_predict_endpoint():
    """Test predict endpoint with mock predictor"""
    # Create a dummy file for testing
    files = {"file": ("test_cat.jpg", b"fake image content", "image/jpeg")}
    response = client.put("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert data["label"] in ["Cat", "Dog"]
    assert 0.0 <= data["probability"] <= 1.0

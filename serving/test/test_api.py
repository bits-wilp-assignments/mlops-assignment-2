import pytest
import io
from fastapi.testclient import TestClient
from serving.src.api.main import app
from PIL import Image

client = TestClient(app)

def test_health():
    """Test health endpoint returns correct status and model version"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert data["model_version"] == "baseline_model"

def test_predict_endpoint():
    """Test predict endpoint with an image file"""
    # Read a test image
    with open("data/processed/test/Cat/3.jpg", "rb") as img_file:
        files = {"file": ("test_cat.jpg", img_file, "image/jpeg")}
        response = client.put("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert data["label"] in ["Cat", "Dog"]
    assert 0.0 <= data["probability"] <= 1.0

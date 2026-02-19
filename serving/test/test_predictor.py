import pytest
from serving.src.inference.predictor import Predictor

@pytest.fixture
def predictor():
    """Fixture to create a predictor instance"""
    return Predictor()

def test_predictor_cat_image(predictor):
    """Test prediction on a cat image"""
    result = predictor.predict("data/processed/test/Cat/3.jpg")
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ["Cat", "Dog"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] == "Cat"

def test_predictor_dog_image(predictor):
    """Test prediction on a dog image"""
    result = predictor.predict("data/processed/test/Dog/9.jpg")
    assert "label" in result
    assert "probability" in result
    assert result["label"] in ["Cat", "Dog"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] == "Dog"

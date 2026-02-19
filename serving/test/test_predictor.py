from serving.src.inference.predictor import Predictor

def test_predictor():
    predictor = Predictor()
    result = predictor.predict("data/raw/cats_and_dogs/cats/cat.0.jpg")
    assert "label" in result
    assert "probability" in result

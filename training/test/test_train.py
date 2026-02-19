from training.src.models.train import build_cnn

def test_build_cnn():
    model = build_cnn()
    assert model is not None, "Model not created"
    assert model.output_shape[-1] == 1, "Output shape should be 1 for binary classification"

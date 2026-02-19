import numpy as np
from training.src.model.train import build_cnn, setup_device
from training.src.config.settings import INPUT_SHAPE, LOSS_FUNCTION


def test_build_cnn():
    """Test CNN model architecture and configuration."""
    model = build_cnn()

    # Test model creation
    assert model is not None, "Model not created"

    # Test output shape for binary classification
    assert model.output_shape[-1] == 1, "Output shape should be 1 for binary classification"

    # Test input shape
    assert model.input_shape[1:] == INPUT_SHAPE, f"Input shape should be {INPUT_SHAPE}"

    # Test model has been compiled
    assert model.optimizer is not None, "Model should be compiled with optimizer"
    assert model.loss == LOSS_FUNCTION, f"Loss function should be {LOSS_FUNCTION}"

    # Test model has trainable parameters
    assert model.count_params() > 0, "Model should have trainable parameters"

    # Test model has expected number of layers
    assert len(model.layers) > 0, "Model should have layers"


def test_model_prediction_shape():
    """Test model prediction output shape."""
    model = build_cnn()

    # Create dummy input batch
    batch_size = 4
    dummy_input = np.random.rand(batch_size, *INPUT_SHAPE).astype(np.float32)

    # Get prediction
    predictions = model.predict(dummy_input, verbose=0)

    # Test prediction shape
    assert predictions.shape == (batch_size, 1), f"Predictions shape should be ({batch_size}, 1)"

    # Test prediction range (sigmoid output should be between 0 and 1)
    assert np.all(predictions >= 0) and np.all(predictions <= 1), "Predictions should be between 0 and 1"


def test_setup_device():
    """Test device setup function."""
    # Test that setup_device runs without errors
    gpus = setup_device(use_mixed_precision=False)

    # Should return a list (empty if no GPU, populated if GPU available)
    assert isinstance(gpus, list), "setup_device should return a list"


import os
from training.src.data.preprocess import preprocess_and_split, get_generators
from training.src.config.settings import PROCESSED_DIR, IMG_SIZE, BATCH_SIZE


def test_preprocess_and_split():
    """Test data preprocessing and splitting."""
    preprocess_and_split()

    # Check processed folders exist
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(PROCESSED_DIR, split)
        assert os.path.exists(split_dir), f"{split_dir} does not exist"

        # Check class subdirectories exist
        for class_name in ["Cat", "Dog"]:
            class_dir = os.path.join(split_dir, class_name)
            assert os.path.exists(class_dir), f"{class_dir} does not exist"


def test_generators():
    """Test data generators creation and output."""
    train_gen, val_gen, test_gen = get_generators()

    # Test generator objects are not None
    assert train_gen is not None, "Train generator is None"
    assert val_gen is not None, "Validation generator is None"
    assert test_gen is not None, "Test generator is None"

    # Check first batch shape from train generator
    x, y = next(train_gen)
    assert x.shape[1:3] == IMG_SIZE, f"Image size should be {IMG_SIZE}"
    assert y.shape[0] == x.shape[0], "Batch label size mismatch"
    assert x.shape[0] <= BATCH_SIZE, f"Batch size should not exceed {BATCH_SIZE}"

    # Check pixel values are normalized (should be between 0 and 1)
    assert x.min() >= 0 and x.max() <= 1, "Images should be normalized between 0 and 1"


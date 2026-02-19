import os
from training.src.data.preprocess import preprocess_and_split, get_generators
from common.base import PROCESSED_DIR

def test_preprocess_and_split():
    preprocess_and_split()
    # Check processed folders exist
    for split in ["train","val","test"]:
        split_dir = os.path.join(PROCESSED_DIR, split)
        assert os.path.exists(split_dir), f"{split_dir} does not exist"

def test_generators():
    train_gen, val_gen, test_gen = get_generators()
    # Check first batch shape
    x, y = next(train_gen)
    assert x.shape[1:3] == (224,224), "Image size mismatch"
    assert y.shape[0] == x.shape[0], "Batch label size mismatch"

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch
from PIL import Image
from training.src.data.preprocess import preprocess_and_split, get_generators
from training.src.config.settings import IMG_SIZE, BATCH_SIZE


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_base = tempfile.mkdtemp()
    raw_dir = os.path.join(temp_base, "raw")
    processed_dir = os.path.join(temp_base, "processed")
    
    # Create raw data structure with fake images
    os.makedirs(os.path.join(raw_dir, "Cat"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "Dog"), exist_ok=True)
    
    # Create dummy images (30 per class for proper splitting)
    for class_name in ["Cat", "Dog"]:
        class_dir = os.path.join(raw_dir, class_name)
        for i in range(30):
            img = Image.new('RGB', (100, 100), color=(i * 8, i * 8, i * 8))
            img.save(os.path.join(class_dir, f"img_{i}.jpg"))
    
    yield raw_dir, processed_dir
    
    # Cleanup
    shutil.rmtree(temp_base)


def test_preprocess_and_split(temp_dirs):
    """Test data preprocessing and splitting with temporary directories."""
    raw_dir, processed_dir = temp_dirs
    
    # Patch the settings to use temp directories
    with patch('training.src.data.preprocess.RAW_DIR', raw_dir), \
         patch('training.src.data.preprocess.PROCESSED_DIR', processed_dir), \
         patch('training.src.data.preprocess.CLASS_DIRS', ["Cat", "Dog"]):
        
        preprocess_and_split()
        
        # Check processed folders exist
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(processed_dir, split)
            assert os.path.exists(split_dir), f"{split_dir} does not exist"
            
            # Check class subdirectories exist
            for class_name in ["Cat", "Dog"]:
                class_dir = os.path.join(split_dir, class_name)
                assert os.path.exists(class_dir), f"{class_dir} does not exist"
                
                # Verify images were created
                images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                assert len(images) > 0, f"No images found in {class_dir}"


def test_generators(temp_dirs):
    """Test data generators creation and output with temporary directories."""
    raw_dir, processed_dir = temp_dirs
    
    # First, create processed data
    with patch('training.src.data.preprocess.RAW_DIR', raw_dir), \
         patch('training.src.data.preprocess.PROCESSED_DIR', processed_dir), \
         patch('training.src.data.preprocess.CLASS_DIRS', ["Cat", "Dog"]):
        preprocess_and_split()
    
    # Now test generators with the processed data
    with patch('training.src.data.preprocess.PROCESSED_DIR', processed_dir):
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


def test_preprocess_handles_corrupt_images(temp_dirs):
    """Test that preprocessing handles corrupt images gracefully."""
    raw_dir, processed_dir = temp_dirs
    
    # Create a corrupted image file
    corrupt_file = os.path.join(raw_dir, "Cat", "corrupt.jpg")
    with open(corrupt_file, 'wb') as f:
        f.write(b'not_an_image')
    
    # Should not raise exception
    with patch('training.src.data.preprocess.RAW_DIR', raw_dir), \
         patch('training.src.data.preprocess.PROCESSED_DIR', processed_dir), \
         patch('training.src.data.preprocess.CLASS_DIRS', ["Cat", "Dog"]):
        
        preprocess_and_split()
        
        # Verify processing completed despite corrupt file
        train_dir = os.path.join(processed_dir, "train", "Cat")
        assert os.path.exists(train_dir), "Train directory should exist"


def test_preprocess_filters_hidden_files(temp_dirs):
    """Test that preprocessing filters out hidden files like .DS_Store."""
    raw_dir, processed_dir = temp_dirs
    
    # Create a hidden file
    hidden_file = os.path.join(raw_dir, "Cat", ".DS_Store")
    with open(hidden_file, 'w') as f:
        f.write("hidden")
    
    with patch('training.src.data.preprocess.RAW_DIR', raw_dir), \
         patch('training.src.data.preprocess.PROCESSED_DIR', processed_dir), \
         patch('training.src.data.preprocess.CLASS_DIRS', ["Cat", "Dog"]):
        
        # Should not raise exception or process hidden files
        preprocess_and_split()
        
        # Verify only valid images were processed
        for split in ["train", "val", "test"]:
            for class_name in ["Cat", "Dog"]:
                class_dir = os.path.join(processed_dir, split, class_name)
                if os.path.exists(class_dir):
                    files = os.listdir(class_dir)
                    hidden_files = [f for f in files if f.startswith('.')]
                    assert len(hidden_files) == 0, "Hidden files should not be processed"


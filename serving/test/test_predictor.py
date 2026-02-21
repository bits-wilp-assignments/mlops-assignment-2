"""Simple smoke tests for serving predictor"""

def test_config_values():
    """Test serving configuration values"""
    from serving.src.config.settings import IMG_SIZE, CLASS_DIRS
    
    assert IMG_SIZE == (224, 224)
    assert CLASS_DIRS == ["Cat", "Dog"]

def test_base_config():
    """Test common configuration"""
    from common.base import REGISTERED_MODEL_NAME
    
    assert REGISTERED_MODEL_NAME == "pet-classification-model"




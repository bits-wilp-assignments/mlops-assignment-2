"""Simple smoke tests for serving API structure"""

def test_config_exists():
    """Test serving config module exists"""
    import serving.src.config.settings as settings
    assert hasattr(settings, 'IMG_SIZE')
    assert hasattr(settings, 'CLASS_DIRS')

def test_monitoring_exists():
    """Test monitoring module exists"""
    import serving.monitoring.metrics as metrics
    assert hasattr(metrics, 'RequestMetrics')




from training.src.models.evaluate import classification_report
import numpy as np

def test_confusion_matrix():
    # dummy test
    y_true = np.array([0,1,0,1])
    y_pred = np.array([0,1,1,0])
    report = classification_report(y_true, y_pred)
    assert "precision" in report, "Report missing precision"

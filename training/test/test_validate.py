"""
Tests for model validation gate functionality
"""
import pytest
from unittest.mock import Mock, patch
from training.src.model.validate import validation_gate


class TestValidationGate:
    """Test suite for validation gate functionality"""

    @patch('training.src.model.validate.mlflow')
    @patch('training.src.model.validate.MlflowClient')
    def test_validation_gate_challenger_wins(self, mock_client_class, mock_mlflow):
        """Test validation gate when new model is better"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock production model
        mock_prod_model = Mock()
        mock_prod_model.run_id = "run_123"
        mock_client.get_model_version_by_alias.return_value = mock_prod_model

        # Mock production run with metric
        mock_prod_run = Mock()
        mock_prod_run.data.metrics = {'test_accuracy': 0.85}

        # Mock challenger run with better metric
        mock_challenger_run = Mock()
        mock_challenger_run.data.metrics = {'test_accuracy': 0.90}

        mock_client.get_run.side_effect = [mock_prod_run, mock_challenger_run]

        # Mock model registration
        mock_model_version = Mock()
        mock_model_version.version = "2"
        mock_mlflow.register_model.return_value = mock_model_version

        # Execute
        promoted = validation_gate(challenger_run_id="run_456")

        # Assert
        assert promoted is True
        mock_mlflow.register_model.assert_called_once()
        mock_client.set_registered_model_alias.assert_called_once()

    @patch('training.src.model.validate.mlflow')
    @patch('training.src.model.validate.MlflowClient')
    def test_validation_gate_challenger_loses(self, mock_client_class, mock_mlflow):
        """Test validation gate when new model is worse"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock production model
        mock_prod_model = Mock()
        mock_prod_model.run_id = "run_123"
        mock_client.get_model_version_by_alias.return_value = mock_prod_model

        # Mock production run with higher metric
        mock_prod_run = Mock()
        mock_prod_run.data.metrics = {'test_accuracy': 0.90}

        # Mock challenger run with worse metric
        mock_challenger_run = Mock()
        mock_challenger_run.data.metrics = {'test_accuracy': 0.85}

        mock_client.get_run.side_effect = [mock_prod_run, mock_challenger_run]

        # Execute
        promoted = validation_gate(challenger_run_id="run_456")

        # Assert
        assert promoted is False
        mock_mlflow.register_model.assert_not_called()

    @patch('training.src.model.validate.mlflow')
    @patch('training.src.model.validate.MlflowClient')
    def test_validation_gate_first_model(self, mock_client_class, mock_mlflow):
        """Test validation gate when no production model exists"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # No production model exists
        mock_client.get_model_version_by_alias.side_effect = Exception("Not found")

        # Mock challenger run
        mock_challenger_run = Mock()
        mock_challenger_run.data.metrics = {'test_accuracy': 0.80}
        mock_client.get_run.return_value = mock_challenger_run

        # Mock model registration
        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_mlflow.register_model.return_value = mock_model_version

        # Execute
        promoted = validation_gate(challenger_run_id="run_456")

        # Assert
        assert promoted is True
        mock_mlflow.register_model.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


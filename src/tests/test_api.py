"""Unit tests for Flask API."""
import pytest
import json
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from src.serving.server import app
import os

APP_PATH = 'src.serving.server'

MOCK_MODEL_NAME = "test_model_env"
MOCK_MODEL_ALIAS = "staging_env"

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([16, 4, 1]))
    return model

@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing"""
    pipeline = Mock()
    pipeline.transform = Mock(return_value=pd.DataFrame({
        'VendorID': [2, 2, 2],
        'passenger_count': [3, 5, 3],
        'trip_distance': [3.86,0,0],
        'PULocationID': [238,264,264],
        'DOLocationID': [116,264,264],
        'pickup_hour': [0,0,0],
        'pickup_day_of_week': [0,0,0],
        'pickup_month': [1,1,1],
        'pickup_day': [1,1,1]
        }))
    return pipeline

def test_health_endpoint_without_model(client):
    """Test health endpoint when no model is loaded."""
    with patch.dict(app.config, {"MODEL": None, "PIPELINE": None}):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is False


def test_health_endpoint_with_model(client, mock_model, mock_pipeline):
    """Test health endpoint when model is loaded."""
    with patch.dict(app.config, {"MODEL": mock_model, "PIPELINE": mock_pipeline}):
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['model_loaded'] is True


def test_predict_without_model(client):
    """Test predict endpoint when no model is loaded."""
    with patch.dict(app.config, {"MODEL": None, "PIPELINE": None}):
        response = client.post(
            '/predict',
            data=json.dumps({
                "data": [
                            {
                            "VendorID": 2,
                            "tpep_pickup_datetime": "2013-01-01 00:39:00",
                            "passenger_count": 3,
                            "trip_distance": 3.86,
                            "PULocationID": 238,
                            "DOLocationID": 116
                            }
                        ]
            }),
            content_type='application/json'
        )
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Model or pipeline not loaded. Please call /reload first.' in data['error']


def test_predict_with_model(client, mock_model, mock_pipeline):
    """Test predict endpoint with a loaded model."""
    with patch.dict(app.config, {"MODEL": mock_model, "PIPELINE": mock_pipeline}):
        response = client.post(
            '/predict',
            data=json.dumps({
                "data": [
                    {
                    "VendorID": 2,
                    "tpep_pickup_datetime": "2013-01-01 00:39:00",
                    "passenger_count": 3,
                    "trip_distance": 3.86,
                    "PULocationID": 238,
                    "DOLocationID": 116
                    },
                    {
                    "VendorID": 2,
                    "tpep_pickup_datetime": "2013-01-01 00:12:00",
                    "passenger_count": 5,
                    "trip_distance": 0.0,
                    "PULocationID": 264,
                    "DOLocationID": 264
                    },
                    {
                    "VendorID": 2,
                    "tpep_pickup_datetime": "2013-01-01 00:02:00",
                    "passenger_count": 3,
                    "trip_distance": 0.0,
                    "PULocationID": 264,
                    "DOLocationID": 264
                    }
                ]
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'predictions' in data
        assert len(data['predictions']) == 3

        assert data['predictions'] == [16, 4, 1]


def test_predict_invalid_data(client, mock_model, mock_pipeline):
    """Test predict endpoint with invalid data format."""
    with patch.dict(app.config, {"MODEL": mock_model, "PIPELINE": mock_pipeline}):
        response = client.post(
            '/predict',
            data=json.dumps({'invalid': 'data'}),
            content_type='application/json'
        )
        # This should fail with 400
        assert response.status_code == 400


def test_reload_model_success(client):
    """Test reload endpoint when model loads successfully."""
    with patch('mlflow.sklearn.load_model') as mock_model_load, \
        patch('mlflow.artifacts.download_artifacts') as mock_download_artifacts, \
        patch('joblib.load') as mock_joblib_load, \
        patch('mlflow.tracking.MlflowClient') as MockMlflowClient, \
        patch(f'{APP_PATH}.MODEL_NAME', new=MOCK_MODEL_NAME), \
        patch(f'{APP_PATH}.MODEL_ALIAS', new=MOCK_MODEL_ALIAS):

        # Mock model
        mock_model = Mock()
        mock_model_load.return_value = mock_model

        mock_alias_info = Mock()
        mock_alias_info.run_id = "test_run_mock"

        mock_client_instance = MockMlflowClient.return_value
        mock_client_instance.get_model_version_by_alias.return_value = mock_alias_info

        mock_download_artifacts.return_value = "/tmp/mock_pipeline.pkl"

        mock_pipeline = Mock()
        mock_joblib_load.return_value = mock_pipeline
        
        response = client.get('/reload')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'message' in data
        assert 'Model and pipeline reloaded successfully.' in data['message']

        mock_model_load.assert_called_once()
        mock_client_instance.get_model_version_by_alias.assert_called_once()
        mock_download_artifacts.assert_called_once()
        mock_joblib_load.assert_called_once_with("/tmp/mock_pipeline.pkl")

def test_reload_model_failure_on_model_load(client):
    """Test reload endpoint when  model fails to load."""
    with patch('mlflow.sklearn.load_model') as mock_load:
        mock_load.side_effect = Exception("Model Load error")
        
        response = client.get('/reload')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load model/pipeline: Model Load error' in data['error']

def test_reload_model_failure_on_artifact_download(client):
    """Test reload endpoint when an artifact fails to load."""
    with patch('mlflow.sklearn.load_model') as mock_load_model, \
        patch('mlflow.artifacts.download_artifacts') as mock_download_artifacts, \
        patch('mlflow.tracking.MlflowClient') as MockMlflowClient, \
        patch(f'{APP_PATH}.MODEL_NAME', new=MOCK_MODEL_NAME), \
        patch(f'{APP_PATH}.MODEL_ALIAS', new=MOCK_MODEL_ALIAS):
        
        mock_load_model.return_value = Mock()

        mock_alias_info = Mock()
        mock_alias_info.run_id = "test_run_mock"
        MockMlflowClient.return_value.get_model_version_by_alias.return_value = mock_alias_info

        mock_download_artifacts.side_effect = Exception("Artifact download error")

        response = client.get('/reload')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load model/pipeline: Artifact download error' in data['error']

def test_reload_model_failure_on_joblib_load(client):
    """Test reload endpoint when joblib fails to load."""
    with patch('mlflow.sklearn.load_model') as mock_load_model, \
        patch('mlflow.artifacts.download_artifacts') as mock_download_artifacts, \
        patch('joblib.load') as mock_joblib_load, \
        patch('mlflow.tracking.MlflowClient') as MockMlflowClient,\
        patch(f'{APP_PATH}.MODEL_NAME', new=MOCK_MODEL_NAME), \
        patch(f'{APP_PATH}.MODEL_ALIAS', new=MOCK_MODEL_ALIAS):
        
        mock_load_model.return_value = Mock()

        mock_alias_info = Mock()
        mock_alias_info.run_id = "test_run_mock"
        MockMlflowClient.return_value.get_model_version_by_alias.return_value = mock_alias_info

        mock_download_artifacts.side_effect = "/tmp/mock_pipeline.pkl"

        mock_joblib_load.side_effect = Exception("Pipeline Load error")

        response = client.get('/reload')
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load model/pipeline: Pipeline Load error' in data['error']

def test_predict_correct_dataframe_creation(client, mock_model, mock_pipeline):
    """Test that predict correctly creates DataFrame from request."""
    with patch.dict(app.config, {"MODEL": mock_model, "PIPELINE": mock_pipeline}):
        test_data = [
            {
            "VendorID": 2,
            "tpep_pickup_datetime": "2013-01-01 00:39:00",
            "passenger_count": 3,
            "trip_distance": 3.86,
            "PULocationID": 238,
            "DOLocationID": 116
            }
        ]
        
        response = client.post(
            '/predict',
            data=json.dumps({
                'data': test_data
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        # Verify that model.predict was called
        mock_model.predict.assert_called_once()
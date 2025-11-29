import json
import pytest
from flask import Flask
from server import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    res = client.get('/health')
    assert res.status_code == 200
    data = res.get_json()
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data

def test_predict_no_model_loaded(client, monkeypatch):
    monkeypatch.setitem(app.config, 'MODEL', None)
    res = client.post('/predict', json={"inputs": []})
    assert res.status_code == 503
    assert 'error' in res.get_json()

def test_predict_success(client, monkeypatch):
    class DummyModel:
        def predict(self, X):
            return [42] * len(X)
            
    class DummyPipeline:
        def transform(self, X):
            return X

    monkeypatch.setitem(app.config, 'MODEL', DummyModel())
    monkeypatch.setitem(app.config, 'PIPELINE', DummyPipeline())

    sample_input = {
        "inputs": [
            [0, 2, "2011-01-01 00:10:00", 4, 0.0, 1, 145, 145, 1, 2.9, 0.5, 0.5, 0.28, 0.0, 0.0, 4.18]
        ]
    }
    res = client.post('/predict', json=sample_input)
    assert res.status_code == 200
    data = res.get_json()
    assert 'predictions' in data
    #assert data['predictions'][0] == 42

def test_reload_model(client, monkeypatch):
    class DummyModel:
        pass

    def dummy_loader(*args, **kwargs):
        return DummyModel()
        
    def dummy_download(*args, **kwargs):
        return "dummy_path"
        
    def dummy_joblib_load(*args, **kwargs):
        return "dummy_pipeline"

    monkeypatch.setattr('app.mlflow.pyfunc.load_model', dummy_loader)
    monkeypatch.setattr('app.mlflow.sklearn.load_model', dummy_loader)
    monkeypatch.setattr('app.mlflow.artifacts.download_artifacts', dummy_download)
    monkeypatch.setattr('app.joblib.load', dummy_joblib_load)
    
    # Mock MlflowClient
    class DummyClient:
        def get_model_version_by_alias(self, *args, **kwargs):
            class VersionInfo:
                run_id = "dummy_run_id"
            return VersionInfo()
            
    monkeypatch.setattr('app.MlflowClient', DummyClient)

    res = client.get('/reload')
    assert res.status_code == 200
    data = res.get_json()
    assert 'message' in data
    assert app.config['MODEL'] is not None
    assert app.config['PIPELINE'] is not None
    
def test_placeholder():
    """Placeholder test that always passes."""
    assert True
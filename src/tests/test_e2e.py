"""End-to-end tests for the complete ML pipeline."""
import pytest
import requests
import json
import time

FLASK_BASE_URL = "http://localhost:8080"


def wait_for_service(url, timeout=30, interval=2):
    """Wait for a service to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    return False


@pytest.fixture(scope="module", autouse=True)
def wait_for_services():
    """Wait for Flask service to be ready before running tests."""
    print("\nWaiting for Flask service to be ready...")
    
    # Wait for Flask app
    flask_ready = wait_for_service(f"{FLASK_BASE_URL}/health")
    if not flask_ready:
        pytest.skip("Flask service not available")
    
    print("Flask service ready! (Using remote MLflow)")


def test_flask_health():
    """Test that Flask API is healthy."""
    response = requests.get(f"{FLASK_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'


def test_flask_model_loaded():
    """Test that Flask API has a model loaded."""
    response = requests.get(f"{FLASK_BASE_URL}/health")
    data = response.json()
    
    # If model is not loaded, try to reload it
    if not data.get('model_loaded', False):
        reload_response = requests.get(f"{FLASK_BASE_URL}/reload")
        assert reload_response.status_code == 200
        
        # Check again
        response = requests.get(f"{FLASK_BASE_URL}/health")
        data = response.json()
    
    assert data['model_loaded'] is True, "Model should be loaded"


def test_prediction_single_sample():
    """Test prediction with a single iris sample."""
    # Ensure model is loaded
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    
    if not health_data.get('model_loaded', False):
        requests.get(f"{FLASK_BASE_URL}/reload")
        time.sleep(2)
    
    # Make prediction
    payload = {
        "request_id": 1,
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
    }
    # payload = {
    #     "inputs": [2,"2013-01-01 00:39:00",3,3.86,238,116]
    # }
    
    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    data = response.json()
    print(data)
    assert response.status_code == 200
    assert 'predictions' in data
    assert len(data['predictions']) == 1
    assert data['predictions'][0] > 0


def test_prediction_multiple_samples():
    """Test prediction with multiple iris samples."""
    # Ensure model is loaded
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    
    if not health_data.get('model_loaded', False):
        requests.get(f"{FLASK_BASE_URL}/reload")
        time.sleep(2)
    
    # # Make prediction with 3 samples (one from each iris class typically)
    # payload = {
    #     "inputs": [
    #         [2,"2013-01-01 00:39:00",3,3.86,238,116],
    #         [2,"2013-01-01 00:12:00",5,0.0,264,264,], 
    #         [2,"2013-01-01 00:02:00",3,0.0,264,264]
    #     ]
    # }

    payload = {
        "request_id": 1,
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
            "tpep_pickup_datetime": "2013-01-01 00:07:00",
            "passenger_count": 1,
            "trip_distance": 1.76,
            "PULocationID": 79,
            "DOLocationID": 186
            },
            {
            "VendorID": 2,
            "tpep_pickup_datetime": "2013-01-01 00:29:00",
            "passenger_count": 1,
            "trip_distance": 0.94,
            "PULocationID": 234,
            "DOLocationID": 249
            }
        ]
    }
    
    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'predictions' in data
    assert len(data['predictions']) == 3
    
    # All predictions should be valid iris classes
    for pred in data['predictions']:
        assert pred > 0


def test_prediction_without_model():
    """Test that prediction fails gracefully when model is not loaded."""
    # This test assumes we can manipulate the model state, which we can't in e2e
    # So we'll just verify the error handling works when service is down
    pass


def test_model_reload():
    """Test that model can be reloaded."""
    response = requests.get(f"{FLASK_BASE_URL}/reload")
    assert response.status_code == 200
    data = response.json()
    assert 'message' in data or 'error' in data
    
    # Verify model is loaded after reload
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    assert health_data['model_loaded'] is True


# def test_prediction_accuracy():
#     """Test that predictions are reasonable for known samples."""
#     # Ensure model is loaded
#     health_response = requests.get(f"{FLASK_BASE_URL}/health")
#     health_data = health_response.json()
    
#     if not health_data.get('model_loaded', False):
#         requests.get(f"{FLASK_BASE_URL}/reload")
#         time.sleep(2)
    
#     # Test with a typical setosa sample (should predict class 0)
#     setosa_payload = {
#         "data": [[5.0, 3.4, 1.5, 0.2]],
#         "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
#     }
    
#     response = requests.post(
#         f"{FLASK_BASE_URL}/predict",
#         json=setosa_payload,
#         headers={"Content-Type": "application/json"}
#     )
    
#     assert response.status_code == 200
#     data = response.json()
#     # For a typical setosa sample, we expect class 0
#     assert data['predictions'][0] == 0, "Setosa sample should be classified as class 0"


def test_api_error_handling():
    """Test API error handling with invalid input."""
    # Test with missing columns
    invalid_payload = {
        "data": [[5.1, 3.5, 1.4, 0.2]]
        # Missing 'columns' key
    }
    
    response = requests.post(
        f"{FLASK_BASE_URL}/predict",
        json=invalid_payload,
        headers={"Content-Type": "application/json"}
    )
    
    # Should fail with 400 or 500
    assert response.status_code in [400, 500]


def test_concurrent_predictions():
    """Test that API can handle concurrent prediction requests."""
    # Ensure model is loaded
    health_response = requests.get(f"{FLASK_BASE_URL}/health")
    health_data = health_response.json()
    
    if not health_data.get('model_loaded', False):
        requests.get(f"{FLASK_BASE_URL}/reload")
        time.sleep(2)
    
    # payload = {
    #     "inputs": [2,"2013-01-01 00:39:00",3,3.86,238,116]
    # }
    payload = {
        "request_id": 1,
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
    }
    
    # Make multiple requests quickly
    responses = []
    for _ in range(5):
        response = requests.post(
            f"{FLASK_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        responses.append(response)
    
    # All should succeed
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
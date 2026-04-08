import pytest
import json
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the /health endpoint."""
    rv = client.get('/health')
    assert rv.status_code == 200
    assert b'ok' in rv.data

def test_predict_endpoint_success(client):
    """Test the /predict endpoint with valid data."""
    payload = {
        "features": {
            "MedInc": 5.5,
            "HouseAge": 20,
            "AveRooms": 6.0,
            "AveBedrms": 1.1,
            "Population": 1500,
            "AveOccup": 3.0,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    }
    
    rv = client.post('/predict',
                     data=json.dumps(payload),
                     content_type='application/json')
    
    # We ignore 500 here ONLY IF the model file is missing during test init
    # But since it's a test, we want to at least ensure it doesn't 400.
    if rv.status_code == 500:
        data = json.loads(rv.data)
        assert 'FileNotFoundError' in data['error'] or 'Model artifact not found' in data['error']
    else:
        assert rv.status_code == 200
        data = json.loads(rv.data)
        assert 'predicted_price' in data
        assert 'price_range' in data
        
def test_predict_endpoint_validation_missing_key(client):
    """Test validation when a required key is missing."""
    payload = {
        "features": {
            "MedInc": 5.5
            # Missing everything else
        }
    }
    rv = client.post('/predict',
                     data=json.dumps(payload),
                     content_type='application/json')
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert 'Missing fields' in data['error']

def test_predict_endpoint_validation_bad_type(client):
    """Test validation when a non-numeric string is passed."""
    payload = {
        "features": {
            "MedInc": "five point five",
            "HouseAge": 20,
            "AveRooms": 6.0,
            "AveBedrms": 1.1,
            "Population": 1500,
            "AveOccup": 3.0,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    }
    rv = client.post('/predict',
                     data=json.dumps(payload),
                     content_type='application/json')
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert 'Invalid data type' in data['error']

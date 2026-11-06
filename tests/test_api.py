from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    """Test 1: Verify the health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_invalid_input():
    """Test 2: Verify the API catches missing features (422 Error)."""
    # Send only 29 features instead of the required 30
    payload = {"features": [0.0] * 29}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
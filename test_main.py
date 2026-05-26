from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_add():
    response = client.get("/add", params={"a": 2, "b": 3})
    assert response.status_code == 200
    assert response.json() == {"result": 5}
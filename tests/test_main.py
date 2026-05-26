from main import app

def test_add():
    response = app.get("/add?a=1&b=2")
    assert response.status_code == 200
    assert response.json()["result"] == 3
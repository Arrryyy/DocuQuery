import pytest
from src.app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_get(client):
    response = client.get("/")
    assert response.status_code == 200
    # Check that the form contains a field for query input
    assert b"Ask a Question" in response.data

def test_index_post_empty_query(client):
    response = client.post("/", data={"query": ""})
    assert response.status_code == 200
    assert b"Please enter a query" in response.data
import pytest
from fastapi.testclient import TestClient

from skuld.api.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app=app)


def test_health(client: TestClient):
    assert client.get("/health").status_code == 200

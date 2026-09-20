import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
AUTH = {"Authorization": "Bearer 00000"}


@pytest.fixture
def create_data():
    return json.loads((ROOT / "data/example_request.json").read_text())


@pytest.fixture
def init_test_client(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(ROOT / "models/pipeline.pkl"))
    from main import app
    with TestClient(app) as client:
        yield client


def test_healthcheck(init_test_client):
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference(init_test_client, create_data):
    response = init_test_client.post(
        "/predictions", headers=AUTH, json=create_data)
    assert response.status_code == 200
    value = response.json()["predicted_traffic_volume"]
    assert isinstance(value, (float, int))
    assert math.isfinite(value) and value >= 0
    from model_utils import make_inference
    expected = make_inference(init_test_client.app.state.model, create_data)
    assert response.json() == expected


def test_wrong_token(init_test_client, create_data):
    response = init_test_client.post(
        "/predictions", headers={"Authorization": "Bearer wrong"},
        json=create_data)
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"}


def test_no_token(init_test_client, create_data):
    response = init_test_client.post("/predictions", json=create_data)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_missing_field(init_test_client, create_data):
    del create_data["hour"]
    response = init_test_client.post(
        "/predictions", headers=AUTH, json=create_data)
    assert response.status_code == 422


def test_wrong_numeric_type(init_test_client, create_data):
    create_data["temperature_c"] = "not a number"
    response = init_test_client.post(
        "/predictions", headers=AUTH, json=create_data)
    assert response.status_code == 422


def test_negative_rain(init_test_client, create_data):
    create_data["rain_1h"] = -1
    response = init_test_client.post(
        "/predictions", headers=AUTH, json=create_data)
    assert response.status_code == 422

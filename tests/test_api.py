import pytest
from fastapi.testclient import TestClient

from api.main import EXAMPLE_LISTING, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_version_reports_metrics(client):
    body = client.get("/version").json()
    assert body["model_type"] == "GradientBoostingRegressor"
    assert "r2_log" in body["test_metrics"]


def test_predict_one(client):
    r = client.post("/predict_one", json=EXAMPLE_LISTING)
    assert r.status_code == 200
    body = r.json()
    assert body["currency"] == "AUD"
    assert 200_000 < body["predictions"][0]["price_aud"] < 5_000_000


def test_optional_fields_can_be_omitted(client):
    listing = {
        k: v
        for k, v in EXAMPLE_LISTING.items()
        if k not in {"sale_method", "seller", "suburb_property_count"}
    }
    assert client.post("/predict_one", json=listing).status_code == 200


def test_unknown_suburb_still_predicts(client):
    r = client.post("/predict_one", json={**EXAMPLE_LISTING, "suburb": "Atlantis"})
    assert r.status_code == 200


def test_batch(client):
    r = client.post("/predict", json=[EXAMPLE_LISTING, {**EXAMPLE_LISTING, "bedrooms": 5}])
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict", json=[]).status_code == 422


def test_missing_required_field_rejected(client):
    listing = {k: v for k, v in EXAMPLE_LISTING.items() if k != "suburb"}
    assert client.post("/predict_one", json=listing).status_code == 422


def test_invalid_property_type_rejected(client):
    r = client.post("/predict_one", json={**EXAMPLE_LISTING, "property_type": "castle"})
    assert r.status_code == 422

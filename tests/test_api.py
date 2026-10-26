import pytest
from fastapi.testclient import TestClient

from swotbot.api import _clean_legacy_text, app
from swotbot.model import LABELS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert sorted(r.json()["classes"]) == sorted(LABELS)


def test_predict_batch(client):
    texts = [
        "Long-standing presence in industry coupled with strong promoter group support.",
        "Exposure to foreign exchange fluctuation risk and intense competition.",
    ]
    r = client.post("/predict", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    for item in body:
        assert item["label"] in LABELS
        assert abs(sum(item["scores"].values()) - 1) < 1e-3


def test_predict_rejects_empty(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422
    assert client.post("/predict", json={"texts": [""]}).status_code == 422


def test_legacy_endpoint_matches_word_addin(client):
    r = client.post("/api/", params={"TEXT": "['Manpower attrition risk along with increasing salary cost.']"})
    assert r.status_code == 200
    label, pct = r.json()
    assert label in LABELS and 0 <= pct <= 100


def test_legacy_endpoint_requires_text(client):
    assert client.post("/api/").status_code == 400


@pytest.mark.parametrize(
    "raw,expected",
    [("['abc']", "abc"), ('["abc"]', "abc"), ("abc", "abc"), ("%5B%27a%20b%27%5D", "a b")],
)
def test_clean_legacy_text(raw, expected):
    assert _clean_legacy_text(raw) == expected

from fastapi.testclient import TestClient

from services.api.main import app


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_predict_endpoint():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "There is an unauthorized charge on my debit card"})
        assert r.status_code == 200
        body = r.json()
        assert body["category"] == "fraud_unauthorized_transactions"
        assert set(body) == {"category", "confidence", "priority"}


def test_predict_rejects_short_text():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "help"})
        assert r.status_code == 422

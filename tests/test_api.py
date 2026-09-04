import requests
from fastapi.testclient import TestClient

from services.api import main as api_main
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


def test_draft_reply_when_rag_is_up(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "answer": "Investigate within 10 business days...",
                "citations": [
                    {"source": "reg_e_error_resolution.md", "heading": "Investigation timeline", "score": 0.5}
                ],
                "model": "retrieval-fallback",
            }

    monkeypatch.setattr(api_main.requests, "post", lambda *a, **k: FakeResp())
    with TestClient(app) as client:
        r = client.post("/draft-reply", json={"text": "Someone made withdrawals from my account that I never authorized"})
        assert r.status_code == 200
        body = r.json()
        assert body["rag_available"] is True
        assert body["category"] == "fraud_unauthorized_transactions"
        assert body["citations"][0]["source"] == "reg_e_error_resolution.md"


def test_draft_reply_falls_back_when_rag_is_down(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("rag service down")

    monkeypatch.setattr(api_main.requests, "post", boom)
    with TestClient(app) as client:
        r = client.post("/draft-reply", json={"text": "Three overdraft fees hit my account in a single day"})
        assert r.status_code == 200
        body = r.json()
        assert body["rag_available"] is False
        assert body["category"] == "overdraft_fees"
        assert body["citations"] == []

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests
from fastapi import Depends, FastAPI, HTTPException

from . import storage
from .model.classifier import ComplaintClassifier
from .schemas import (
    Complaint,
    ComplaintIn,
    DraftReplyIn,
    DraftReplyOut,
    PredictIn,
    PredictOut,
    StatsOut,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = ComplaintClassifier()
    yield


app = FastAPI(
    title="Complaint Triage API",
    version="1.0.0",
    lifespan=lifespan,
)

RAG_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")


def get_conn():
    conn = storage.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    return app.state.classifier.predict(payload.text)


@app.post("/complaints", response_model=Complaint, status_code=201)
def create_complaint(payload: ComplaintIn, conn=Depends(get_conn)):
    result = app.state.classifier.predict(payload.text)
    row = storage.insert(
        conn,
        customer_ref=payload.customer_ref,
        text=payload.text,
        channel=payload.channel.value,
        category=result["category"],
        priority=result["priority"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return dict(row)


@app.get("/complaints")
def list_complaints(conn=Depends(get_conn)):
    return [dict(r) for r in storage.fetch_all(conn)]


@app.get("/complaints/{complaint_id}", response_model=Complaint)
def get_complaint(complaint_id: int, conn=Depends(get_conn)):
    row = storage.fetch_one(conn, complaint_id)
    if row is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    return dict(row)


@app.post("/draft-reply", response_model=DraftReplyOut)
def draft_reply(payload: DraftReplyIn):
    """Classify the complaint, then call the RAG service for a grounded draft.

    Service-to-service orchestration against RAG_URL with graceful degradation:
    if the RAG service is down, triage still returns with rag_available=False.
    """
    triage = app.state.classifier.predict(payload.text)
    draft = "Policy assistant unavailable - triage returned, draft the reply manually."
    citations = []
    model = None
    rag_available = False
    try:
        resp = requests.post(
            f"{RAG_URL}/ask",
            json={"question": payload.text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        draft = data["answer"]
        citations = data["citations"]
        model = data["model"]
        rag_available = True
    except requests.RequestException:
        pass
    return {
        **triage,
        "draft_answer": draft,
        "citations": citations,
        "model": model,
        "rag_available": rag_available,
    }


@app.get("/stats/summary", response_model=StatsOut)
def summary(conn=Depends(get_conn)):
    total, rows, high = storage.stats(conn)
    return {
        "total": total,
        "by_category": [{"category": r["category"], "count": r["n"]} for r in rows],
        "high_priority": high,
    }


@app.get("/health")
def health():
    return {"status": "ok"}

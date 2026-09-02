import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .retriever import PolicyRetriever

SYSTEM_PROMPT = (
    "You are an assistant for customer service agents at a US credit union. "
    "Draft a member-facing reply using ONLY the policy excerpts provided. "
    "Do not invent fee amounts, deadlines, or promises. If the excerpts do not "
    "cover the question, say so and suggest escalation."
)


class AskIn(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = 3


class Citation(BaseModel):
    source: str
    heading: str
    score: float


class AskOut(BaseModel):
    answer: str
    citations: list[Citation]
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = PolicyRetriever()
    yield


app = FastAPI(title="Policy RAG Service", version="1.0.0", lifespan=lifespan)


def compose_fallback_answer(question: str, hits) -> str:
    """Used when no OPENAI_API_KEY is set: stitch the top sections into a
    readable draft instead of leaving agents empty handed."""
    parts = [c.text for c, score in hits if score > 0]
    if not parts:
        return (
            "I could not find a policy section that matches this question. "
            "Escalate to the operations team for guidance."
        )
    bullets = []
    for chunk, _ in hits[:2]:
        lines = [ln.strip("- ").strip() for ln in chunk.text.splitlines() if ln.strip()][:4]
        bullets.append(
            f"From {chunk.heading} ({chunk.source}): " + "; ".join(lines)
        )
    return "\n\n".join(bullets)


def llm_answer(question: str, hits) -> str:
    from openai import OpenAI  # imported lazily so the service runs without the key

    context = "\n\n".join(
        f"[{c.heading} — {c.source}]\n{c.text}" for c, _ in hits
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("COPILOT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n{context}\n\nAgent question: {question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    hits = app.state.retriever.search(payload.question, k=payload.top_k)
    citations = [
        {"source": c.source, "heading": c.heading, "score": s} for c, s in hits
    ]
    if os.getenv("OPENAI_API_KEY"):
        try:
            answer = llm_answer(payload.question, hits)
            model = os.getenv("COPILOT_MODEL", "gpt-4o-mini")
        except Exception:
            answer = compose_fallback_answer(payload.question, hits)
            model = "retrieval-fallback"
    else:
        answer = compose_fallback_answer(payload.question, hits)
        model = "retrieval-fallback"
    return {"answer": answer, "citations": citations, "model": model}


@app.get("/health")
def health():
    return {"status": "ok"}

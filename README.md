# bank-complaint-copilot

A triage and response-drafting tool for customer complaints at a US community bank or credit union.

I built this around a very specific problem: at a lot of smaller banks, every inbound complaint — a disputed debit charge, a Zelle scam, a missing wire, an overdraft fee — lands in the same shared inbox, and someone has to read each one, figure out what kind of issue it is, decide how urgent it is, and then hunt through policy manuals to write back something that is actually accurate under Regulation E / Regulation Z timelines. That is slow, inconsistent, and risky when deadlines like the 10-business-day Reg E provisional credit window get missed.

This repo has three parts wired together:

- **complaint-api** — a FastAPI REST service that accepts complaints, runs them through a scikit-learn NLP classifier, tags category and priority, and stores everything in SQLite. Also exposes triage stats endpoints for a dashboard.
- **rag-service** — a retrieval service that indexes the bank's internal policy docs (markdown) and answers agent questions by pulling the relevant policy sections and drafting a reply with an LLM. If no OpenAI key is configured it falls back to an extractive draft built from the retrieved sections, so it works offline.
- **ui** — a Streamlit front end with three tabs: a chatbot for agents, a complaint intake form with live classification, and a triage board with counts and charts.

## Requirements

- Python 3.12+
- (optional) an `OPENAI_API_KEY` for LLM-drafted answers

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
pip install -r requirements.txt

uvicorn services.api.main:app --port 8001 --reload
uvicorn services.rag.main:app --port 8002 --reload
streamlit run services/ui/app.py
```

Then open http://localhost:8501.

The classifier trains itself on first use from the sample data in `services/api/model/train.py` and caches the model as a joblib file. That takes about a second.

Or with Docker:

```bash
docker compose up --build
```

## API quick reference

- `POST /complaints` — file a complaint; response includes the auto-assigned `category` and `priority`
- `POST /predict` — classify complaint text without storing it
- `GET /complaints`, `GET /complaints/{id}` — list / fetch
- `GET /stats/summary` — totals, per-category counts, high-priority count
- `POST /ask` (rag service) — retrieve policy sections and draft a reply, returns citations

## Design notes

- The classifier is TF-IDF + logistic regression. I picked that over a heavier transformer model on purpose: complaint categories are well separated by vocabulary ("overdraft fee", "wire", "APR"), and this trains in under a second and runs on a laptop. Swapping in embeddings later is a one-file change in the retriever/classifier.
- Retrieval also uses TF-IDF + cosine similarity rather than a vector database, for the same reason — the corpus is five policy docs. The retrieval code is isolated in `services/rag/retriever.py` so Chroma or pgvector can slot in without touching the API.
- The LLM only ever sees retrieved policy excerpts plus the agent's question, and the system prompt forbids inventing numbers. The citations come back with every answer so an agent can check the source section before sending anything to a customer.
- Priority is rule-based on top of the classifier: fraud, P2P scams, and mortgage servicing are always flagged high because of regulatory deadlines.

## Tests

```bash
pytest -q
```

## What's next

- Train on the real CFPB public complaint database and cross-validate category accuracy
- Auth on the API (internal SSO / JWT) before anything like this touches real customer data
- Swap TF-IDF retrieval for embeddings once the policy corpus grows past a few dozen docs

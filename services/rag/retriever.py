"""TF-IDF based retrieval over the policy docs folder.

Deliberately lightweight: no vector DB, no GPU. Each markdown section becomes a
chunk, we vectorize once at startup, and answer questions by cosine similarity.
Swap `TfidfVectorizer` for sentence embeddings + Chroma later if the corpus grows.
"""
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "docs"


@dataclass
class Chunk:
    source: str
    heading: str
    text: str


def load_chunks() -> list[Chunk]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        heading = None
        buf = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") and buf:
                chunks.append(Chunk(path.name, heading or path.stem, "\n".join(buf).strip()))
                buf = []
            if line.lstrip().startswith("#"):
                heading = line.strip("# ").strip()
            else:
                buf.append(line)
        if buf:
            chunks.append(Chunk(path.name, heading or path.stem, "\n".join(buf).strip()))
    return chunks


class PolicyRetriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [(self.chunks[i], round(float(sims[i]), 3)) for i in top]

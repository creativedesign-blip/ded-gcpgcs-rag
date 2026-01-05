from typing import List, Dict
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_DB: list[dict] = []

def embed(text: str) -> list[float]:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding

def upsert_text(text: str, meta: Dict):
    VECTOR_DB.append({
        "embedding": embed(text),
        "text": text,
        "meta": meta,
    })

def cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb)

def search(query: str, top_k: int = 4) -> List[Dict]:
    if not VECTOR_DB:
        return []

    q_emb = embed(query)

    scored = []
    for d in VECTOR_DB:
        score = cosine(q_emb, d["embedding"])
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

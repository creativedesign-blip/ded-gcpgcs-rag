import os
import math
from dotenv import load_dotenv
from openai import OpenAI

# 確保 .env 會被載入（本機跑 uvicorn 最常卡這裡）
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== 簡單的 in-memory vector store（POC） ======
VECTOR_STORE = []  # 每筆：{text, meta, embedding}

def _require_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Please set it in .env or environment variables.")

def embed(text: str):
    _require_key()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def add_document(text: str, meta: dict):
    if not text or not text.strip():
        return
    VECTOR_STORE.append({
        "text": text.strip(),
        "meta": meta or {},
        "embedding": embed(text.strip())
    })

def search(query: str, top_k=4):
    if not VECTOR_STORE:
        return []
    q_emb = embed(query)
    scored = [(cosine(q_emb, d["embedding"]), d) for d in VECTOR_STORE]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

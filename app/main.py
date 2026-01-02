import os
import json
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI

from app.ingest import ingest_payload
from app.rag import search
from app.prompt import SYSTEM_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="DED_GCPGCS_RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC 直接全開，之後上線再鎖 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
STATIC_DIR = os.path.abspath(STATIC_DIR)

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/popup", response_class=HTMLResponse)
def popup():
    return FileResponse(os.path.join(STATIC_DIR, "popup.html"))

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/ingest")
async def api_ingest(payload: dict):
    """
    讓爬蟲或其他服務用 JSON 丟資料進來：
    { "text": "...", "source": "crawler", "filename": "xxx.txt" }
    """
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    meta = ingest_payload(
        text=text,
        source=payload.get("source", "manual"),
        filename=payload.get("filename"),
    )
    return {"ingested": True, "meta": meta}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """
    上傳 TXT / MD / 任何純文字檔，讀進來後 ingest
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # 有些 Windows txt 是 cp950/big5，這裡做一次 fallback
        text = raw.decode("cp950", errors="ignore")

    meta = ingest_payload(text=text, source="upload", filename=file.filename)
    return {"uploaded": True, "meta": meta, "chars": len(text)}

def build_rag_text(user_query: str) -> str:
    docs = search(user_query, top_k=4)
    if not docs:
        return "（目前沒有已匯入的文件內容）"

    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.get("meta", {})
        prefix = f"[{i}] source={meta.get('source')} file={meta.get('filename')} ts={meta.get('ts')}"
        lines.append(prefix)
        lines.append(d["text"])
        lines.append("")
    return "\n".join(lines).strip()

@app.get("/api/stream")
def stream_answer(q: str = Query(..., description="user query")):
    """
    SSE：前端用 EventSource 直接接這個 endpoint
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    rag_text = build_rag_text(q)

    def event_gen():
        # 先送一個「開始」訊號（前端可以用來顯示狀態）
        yield "event: status\ndata: start\n\n"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"RAG Snippets:\n{rag_text}"},
            {"role": "user", "content": q},
        ]

        try:
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    # SSE data 不能直接換行，簡單處理：把 \n 變成 \\n，前端再轉回來
                    safe = delta.content.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

            yield "event: status\ndata: done\n\n"
        except Exception as e:
            msg = str(e).replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from google.cloud import storage

from app.ingest import ingest_payload
from app.rag import search
from app.prompt import SYSTEM_PROMPT

# ============ basic setup ============
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DED_GCPGCS_RAG")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="DED_GCPGCS_RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC 直接全開，之後上線再鎖 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))

# GCS client (Cloud Run 上會用預設 Service Account)
_gcs_client: Optional[storage.Client] = None
_processed_event_ids = set()


def get_gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp950", errors="ignore")


# ============ routes ============
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
    text = decode_text(raw)

    meta = ingest_payload(text=text, source="upload", filename=file.filename)
    return {"uploaded": True, "meta": meta, "chars": len(text)}


# ✅ Eventarc → Cloud Run 會打這支
@app.post("/api/eventarc/gcs")
async def eventarc_gcs(request: Request):
    """
    Eventarc (GCS object finalized) 觸發後，會 POST CloudEvent 到 Cloud Run。
    我們在這裡：
    1) 解析 bucket + object name
    2) 去 GCS 下載檔案內容
    3) ingest_payload() 進 RAG (in-memory)
    """
    ce_id = request.headers.get("ce-id") or request.headers.get("Ce-Id")
    ce_type = request.headers.get("ce-type") or request.headers.get("Ce-Type")
    body = await request.json()

    # 簡單去重（同一個 event 不重複吃）
    if ce_id and ce_id in _processed_event_ids:
        return {"ok": True, "skipped": True, "reason": "duplicate", "ce_id": ce_id}
    if ce_id:
        _processed_event_ids.add(ce_id)

    # Eventarc 的 payload 有時候是直接包含 name/bucket，有時候在 data 裡
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, dict):
        bucket = data.get("bucket")
        name = data.get("name")
    else:
        bucket = body.get("bucket") if isinstance(body, dict) else None
        name = body.get("name") if isinstance(body, dict) else None

    if not bucket or not name:
        logger.warning("Eventarc payload missing bucket/name. headers=%s body=%s", dict(request.headers), body)
        raise HTTPException(status_code=400, detail="bucket/name not found in event payload")

    # 只吃你要的格式（先 txt / md）
    lower = name.lower()
    if not (lower.endswith(".txt") or lower.endswith(".md")):
        logger.info("Skip non-text object: gs://%s/%s", bucket, name)
        return {"ok": True, "skipped": True, "reason": "not_txt_or_md", "bucket": bucket, "name": name}

    logger.info("Eventarc received: ce_id=%s ce_type=%s gs://%s/%s", ce_id, ce_type, bucket, name)

    # 下載檔案
    gcs = get_gcs_client()
    bkt = gcs.bucket(bucket)
    blob = bkt.blob(name)
    raw = blob.download_as_bytes()
    text = decode_text(raw)

    meta = ingest_payload(text=text, source="gcs", filename=name)
    logger.info("Ingested from GCS: gs://%s/%s chars=%s", bucket, name, len(text))

    return {"ok": True, "bucket": bucket, "name": name, "meta": meta, "chars": len(text)}


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
                    safe = delta.content.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

            yield "event: status\ndata: done\n\n"
        except Exception as e:
            msg = str(e).replace("\n", " ")
            yield f"event: error\ndata: {msg}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

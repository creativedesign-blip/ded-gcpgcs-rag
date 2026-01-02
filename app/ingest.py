from datetime import datetime
from app.rag import add_document

def ingest_payload(text: str, source: str = "manual", filename: str = None):
    """
    統一 ingest 入口：爬蟲/上傳/未來 GCS event 都走這裡
    """
    meta = {
        "source": source,
        "filename": filename,
        "ts": datetime.utcnow().isoformat()
    }
    add_document(text=text, meta=meta)
    return meta

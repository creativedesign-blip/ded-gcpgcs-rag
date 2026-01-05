import time
from google.cloud import storage
from app.rag import upsert_text

def ingest_payload(text: str, source: str, filename: str | None):
    meta = {
        "source": source,
        "filename": filename,
        "ts": int(time.time()),
    }
    upsert_text(text, meta)
    return meta

def ingest_gcs_object(bucket_name: str, object_name: str):
    # 只處理 ingest/ 底下的 txt
    if not object_name.startswith("ingest/"):
        return
    if not object_name.endswith(".txt"):
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    if not blob.exists():
        return

    text = blob.download_as_text(encoding="utf-8")

    ingest_payload(
        text=text,
        source="gcs",
        filename=object_name,
    )

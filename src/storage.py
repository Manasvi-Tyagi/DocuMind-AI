"""Optional: store uploaded documents in Google Cloud Storage (set GCS_BUCKET)."""

import hashlib
from datetime import datetime, timezone

from src.config import settings


def gcs_enabled() -> bool:
    return bool(settings.gcs_bucket)


def upload_document(filename: str, data: bytes, content_type: str | None = None) -> str:
    """Upload to gs://<GCS_BUCKET>/uploads/... and return the gs:// URI."""

    from google.cloud import storage  # imported only when GCS is used

    digest = hashlib.sha256(data).hexdigest()[:16]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    blob_name = f"uploads/{stamp}/{digest}-{filename}"

    client = storage.Client()
    blob = client.bucket(settings.gcs_bucket).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{settings.gcs_bucket}/{blob_name}"

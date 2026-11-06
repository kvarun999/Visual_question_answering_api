import hashlib
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import get_settings


def audit_image_payload(request_id: str, image_bytes: bytes) -> str:
    """Calculates the SHA-256 hash of the image bytes and appends an entry to audit.log.

    This cryptographic auditing mechanism provides an immutable record proving
    that the exact image received into memory was not dropped or altered before VLM processing.

    Args:
        request_id: Unique UUID string identifying the request.
        image_bytes: Raw binary data of the uploaded image.

    Returns:
        The SHA-256 hex digest string.
    """
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = f"{timestamp} | {request_id} | {image_hash}\n"

    settings = get_settings()
    log_path = Path(settings.audit_log_path)
    if log_path.parent:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)
        f.flush()

    return image_hash


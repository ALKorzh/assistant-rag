import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import get_rag_service
from app.api.security import require_api_key
from app.core.config import get_settings
from app.schemas.chat import UploadResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])

_WHITESPACE_RE = re.compile(r"\s+")


def _safe_stored_name(filename: str | None, settings) -> str:
    raw = (filename or "upload").strip().replace("\\", "/").split("/")[-1]
    raw = raw or "upload"
    raw = _WHITESPACE_RE.sub("_", raw)
    if len(raw) > 180:
        raw = raw[:180]
    path = Path(raw)
    if path.name in (".", "..") or not path.name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suffix = path.suffix.lower()
    if suffix not in settings.allowed_upload_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension not allowed. Use one of: {', '.join(settings.allowed_upload_extensions)}",
        )
    return path.name


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> UploadResponse:
    """Persist the uploaded file and index it into the vector store."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = _safe_stored_name(file.filename, settings)
    target_path = upload_dir / safe_filename
    rid = getattr(request.state, "request_id", "-")

    logger.info("request_id=%s Upload start name=%s", rid, safe_filename)

    max_bytes = settings.upload_max_bytes
    total = 0
    chunk_size = 1024 * 512

    try:
        with open(target_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {max_bytes} bytes",
                    )
                buffer.write(chunk)
    except HTTPException:
        target_path.unlink(missing_ok=True)
        raise

    result = await run_in_threadpool(rag_service.process_file, str(target_path))
    logger.info("request_id=%s Upload done name=%s bytes=%d", rid, safe_filename, total)
    return UploadResponse(status=result)

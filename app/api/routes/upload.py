import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_rag_service
from app.core.config import get_settings
from app.schemas.chat import UploadResponse
from app.services.rag_service import RAGService


router = APIRouter(tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> UploadResponse:
    """Persist the uploaded file and index it into the vector store."""
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename or "uploaded_file").name
    target_path = upload_dir / safe_filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = rag_service.process_file(str(target_path))
    return UploadResponse(status=result)

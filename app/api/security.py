import logging
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> None:
    settings = get_settings()
    expected = settings.api_key
    if not expected:
        return
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        logger.warning("Rejected request: invalid or missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

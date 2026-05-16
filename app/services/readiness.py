import logging
from typing import Any

import requests
from qdrant_client import QdrantClient

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def check_readiness() -> dict[str, Any]:
    settings = get_settings()
    qdrant_ok, qdrant_detail = _check_qdrant(settings)
    ollama_ok, ollama_detail = _check_ollama(settings)
    ready = qdrant_ok and ollama_ok
    return {
        "ready": ready,
        "checks": {
            "qdrant": {"ok": qdrant_ok, "detail": qdrant_detail},
            "ollama": {"ok": ollama_ok, "detail": ollama_detail},
        },
    }


def _check_qdrant(settings) -> tuple[bool, str]:
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=5)
        client.get_collections()
        return True, "ok"
    except Exception as exc:
        logger.warning("Qdrant readiness check failed: %s", exc)
        return False, str(exc)


def _check_ollama(settings) -> tuple[bool, str]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        data = response.json()
        models = {m.get("name", "") for m in data.get("models", [])}
        name = settings.ollama_embedding_model
        if name in models or f"{name}:latest" in models or any(name in m for m in models):
            return True, f"model {name} available"
        return False, f"embedding model '{name}' not pulled yet"
    except Exception as exc:
        logger.warning("Ollama readiness check failed: %s", exc)
        return False, str(exc)

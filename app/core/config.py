import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralised runtime configuration loaded from environment variables."""

    api_title: str = "Agentic RAG Assistant"
    api_description: str = "Personal assistant with routing, RAG, and external tools."
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    gemini_model: str = "gemini-2.5-flash"
    gemini_router_temperature: float = 0.0
    gemini_answer_temperature: float = 0.7
    google_api_key: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "my_documents"
    ollama_embedding_model: str = "nomic-embed-text"

    chunk_size: int = 600
    chunk_overlap: int = 100
    rag_search_limit: int = 3
    upload_dir: str = "data/raw"

    openweather_api_key: str | None = None
    openweather_url: str = "https://api.openweathermap.org/data/2.5/weather"
    openweather_timeout_seconds: int = 5

    wikipedia_language: str = "ru"
    wikipedia_results_limit: int = 1
    wikipedia_content_limit: int = 2000

    youtube_results_limit: int = 3


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once and reuse the same instance across the app."""
    return Settings(
        api_title=os.getenv("API_TITLE", "Agentic RAG Assistant"),
        api_description=os.getenv(
            "API_DESCRIPTION",
            "Personal assistant with routing, RAG, and external tools.",
        ),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_env_int("API_PORT", 8000),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_router_temperature=_env_float("GEMINI_ROUTER_TEMPERATURE", 0.0),
        gemini_answer_temperature=_env_float("GEMINI_ANSWER_TEMPERATURE", 0.7),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "my_documents"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 600),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
        rag_search_limit=_env_int("RAG_SEARCH_LIMIT", 3),
        upload_dir=os.getenv("UPLOAD_DIR", "data/raw"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        openweather_url=os.getenv(
            "OPENWEATHER_URL",
            "https://api.openweathermap.org/data/2.5/weather",
        ),
        openweather_timeout_seconds=_env_int("OPENWEATHER_TIMEOUT", 5),
        wikipedia_language=os.getenv("WIKIPEDIA_LANGUAGE", "ru"),
        wikipedia_results_limit=_env_int("WIKIPEDIA_RESULTS_LIMIT", 1),
        wikipedia_content_limit=_env_int("WIKIPEDIA_CONTENT_LIMIT", 2000),
        youtube_results_limit=_env_int("YOUTUBE_RESULTS_LIMIT", 3),
    )

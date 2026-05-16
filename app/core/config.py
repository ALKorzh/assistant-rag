import os
from dataclasses import dataclass, field
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
    cors_allow_origins: tuple[str, ...] = field(default_factory=lambda: ("*",))
    api_key: str | None = None

    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_router_temperature: float = 0.0
    gemini_answer_temperature: float = 0.7
    google_api_key: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "my_documents"
    qdrant_vector_size: int = 768
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    chunk_size: int = 600
    chunk_overlap: int = 100
    rag_search_limit: int = 3
    upload_dir: str = "data/raw"
    upload_max_bytes: int = 15 * 1024 * 1024
    allowed_upload_extensions: tuple[str, ...] = (".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp")

    http_retry_attempts: int = 3
    http_retry_base_delay_seconds: float = 0.5
    duckduckgo_timeout_seconds: int = 20

    youtube_results_limit: int = 3
    youtube_primary_provider: str = "youtube"
    # YouTube Data API v3 (search.list). Если задан — по умолчанию идёт первым в цепочке поиска.
    youtube_data_api_key: str | None = None
    youtube_region_code: str = "RU"
    youtube_relevance_language: str = "ru"
    youtube_data_api_timeout_seconds: int = 15

    openweather_api_key: str | None = None
    openweather_url: str = "https://api.openweathermap.org/data/2.5/weather"
    openweather_forecast_url: str = "https://api.openweathermap.org/data/2.5/forecast"
    openweather_timeout_seconds: int = 5

    wikipedia_language: str = "ru"
    wikipedia_results_limit: int = 1
    wikipedia_content_limit: int = 2000
    wikipedia_timeout_seconds: int = 12
    wikipedia_user_agent: str = "AssistantRAG/1.0 (MediaWiki API; local RAG assistant)"


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


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _normalize_extensions(items: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in items:
        piece = item.strip().lower()
        if not piece.startswith("."):
            piece = f".{piece}"
        normalized.append(piece)
    return tuple(normalized)


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
        cors_allow_origins=_env_tuple("CORS_ALLOW_ORIGINS", ("*",)),
        api_key=(k if (k := os.getenv("API_KEY", "").strip()) else None),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        gemini_router_temperature=_env_float("GEMINI_ROUTER_TEMPERATURE", 0.0),
        gemini_answer_temperature=_env_float("GEMINI_ANSWER_TEMPERATURE", 0.7),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "my_documents"),
        qdrant_vector_size=_env_int("QDRANT_VECTOR_SIZE", 768),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 600),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
        rag_search_limit=_env_int("RAG_SEARCH_LIMIT", 3),
        upload_dir=os.getenv("UPLOAD_DIR", "data/raw"),
        upload_max_bytes=_env_int("UPLOAD_MAX_BYTES", 15 * 1024 * 1024),
        allowed_upload_extensions=_normalize_extensions(
            _env_tuple(
                "ALLOWED_UPLOAD_EXTENSIONS",
                (".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"),
            )
        ),
        http_retry_attempts=_env_int("HTTP_RETRY_ATTEMPTS", 3),
        http_retry_base_delay_seconds=_env_float("HTTP_RETRY_BASE_DELAY", 0.5),
        duckduckgo_timeout_seconds=_env_int("DUCKDUCKGO_TIMEOUT", 20),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        openweather_url=os.getenv(
            "OPENWEATHER_URL",
            "https://api.openweathermap.org/data/2.5/weather",
        ),
        openweather_forecast_url=os.getenv(
            "OPENWEATHER_FORECAST_URL",
            "https://api.openweathermap.org/data/2.5/forecast",
        ),
        openweather_timeout_seconds=_env_int("OPENWEATHER_TIMEOUT", 5),
        wikipedia_language=os.getenv("WIKIPEDIA_LANGUAGE", "ru"),
        wikipedia_results_limit=_env_int("WIKIPEDIA_RESULTS_LIMIT", 1),
        wikipedia_content_limit=_env_int("WIKIPEDIA_CONTENT_LIMIT", 2000),
        wikipedia_timeout_seconds=_env_int("WIKIPEDIA_TIMEOUT", 12),
        wikipedia_user_agent=os.getenv(
            "WIKIPEDIA_USER_AGENT",
            "AssistantRAG/1.0 (MediaWiki API; local RAG assistant)",
        ),
        youtube_results_limit=_env_int("YOUTUBE_RESULTS_LIMIT", 3),
        youtube_primary_provider=(
            yp
            if (yp := os.getenv("YOUTUBE_PRIMARY_PROVIDER", "youtube").lower())
            in ("youtube", "duckduckgo")
            else "youtube"
        ),
        youtube_data_api_key=(
            k
            if (k := (os.getenv("YOUTUBE_DATA_API_KEY") or os.getenv("YOUTUBE_API_KEY", "")).strip())
            else None
        ),
        youtube_region_code=os.getenv("YOUTUBE_REGION_CODE", "RU").strip() or "RU",
        youtube_relevance_language=os.getenv("YOUTUBE_RELEVANCE_LANGUAGE", "ru").strip() or "ru",
        youtube_data_api_timeout_seconds=_env_int("YOUTUBE_DATA_API_TIMEOUT", 15),
    )

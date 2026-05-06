import logging
from functools import lru_cache

from app.agent import app_graph
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Provide a single shared RAGService instance per process."""
    logger.info("Initializing shared RAG service dependency")
    return RAGService()


def get_agent_graph():
    """Return the compiled LangGraph state machine."""
    logger.debug("Providing compiled agent graph dependency")
    return app_graph

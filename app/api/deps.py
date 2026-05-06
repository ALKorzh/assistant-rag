from functools import lru_cache

from app.agent import app_graph
from app.services.rag_service import RAGService


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Provide a single shared RAGService instance per process."""
    return RAGService()


def get_agent_graph():
    """Return the compiled LangGraph state machine."""
    return app_graph

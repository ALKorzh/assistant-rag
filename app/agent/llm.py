import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_router_llm() -> ChatGoogleGenerativeAI:
    """Deterministic Gemini client used for routing and entity extraction."""
    settings = get_settings()
    logger.info("Initializing router LLM client: model=%s", settings.gemini_model)
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_router_temperature,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatGoogleGenerativeAI:
    """Creative Gemini client used for the final answer synthesis."""
    settings = get_settings()
    logger.info("Initializing answer LLM client: model=%s", settings.gemini_model)
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_answer_temperature,
        google_api_key=settings.google_api_key,
    )


def invoke_router_text(prompt: str) -> str:
    """Run the router LLM on a plain prompt and return stripped text."""
    logger.debug("Invoking router text prompt, length=%d", len(prompt))
    response = get_router_llm().invoke(prompt)
    return str(response.content).strip()

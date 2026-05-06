from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_router_llm() -> ChatGoogleGenerativeAI:
    """Deterministic Gemini client used for routing and entity extraction."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_router_temperature,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatGoogleGenerativeAI:
    """Creative Gemini client used for the final answer synthesis."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_answer_temperature,
        google_api_key=settings.google_api_key,
    )


def invoke_router_text(prompt: str) -> str:
    """Run the router LLM on a plain prompt and return stripped text."""
    response = get_router_llm().invoke(prompt)
    return str(response.content).strip()

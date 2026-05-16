import logging
from functools import lru_cache
from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

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
    # encyclopedic/tool context is often flagged too aggressively; BLOCK_ONLY_HIGH reduces empty replies
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_answer_temperature,
        google_api_key=settings.google_api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )


def _stringify_model_content(content: Any) -> str:
    """
    Достаёт видимый текст из ответа Gemini/LangChain.
    Нельзя делать str(content) для списка блоков — получится repr вида
    "[{'type': 'text', 'text': '...'}]", который улетает в YouTube/DDG как запрос.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (bytes, bytearray)):
        return bytes(content).decode("utf-8", errors="replace").strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                piece = block.get("text")
                if piece is None:
                    piece = block.get("content")
                if piece is not None:
                    parts.append(str(piece))
            else:
                parts.append(_stringify_model_content(block))
        return "".join(parts).strip()
    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content.get("text", "")).strip()
        if "content" in content:
            return _stringify_model_content(content["content"])
        if "parts" in content:
            return _stringify_model_content(content["parts"])
        return ""
    return str(content).strip() if content else ""


def invoke_router_text(prompt: str) -> str:
    """Run the router LLM on a plain prompt and return stripped text."""
    logger.debug("Invoking router text prompt, length=%d", len(prompt))
    response = get_router_llm().invoke(prompt)
    text = _stringify_model_content(getattr(response, "content", None))
    if text:
        return text
    tx = getattr(response, "text", None)
    if isinstance(tx, str) and tx.strip():
        return tx.strip()
    return ""

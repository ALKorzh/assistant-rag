import logging
import re
import unicodedata
from datetime import date
from functools import lru_cache
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.grader import advanced_keyword_check
from app.agent.llm import get_answer_llm, get_router_llm, invoke_router_text
from app.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    CALCULATOR_EXTRACT_PROMPT,
    RAG_CONTEXT_PREFIX,
    REFLECTION_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    WEATHER_EXTRACT_PROMPT,
    WEB_SEARCH_REWRITE_PROMPT,
    WIKIPEDIA_QUERY_PROMPT,
    YOUTUBE_QUERY_PROMPT,
)
from app.agent.state import AgentState, AnswerReflection, RelevanceFlag, RouteResponse
from app.core.config import get_settings
from app.core.retry_utils import call_with_retry
from app.services.rag_service import RAGService
from app.tools.calculator import evaluate_expression
from app.tools.weather import get_weather
from app.tools.wiki_tool import get_wikipedia_info
from app.tools.youtube_tool import search_youtube_videos

logger = logging.getLogger(__name__)

_YOUTUBE_TOOL_PREFIX = "РЕКОМЕНДАЦИИ YOUTUBE:"
_CALCULATOR_TOOL_PREFIX = "РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ:"
_YOUTUBE_LINK_LINE = re.compile(r"Ссылка:\s*(https?://\S+)", re.IGNORECASE)
_YOUTUBE_URL_ANY = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?[^\s)\"]+|youtu\.be/[^\s)\"]+)",
    re.IGNORECASE,
)


def _strip_url_trailing_chars(url: str) -> str:
    u = url.strip()
    while u and u[-1] in ").,;]'\"":  # noqa: RUF001
        u = u[:-1]
    return u


def _is_approved_watch_url(url: str) -> bool:
    """Только ролик watch?v= или youtu.be/... — отсекаем /user, /playlist без v=, /freetutorials."""
    u = _strip_url_trailing_chars(url)
    if not u.startswith("http"):
        return False
    if "…" in u or u.endswith("»"):
        return False
    low = u.lower()
    if "/freetutorials" in low or "youtube.com/@" in low:
        return False
    if re.search(r"youtube\.com/(?:user/|c/|channel/)(?!watch)", low):
        return False
    if "playlist?list=" in low and "watch?v=" not in low and "youtu.be" not in low:
        return False
    if re.search(r"youtu\.be/[\w-]{6,}", u, re.IGNORECASE):
        return True
    if "watch?" in low and re.search(r"[?&]v=[\w-]{6,}", u):
        return True
    return False


def _youtube_urls_from_tool_messages(messages: list[BaseMessage]) -> list[str]:
    """URLs returned by youtube_tool (lines «Ссылка: …») in the latest YouTube block."""
    ordered: list[str] = []
    seen: set[str] = set()
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _message_text(message)
        if not text.startswith(_YOUTUBE_TOOL_PREFIX):
            continue
        for match in _YOUTUBE_LINK_LINE.finditer(text):
            u = _strip_url_trailing_chars(match.group(1))
            if u.startswith("http") and _is_approved_watch_url(u) and u not in seen:
                seen.add(u)
                ordered.append(u)
        if ordered:
            return ordered
        for match in _YOUTUBE_URL_ANY.finditer(text):
            u = _strip_url_trailing_chars(match.group(0))
            if _is_approved_watch_url(u) and u not in seen:
                seen.add(u)
                ordered.append(u)
        return ordered if ordered else []
    return []


def _youtube_only_fallback(urls: list[str]) -> str:
    lines = "\n".join(f"- {u}" for u in urls)
    return (
        "Поиск YouTube вернул такие ролики (ниже — прямые ссылки из выдачи):\n" + lines
    )


def _raw_youtube_tool_message_text(messages: list[BaseMessage]) -> str:
    """Текст последнего сообщения с результатом youtubesearch / DDG."""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        t = _message_text(m)
        if t.startswith(_YOUTUBE_TOOL_PREFIX):
            return t
    return ""


def _deterministic_youtube_answer(state: AgentState) -> str | None:
    """Явный запрос видео на YouTube → ответ только из результата поиска, без LLM «про контекст»."""
    human = _last_human_message_text(state)
    if not _user_intent_youtube_video(human):
        return None
    raw = _raw_youtube_tool_message_text(state["messages"])
    if not raw.strip():
        return None
    urls = _youtube_urls_from_tool_messages(state["messages"])
    if urls:
        return (
            "Ролики с YouTube (результат поиска в приложении, не из загруженных документов):\n\n"
            + "\n".join(f"- {u}" for u in urls)
        )
    lines_out: list[str] = []
    for ln in raw.splitlines():
        if ln.startswith("Инструкция:"):
            continue
        lines_out.append(ln)
    body = "\n".join(lines_out).replace(_YOUTUBE_TOOL_PREFIX, "", 1).strip()
    if not body:
        return "Поиск на YouTube не вернул роликов. Попробуйте позже или уточните запрос."
    return body


def _deterministic_calculator_answer(state: AgentState) -> str | None:
    """
    После calculator_node последнее сообщение — уже готовый результат вычисления.
    Не прогоняем ответ через LLM: пользователю достаточно числа (или текста ошибки парсера).
    """
    msgs = state.get("messages") or []
    if not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, AIMessage):
        return None
    t = _message_text(last)
    if not t.startswith(_CALCULATOR_TOOL_PREFIX):
        return None
    payload = t.split(":", 1)[1].strip() if ":" in t else ""
    return payload if payload else None


def _append_missing_youtube_urls(answer: str, messages: list[BaseMessage]) -> str:
    urls = _youtube_urls_from_tool_messages(messages)
    if not urls:
        return answer
    missing = [u for u in urls if u not in answer]
    if not missing:
        return answer
    logger.info("Appending %d YouTube link(s) missing from synthesized answer", len(missing))
    block = "\n\n**Ссылки на найденные видео:**\n" + "\n".join(f"- {u}" for u in missing)
    return (answer.rstrip() + block).rstrip()


def _strip_lines_with_unapproved_youtube_links(answer: str, approved: list[str]) -> str:
    """Удаляет строки, где есть чужой youtube.com / youtu.be URL, не из списка поиска."""
    if not answer.strip() or not approved:
        return answer
    allow = tuple(approved)
    kept: list[str] = []
    for line in answer.splitlines():
        lower = line.lower()
        if "youtube.com" not in lower and "youtu.be" not in lower:
            kept.append(line)
            continue
        if any(a in line for a in allow):
            kept.append(line)
            continue
        logger.info("Stripping line with YouTube URL not from tool payload: %s", line[:140])
    return "\n".join(kept).strip()


_TOOL_AI_PREFIXES = (
    RAG_CONTEXT_PREFIX,
    "ДАННЫЕ ПОГОДЫ:",
    "СПРАВКА ВИКИПЕДИИ:",
    "РЕКОМЕНДАЦИИ YOUTUBE:",
    "РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ:",
    "Информация из сети:",
)


def _coerce_lc_content(content: Any) -> str:
    """Normalize LangChain / Gemini message content to plain text (handles nested dicts/lists)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, (tuple, set)):
        return "".join(_coerce_lc_content(x) for x in content)
    if isinstance(content, list):
        if not content:
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                piece = block.get("text")
                if piece is None:
                    piece = block.get("content")
                parts.append("" if piece is None else str(piece))
            else:
                parts.append(_coerce_lc_content(block))
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("text") is not None:
            return _coerce_lc_content(content.get("text"))
        if "content" in content:
            return _coerce_lc_content(content["content"])
        if "parts" in content:
            return _coerce_lc_content(content["parts"])
        return ""
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr
    return str(content) if content else ""


def _text_from_llm_message(msg: BaseMessage) -> str:
    """Extract visible model text (Gemini sometimes uses shapes `coerce` alone misses)."""
    if isinstance(msg, AIMessage):
        try:
            tx = getattr(msg, "text", None)
            if isinstance(tx, str) and tx.strip():
                return tx.strip()
        except Exception:
            pass
    text = _coerce_lc_content(msg.content).strip()
    if text:
        return text
    if isinstance(msg, AIMessage):
        try:
            tx = msg.text  # type: ignore[attr-defined]
        except Exception:
            tx = None
        if isinstance(tx, str) and tx.strip():
            return tx.strip()
    return ""


def extract_user_facing_answer(messages: list[BaseMessage]) -> str:
    """Last non-empty assistant reply, skipping tool/RAG payload messages."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _text_from_llm_message(message)
        if not text:
            continue
        if any(text.startswith(p) for p in _TOOL_AI_PREFIXES):
            continue
        return text
    return ""


class WeatherRequest(BaseModel):
    city: str = Field(description="City name in English")
    target_date: str = Field(description="Target date in format YYYY-MM-DD")


@lru_cache(maxsize=1)
def _search_tool() -> DuckDuckGoSearchRun:
    logger.info("Creating DuckDuckGo search tool instance")
    return DuckDuckGoSearchRun()


@lru_cache(maxsize=1)
def _rag_service() -> RAGService:
    logger.info("Creating RAG service instance for agent nodes")
    return RAGService()


def _message_text(message: BaseMessage) -> str:
    return _text_from_llm_message(message)


def _last_message_text(state: AgentState) -> str:
    return _message_text(state["messages"][-1])


def _last_human_message_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return _last_message_text(state)


def _last_rag_context_text(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and _message_text(message).startswith(RAG_CONTEXT_PREFIX):
            return _message_text(message)
    return ""


def _rag_payload_has_hits(rag_message: str) -> bool:
    """True when RAG node embedded real chunks (not empty/error placeholders from RAGService.query)."""
    body = rag_message.strip()
    if not body:
        return False
    if "Информация в локальных документах не найдена" in body:
        return False
    if "Ошибка при поиске в базе документов" in body:
        return False
    return True


def _user_intent_youtube_video(human_text: str) -> bool:
    """Если пользователь явно хочет ролики на YouTube — роутим в youtube без ошибки LLM."""
    if not human_text or not human_text.strip():
        return False
    h = unicodedata.normalize("NFKC", human_text).lower()
    if "youtube" in h or "youtu.be" in h or "ютуб" in h:
        return True
    if re.search(r"найди\s+видео|видео\s+по\b|смотреть\s+видео", h):
        return True
    if "видео" in h and any(
        w in h
        for w in (
            "найди",
            "найти",
            "покажи",
            "дай ",
            " дай",
            "ссылк",
            "ролик",
            "урок",
            "подбор",
            "посмотреть",
            "посоветуй",
            "рекоменд",
            "материал",
        )
    ):
        return True
    if any(p in h for p in ("видеоурок", "видео урок", "видео-урок", "ролик про", "ролики про")):
        return True
    if re.search(r"\bvideos?\b", h) and any(
        x in h for x in ("find", "search", "show", "link", "tutorial", "watch", "lesson", "course")
    ):
        return True
    return False


def router_node(state: AgentState) -> dict[str, str]:
    logger.info("[STEP 1: ROUTER] Intent analysis")
    human = _last_human_message_text(state)
    if _user_intent_youtube_video(human):
        logger.info("Router: forced next_step=youtube (user intent: video / YouTube)")
        return {"next_step": "youtube", "is_relevant": "yes"}
    structured_llm = get_router_llm().with_structured_output(RouteResponse)

    try:
        result = structured_llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=_last_message_text(state)),
            ]
        )
        logger.info("Router decision: next_step=%s reason=%s", result.next_step, result.reason)
        return {"next_step": result.next_step, "is_relevant": "yes"}
    except Exception:
        logger.exception("Router step failed, falling back to direct generation")
        return {"next_step": "direct", "is_relevant": "yes"}


def rag_node(state: AgentState) -> dict[str, list[AIMessage]]:
    query = _last_message_text(state)
    logger.info("[STEP 2: RAG] Searching Qdrant, query_length=%d", len(query))
    context = _rag_service().query(query)
    return {"messages": [AIMessage(content=f"{RAG_CONTEXT_PREFIX} {context}")]}


def relevance_grader_node(state: AgentState) -> dict[str, RelevanceFlag]:
    logger.info("[STEP 3: GRADER] Evaluating RAG relevance")
    rag = _last_rag_context_text(state)
    if not rag.strip():
        logger.info("Grader: no RAG context in message history")
        return {"is_relevant": "no"}

    if _rag_payload_has_hits(rag):
        logger.info(
            "Grader: RAG returned indexed chunks — accepting (skip lexical gate to avoid spurious web fallback)"
        )
        return {"is_relevant": "yes"}

    human = _last_human_message_text(state)
    check = advanced_keyword_check(human, rag)

    if not check["relevant"]:
        logger.info("Grader rejected context: %s", check["reason"])
        return {"is_relevant": "no"}

    logger.info("Grader accepted context: %s", check["reason"])
    return {"is_relevant": "yes"}


def weather_node(state: AgentState) -> dict[str, list[AIMessage]]:
    logger.info("[STEP 2: WEATHER] Fetching weather data")
    today_iso = date.today().isoformat()
    structured_llm = get_router_llm().with_structured_output(WeatherRequest)

    try:
        weather_request = structured_llm.invoke(
            WEATHER_EXTRACT_PROMPT.format(
                message=_last_message_text(state),
                today=today_iso,
            )
        )
        city = weather_request.city.strip()
        target_date = weather_request.target_date.strip()
        logger.info("Weather request parsed: city=%s target_date=%s", city, target_date)
    except Exception:
        logger.exception("Weather request parsing failed, falling back to city-only extraction")
        city = invoke_router_text(
            f"Extract city name in English from this user request. Output only city name: {_last_message_text(state)}"
        )
        target_date = today_iso

    result = get_weather(city=city, target_date=target_date)
    return {"messages": [AIMessage(content=f"ДАННЫЕ ПОГОДЫ: {result}")]}


def wikipedia_node(state: AgentState) -> dict[str, list[AIMessage]]:
    logger.info("[STEP 2: WIKIPEDIA] Fetching encyclopedia data")
    query = invoke_router_text(WIKIPEDIA_QUERY_PROMPT.format(message=_last_message_text(state)))
    info = get_wikipedia_info(query)
    preview = info.replace("\n", " ")[:120]
    logger.info(
        "Wikipedia: query=%r excerpt_chars=%d preview=%s%s",
        query,
        len(info),
        preview,
        "…" if len(info) > 120 else "",
    )
    return {"messages": [AIMessage(content=f"СПРАВКА ВИКИПЕДИИ: {info}")]}


def youtube_node(state: AgentState) -> dict[str, list[AIMessage]]:
    logger.info("[STEP 2: YOUTUBE] Searching videos")
    query = invoke_router_text(YOUTUBE_QUERY_PROMPT.format(message=_last_message_text(state)))
    result = search_youtube_videos(query)
    payload = (
        "РЕКОМЕНДАЦИИ YOUTUBE:\n"
        "Инструкция: в ответе пользователю включи каждую строку «Ссылка: https://…» из блока ниже дословно (можно списком с названием и URL).\n\n"
        f"{result}"
    )
    return {"messages": [AIMessage(content=payload)]}


def calculator_node(state: AgentState) -> dict[str, list[AIMessage]]:
    logger.info("[STEP 2: CALCULATOR] Evaluating expression")
    raw = _last_message_text(state)
    expr = invoke_router_text(CALCULATOR_EXTRACT_PROMPT.format(message=raw)).strip()
    result = evaluate_expression(expr)
    return {"messages": [AIMessage(content=f"РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ: {result}")]}


def rewrite_node(state: AgentState) -> dict[str, list[AIMessage]]:
    logger.info("[STEP 3: REWRITE] Building web-search query")
    query = invoke_router_text(
        WEB_SEARCH_REWRITE_PROMPT.format(message=_last_human_message_text(state))
    )
    return {"messages": [AIMessage(content=query)]}


def search_node(state: AgentState) -> dict[str, list[AIMessage]]:
    query = _last_message_text(state)
    logger.info("[STEP 4: SEARCH] Searching web, query_length=%d", len(query))
    settings = get_settings()

    def _run() -> str:
        return _search_tool().run(query)

    result = call_with_retry(
        _run,
        attempts=max(1, settings.http_retry_attempts),
        base_delay_seconds=settings.http_retry_base_delay_seconds,
        operation_name="DuckDuckGo search",
    )
    return {"messages": [AIMessage(content=f"Информация из сети: {result}")]}


def generate_answer_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    logger.info("[STEP 5: GENERATOR] Synthesizing final answer")
    det_calc = _deterministic_calculator_answer(state)
    if det_calc is not None:
        logger.info("Deterministic calculator reply (Gemini skipped — numeric result only)")
        return {"messages": [AIMessage(content=det_calc)]}

    det = _deterministic_youtube_answer(state)
    if det is not None:
        logger.info("Deterministic YouTube reply (Gemini skipped — user asked for YouTube videos explicitly)")
        return {"messages": [AIMessage(content=det)]}

    is_relevant = state.get("is_relevant", "yes")
    final_messages: list[BaseMessage] = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]

    for message in state["messages"]:
        if (
            isinstance(message, AIMessage)
            and _message_text(message).startswith(RAG_CONTEXT_PREFIX)
            and is_relevant == "no"
        ):
            continue
        final_messages.append(message)

    logger.debug(
        "Sending %d messages to answer model (is_relevant=%s)",
        len(final_messages),
        is_relevant,
    )
    response = get_answer_llm().invoke(final_messages)
    body = _text_from_llm_message(response)
    if not body:
        meta = getattr(response, "response_metadata", None) or {}
        logger.warning(
            "Generator empty output; content_preview=%r finish_reason=%s safety_ratings=%s",
            repr(response.content)[:400],
            meta.get("finish_reason"),
            meta.get("safety_ratings"),
        )
        final_messages.append(
            HumanMessage(
                content=(
                    "Предыдущий ответ получился пустым. Кратко перескажи ответ пользователю на русском "
                    "по его вопросу, используя только данные из сообщений выше (Википедия, документы и т.д.)."
                )
            )
        )
        response = get_answer_llm().invoke(final_messages)
        body = _text_from_llm_message(response)
    if not body:
        logger.warning("Generator still empty; trying low-temperature retry")
        response = get_answer_llm().bind(temperature=0.0).invoke(
            final_messages
            + [
                HumanMessage(
                    content="Сформулируй один короткий ответ на русском по сообщениям выше. "
                    "Только осмысленный текст, нельзя возвращать пустое сообщение."
                )
            ]
        )
        body = _text_from_llm_message(response)

    approved = _youtube_urls_from_tool_messages(state["messages"])
    if not body.strip() and approved:
        logger.warning("Using deterministic YouTube link fallback (model returned no text)")
        body = _youtube_only_fallback(approved)
    elif not body.strip():
        logger.warning("Generator produced no text and no YouTube URLs to fall back to")
        body = (
            "Не удалось сформулировать ответ. Повторите запрос позже или переформулируйте вопрос."
        )

    if approved:
        body = _strip_lines_with_unapproved_youtube_links(body, approved)
        if not body.strip() and approved:
            body = _youtube_only_fallback(approved)

    body = _append_missing_youtube_urls(body, state["messages"])
    return {"messages": [AIMessage(content=body)]}


def reflection_node(state: AgentState) -> dict:
    logger.info("[STEP 6: REFLECTION] Reviewing draft answer")
    retry_count = state.get("reflection_retry_count", 0)
    messages = state["messages"]
    if not messages:
        return {"reflection_branch": "done"}

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {"reflection_branch": "done"}

    draft = _message_text(last)
    if not draft:
        logger.info("Reflection: empty draft, skipping review")
        return {"reflection_branch": "done"}

    human_q = _last_human_message_text(state)
    structured_llm = get_router_llm().with_structured_output(AnswerReflection)

    try:
        verdict = structured_llm.invoke(
            [
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Вопрос пользователя:\n{human_q}\n\nЧерновик ответа:\n{draft}"
                ),
            ]
        )
    except Exception:
        logger.exception("Reflection failed, accepting draft answer")
        return {"reflection_branch": "done"}

    logger.info("Reflection score=%d", verdict.score)
    # Retry only on clearly weak drafts (1–2/5). Score 3+ keeps the first answer — a second pass
    # often regresses (empty output, lost RAG focus) for little gain.
    if verdict.score >= 3 or retry_count >= 1:
        return {"reflection_branch": "done"}

    return {
        "reflection_branch": "retry",
        "reflection_retry_count": retry_count + 1,
        "messages": [
            SystemMessage(
                content=(
                    f"Перепиши свой последний ответ полностью, сохраняя факты из контекста документов. "
                    f"Оценка {verdict.score}/5. Замечания: {verdict.feedback}"
                )
            )
        ],
    }

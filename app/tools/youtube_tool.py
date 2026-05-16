import ast
import logging
import re
from typing import Any

from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch

from app.core.config import get_settings
from app.core.retry_utils import call_with_retry
from app.tools.youtube_data_api import is_valid_youtube_video_url, search_videos

logger = logging.getLogger(__name__)


def _strip_tool_query_noise(raw: str) -> str:
    """
    Защита от str(list/dict) с блоками Gemini: ответ роутера раньше превращался в
    "[{'type': 'text', 'text': 'реальный запрос'}]" и ломал YouTube API.
    """
    t = (raw or "").strip()
    if not t:
        return ""
    if not (t.startswith("[") or t.startswith("{")):
        return t
    if "text" not in t or "type" not in t:
        return t
    try:
        data = ast.literal_eval(t)
    except (ValueError, SyntaxError):
        return t
    if isinstance(data, list):
        texts: list[str] = []
        for item in data:
            if isinstance(item, dict) and item.get("text") is not None:
                texts.append(str(item["text"]).strip())
        if texts:
            return " ".join(texts).strip()
    if isinstance(data, dict) and data.get("text") is not None:
        return str(data["text"]).strip()
    return t


# Развлекательные/общие темы — «tutorial» портит выдачу (котики, приколы, музыка).
_CASUAL_HINT = re.compile(
    r"кот|котик|кошк|щен|собак|песик|прикол|смешн|мил|тикток|танц|музык|"
    r"фильм|трейлер|играть|геймпл|реакци|блог|"
    r"\b(cat|cats|kitten|dog|puppy|funny|cute|meme|music|gameplay)\b",
    re.IGNORECASE,
)


def _is_casual_entertainment_query(q: str) -> bool:
    """Не технический туториал — не заливаем tutorial/course в каждый вариант."""
    s = (q or "").strip()
    if len(s) <= 32 and not re.search(r"langchain|langgraph|python|docker|api\b|tutorial", s, re.I):
        if _CASUAL_HINT.search(s):
            return True
    return bool(_CASUAL_HINT.search(s))


def _search_query_variants(raw: str) -> list[str]:
    """Несколько формулировок: для курсов — tutorial; для «котиков» — видео / funny / EN-синонимы."""
    q = (raw or "").strip()
    if not q:
        return []
    seen: dict[str, None] = {}
    out: list[str] = []

    def push(s: str) -> None:
        t = s.strip()
        if len(t) < 2:
            return
        k = t.casefold()
        if k not in seen:
            seen[k] = None
            out.append(t)

    push(q)
    low = q.casefold()
    casual = _is_casual_entertainment_query(q)

    if casual:
        push(f"{q} видео")
        push(f"{q} youtube")
        push(f"{q} funny")
        if re.search(r"кот|котик|кошк|кис", low):
            push("funny cats kittens")
            push("cute cats youtube")
            push("приколы с котами")
        push(f"{q} compilation")
        return out[:12]

    push(f"{q} tutorial")
    push(f"{q} tutorial beginners")
    push(f"{q} course")
    if "langgraph" in low and "langchain" not in low:
        push(f"LangChain {q} tutorial")
        push(f"{q} LangChain tutorial")
    if "langchain" in low and "tutorial" not in low:
        push(f"{q} tutorial")
    return out[:10]


def search_youtube_videos(query: str, limit: int | None = None) -> str:
    """Search YouTube and return a compact list of relevant videos."""
    clean = _strip_tool_query_noise(query)
    if clean != (query or "").strip():
        logger.info(
            "YouTube query sanitized (removed Gemini block repr): input_len=%d -> %r",
            len(query or ""),
            clean[:160],
        )
    query = clean
    settings = get_settings()
    effective_limit = limit or settings.youtube_results_limit
    variants = _search_query_variants(query)
    if not variants:
        return "Пустой поисковый запрос для YouTube."

    logger.info(
        "YouTube search started base_query=%r variants=%d limit=%d",
        query.strip(),
        len(variants),
        effective_limit,
    )
    primary = settings.youtube_primary_provider
    if primary not in ("youtube", "duckduckgo"):
        primary = "youtube"

    errors: list[str] = []

    def youtube_official_api(q: str) -> str:
        key = settings.youtube_data_api_key
        if not key:
            return ""
        hits = search_videos(
            q,
            api_key=key,
            max_results=effective_limit,
            region_code=settings.youtube_region_code,
            relevance_language=settings.youtube_relevance_language,
            timeout_seconds=float(settings.youtube_data_api_timeout_seconds),
        )
        if not hits:
            return ""
        return "\n\n".join(_format_fallback_video(item) for item in hits)

    def videos_search_lib(q: str) -> str:
        try:
            search = VideosSearch(q, limit=effective_limit)
            results = search.result().get("result", []) or []
        except TypeError as exc:
            # youtubesearchpython + httpx>=0.28: post(..., proxies=...) больше не поддерживается
            if "proxies" in str(exc).lower():
                logger.warning(
                    "youtubesearchpython skipped (httpx proxies incompatibility); q=%r",
                    q[:80],
                )
                return ""
            logger.exception("youtubesearchpython VideosSearch failed for query=%r", q)
            return ""
        except Exception:
            logger.exception("youtubesearchpython VideosSearch failed for query=%r", q)
            return ""
        if not results:
            return ""
        return "\n\n".join(_format_video(video) for video in results)

    def ddg_site(q: str) -> str:
        filtered = _search_youtube_via_duckduckgo(q, effective_limit, broad=False)
        if not filtered:
            return ""
        return "\n\n".join(_format_fallback_video(item) for item in filtered)

    def ddg_broad(q: str) -> str:
        filtered = _search_youtube_via_duckduckgo(q, effective_limit, broad=True)
        if not filtered:
            return ""
        return "\n\n".join(_format_fallback_video(item) for item in filtered)

    strategies: list = []
    if settings.youtube_data_api_key:
        strategies.append(youtube_official_api)
    if primary == "youtube":
        strategies.extend([videos_search_lib, ddg_site, ddg_broad])
    else:
        strategies.extend([ddg_site, videos_search_lib, ddg_broad])

    for variant in variants:
        for fn in strategies:
            name = fn.__name__

            def run_once(q: str = variant, f=fn) -> str:
                return f(q)

            try:
                text = call_with_retry(
                    run_once,
                    attempts=max(1, settings.http_retry_attempts),
                    base_delay_seconds=settings.http_retry_base_delay_seconds,
                    operation_name=f"YouTube {name} q={variant[:48]!r}",
                )
                if text:
                    logger.info("YouTube hit via %s variant=%r", name, variant)
                    return text
            except Exception as exc:
                logger.warning("YouTube %s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")

    logger.error("All YouTube strategies exhausted: %s", "; ".join(errors) or "empty results")
    return (
        "Автоматический поиск на YouTube не вернул ни одного видео по сформулированным запросам "
        "(возможны временные ограничения сети или API). Попробуйте позже или уточните тему "
        "по-русски или ключевыми словами по-английски (например: для курса — "
        "«LangGraph tutorial English»; для развлечения — «funny cats youtube»). "
        "Если включён YouTube Data API, проверьте YOUTUBE_DATA_API_KEY и квоту в Google Cloud Console."
    )


def _search_youtube_via_duckduckgo(query: str, limit: int, *, broad: bool) -> list[dict[str, str]]:
    """DuckDuckGo: перебор нескольких формулировок (особенно для русских «котиков» без tutorial)."""
    settings = get_settings()
    casual = _is_casual_entertainment_query(query)
    max_results = max(limit * 6, 25) if broad else max(limit * 4, 15)

    if broad:
        patterns = [
            f"{query} youtube",
            f"site:youtube.com {query}",
        ]
        if re.search(r"[а-яА-ЯёЁ]", query):
            patterns.append(f"{query} видео site:youtube.com")
        else:
            patterns.append(f"{query} video site:youtube.com")
        if casual:
            patterns.append(f"site:youtube.com {query} funny")
            if re.search(r"кот|кошк|кис", query, re.IGNORECASE):
                patterns.extend(
                    [
                        "cute cats youtube",
                        "funny cats short youtube",
                    ]
                )
    else:
        patterns = [f"site:youtube.com {query}"]

    seen_hrefs: set[str] = set()
    merged: list[dict[str, str]] = []

    for search_query in patterns:

        def _run(sq: str = search_query) -> list[dict]:
            with DDGS(timeout=settings.duckduckgo_timeout_seconds) as ddgs:
                return list(ddgs.text(sq, max_results=max_results))

        try:
            raw_results = call_with_retry(
                _run,
                attempts=max(1, settings.http_retry_attempts),
                base_delay_seconds=settings.http_retry_base_delay_seconds,
                operation_name=f"DuckDuckGo YouTube q={search_query[:60]!r}",
            )
        except Exception:
            logger.exception("DuckDuckGo query failed pattern=%r", search_query)
            continue

        for item in raw_results:
            href = str(item.get("href", ""))
            if not is_valid_youtube_video_url(href):
                continue
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            merged.append(
                {
                    "title": str(item.get("title", "Без названия")),
                    "link": href,
                    "snippet": str(item.get("body", "")),
                }
            )
            if len(merged) >= limit:
                return merged[:limit]

    return merged[:limit]


def _format_video(video: dict[str, Any]) -> str:
    title = video.get("title", "Без названия")
    link = video.get("link", "Ссылка недоступна")
    channel = video.get("channel", {}).get("name", "Неизвестный канал")
    duration = video.get("duration") or "N/A"
    view_count = video.get("viewCount", {}).get("short", "N/A")

    return (
        f"{title}\n"
        f"Канал: {channel} | Длительность: {duration} | Просмотры: {view_count}\n"
        f"Ссылка: {link}"
    )


def _format_fallback_video(video: dict[str, str]) -> str:
    title = video.get("title", "Без названия")
    link = video.get("link", "Ссылка недоступна")
    snippet = video.get("snippet", "")
    if snippet:
        return f"{title}\nОписание: {snippet}\nСсылка: {link}"
    return f"{title}\nСсылка: {link}"

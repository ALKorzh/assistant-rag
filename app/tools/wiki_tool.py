import logging
from typing import Any

import requests

from app.core.config import get_settings
from app.core.retry_utils import call_with_retry

logger = logging.getLogger(__name__)


def _mediawiki_request(lang: str, params: dict[str, str], settings) -> Any:
    """Call MediaWiki Action API with a policy-compliant User-Agent (required by Wikimedia)."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": settings.wikipedia_user_agent}
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=settings.wikipedia_timeout_seconds,
    )
    response.raise_for_status()
    body = response.text.strip()
    if not body:
        raise ValueError("Wikipedia returned an empty body")
    try:
        return response.json()
    except ValueError as exc:
        logger.warning("Wikipedia response is not JSON (first 120 chars): %r", body[:120])
        raise ValueError("Invalid JSON from Wikipedia API") from exc


def _opensearch_title(raw_query: str, lang: str, settings, *, limit: int) -> str | None:
    data = _mediawiki_request(
        lang,
        {
            "action": "opensearch",
            "search": raw_query,
            "limit": str(max(1, limit)),
            "namespace": "0",
            "format": "json",
        },
        settings,
    )
    if isinstance(data, list) and len(data) >= 2:
        titles = data[1]
        if isinstance(titles, list) and titles:
            first = titles[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None


def _fetch_extract(title: str, lang: str, settings) -> str:
    data = _mediawiki_request(
        lang,
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "redirects": "1",
            "titles": title,
        },
        settings,
    )
    pages = data.get("query", {}).get("pages", {})
    for _pid, page in pages.items():
        if not isinstance(page, dict):
            continue
        if page.get("missing"):
            continue
        extract = page.get("extract")
        if isinstance(extract, str) and extract.strip():
            return extract.strip()
    return ""


def get_wikipedia_info(query: str) -> str:
    """Search Wikipedia via the official API (avoids the brittle `wikipedia` PyPI client)."""
    settings = get_settings()
    raw = query.strip()
    if not raw:
        return "Пустой запрос к Википедии."

    lang = settings.wikipedia_language

    def _lookup() -> str:
        title = _opensearch_title(raw, lang, settings, limit=settings.wikipedia_results_limit)
        if not title:
            return ""
        return _fetch_extract(title, lang, settings)

    try:
        text = call_with_retry(
            _lookup,
            attempts=max(1, settings.http_retry_attempts),
            base_delay_seconds=settings.http_retry_base_delay_seconds,
            operation_name="Wikipedia MediaWiki API",
        )
    except Exception:
        logger.exception("Wikipedia query failed")
        return (
            "Не удалось получить статью Википедии (ошибка сети или API). "
            "Попробуйте позже или уточните запрос."
        )

    if not text:
        return f"Ничего не найдено в Википедии по запросу «{raw}»."

    limit = settings.wikipedia_content_limit
    if len(text) > limit:
        cut = text[:limit]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        text = cut + "…"

    return text

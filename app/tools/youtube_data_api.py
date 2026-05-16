"""YouTube Data API v3: поиск видео (search.list) — стабильные watch-ссылки."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# ID видео YouTube — 11 символов (база 64).
_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def normalize_search_query(q: str) -> str:
    """NFKC + обрезка; без потери русского текста."""
    t = unicodedata.normalize("NFKC", (q or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_valid_youtube_video_url(href: str) -> bool:
    """
    Только конкретное видео: watch?v=, youtu.be/<id>, /shorts/<id>.
    Отсекает главную youtube.com, /feed, маркетинговые страницы без videoId.
    """
    raw = (href or "").strip()
    if not raw:
        return False
    try:
        u = urlparse(raw)
    except Exception:
        return False
    host = (u.netloc or "").lower()
    if not host:
        return False

    if host.endswith("youtu.be"):
        part = (u.path or "").strip("/").split("/", 1)[0]
        return bool(part and _VIDEO_ID_RE.fullmatch(part))

    if "youtube.com" not in host and "youtube-nocookie.com" not in host:
        return False

    path = (u.path or "").rstrip("/") or "/"
    if path.startswith("/shorts/"):
        parts = path.split("/")
        vid = parts[2] if len(parts) > 2 else ""
        return bool(vid and _VIDEO_ID_RE.fullmatch(vid))

    if path.startswith("/watch"):
        qs = parse_qs(u.query)
        v = (qs.get("v") or [None])[0]
        return bool(v and _VIDEO_ID_RE.fullmatch(v))

    return False


def _watch_url_from_video_id(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def search_videos(
    query: str,
    *,
    api_key: str,
    max_results: int,
    region_code: str,
    relevance_language: str,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    """
    Возвращает список словарей для _format_fallback_video: title, link, snippet.
    При ошибке API или пустой выдаче — [].
    """
    q = normalize_search_query(query)
    if not q or not api_key:
        return []

    n = max(1, min(int(max_results), 50))
    params: dict[str, Any] = {
        "part": "snippet",
        "type": "video",
        "q": q,
        "key": api_key,
        "maxResults": n,
        "safeSearch": "none",
    }
    if (region_code or "").strip():
        params["regionCode"] = region_code.strip()
    if (relevance_language or "").strip():
        params["relevanceLanguage"] = relevance_language.strip()

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            r = client.get(YOUTUBE_SEARCH_URL, params=params)
    except Exception:
        logger.exception("YouTube Data API request failed q=%r", q[:80])
        return []

    if r.status_code != 200:
        logger.error(
            "YouTube Data API HTTP %s: %s",
            r.status_code,
            (r.text or "")[:800],
        )
        return []

    try:
        payload = r.json()
    except Exception:
        logger.exception("YouTube Data API invalid JSON")
        return []

    items = payload.get("items") or []
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in items:
        vid = _extract_video_id(item)
        if not vid or vid in seen:
            continue
        seen.add(vid)
        sn = item.get("snippet") or {}
        title = str(sn.get("title") or "Без названия")
        channel = str(sn.get("channelTitle") or "")
        desc = (sn.get("description") or "") or ""
        desc_short = desc.strip()
        if len(desc_short) > 400:
            desc_short = desc_short[:397] + "…"
        link = _watch_url_from_video_id(vid)
        snippet_parts: list[str] = []
        if channel:
            snippet_parts.append(f"Канал: {channel}")
        if desc_short:
            snippet_parts.append(desc_short)
        snippet = " | ".join(snippet_parts) if snippet_parts else ""
        out.append({"title": title, "link": link, "snippet": snippet})

    return out


def _extract_video_id(item: dict[str, Any]) -> str | None:
    iid = item.get("id")
    if isinstance(iid, dict):
        vid = iid.get("videoId")
        if isinstance(vid, str) and _VIDEO_ID_RE.fullmatch(vid):
            return vid
    if isinstance(iid, str) and _VIDEO_ID_RE.fullmatch(iid):
        return iid
    return None

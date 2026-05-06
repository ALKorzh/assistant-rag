import logging
from typing import Any

from youtubesearchpython import VideosSearch

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def search_youtube_videos(query: str, limit: int | None = None) -> str:
    """Search YouTube and return a compact list of relevant videos."""
    settings = get_settings()
    effective_limit = limit or settings.youtube_results_limit
    logger.info("YouTube search started, query_length=%d limit=%d", len(query), effective_limit)

    try:
        search = VideosSearch(query, limit=effective_limit)
        results = search.result().get("result", [])

        if not results:
            logger.info("YouTube search returned no results")
            return "Видео по вашему запросу не найдены."

        logger.info("YouTube search completed, results=%d", len(results))
        return "\n\n".join(_format_video(video) for video in results)
    except Exception:
        logger.exception("YouTube search failed")
        return "Ошибка при поиске на YouTube."


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

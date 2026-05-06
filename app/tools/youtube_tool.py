from typing import Any

from youtubesearchpython import VideosSearch


YOUTUBE_RESULTS_LIMIT = 3


def search_youtube_videos(query: str, limit: int = YOUTUBE_RESULTS_LIMIT) -> str:
    """Search YouTube and return a compact list of relevant videos."""
    try:
        search = VideosSearch(query, limit=limit)
        results = search.result().get("result", [])

        if not results:
            return "Видео по вашему запросу не найдены."

        return "\n\n".join(_format_video(video) for video in results)
    except Exception as exc:
        return f"Ошибка при поиске на YouTube: {exc}"


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

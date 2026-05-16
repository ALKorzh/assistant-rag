"""Regression: YouTube tool block yields URLs that can be merged into the final answer."""
from langchain_core.messages import AIMessage

from app.agent import nodes


def test_youtube_urls_extracted_from_tool_message() -> None:
    text = (
        "РЕКОМЕНДАЦИИ YOUTUBE:\n"
        "Intro\n\n"
        "Docker in 5 min\n"
        "Канал: X | Длительность: 5 | Просмотры: 1k\n"
        "Ссылка: https://www.youtube.com/watch?v=testvid01\n"
    )
    msgs = [AIMessage(content=text)]
    urls = nodes._youtube_urls_from_tool_messages(msgs)  # noqa: SLF001
    assert "https://www.youtube.com/watch?v=testvid01" in urls


def test_append_missing_appends_only_if_needed() -> None:
    text = (
        "РЕКОМЕНДАЦИИ YOUTUBE:\nСсылка: https://youtu.be/abc123\n"
    )
    msgs = [AIMessage(content=text)]
    full = "Уже есть https://youtu.be/abc123 в тексте."
    out = nodes._append_missing_youtube_urls(full, msgs)  # noqa: SLF001
    assert out == full


def test_append_missing_adds_block() -> None:
    text = (
        "РЕКОМЕНДАЦИИ YOUTUBE:\nСсылка: https://youtu.be/xyz789\n"
    )
    msgs = [AIMessage(content=text)]
    out = nodes._append_missing_youtube_urls("Общий совет без ссылок.", msgs)  # noqa: SLF001
    assert "https://youtu.be/xyz789" in out
    assert "Ссылки на найденные видео" in out

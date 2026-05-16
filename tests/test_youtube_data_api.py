"""YouTube Data API helper: валидация URL и разбор ответа search.list."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import get_settings
from app.tools.youtube_data_api import is_valid_youtube_video_url, normalize_search_query, search_videos
from app.tools.youtube_tool import search_youtube_videos


@pytest.fixture
def reset_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_normalize_search_query_collapses_spaces() -> None:
    assert normalize_search_query("  a   b  ") == "a b"


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
        ("https://youtu.be/dQw4w9WgXcQ", True),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx", True),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", True),
        ("https://www.youtube.com/", False),
        ("https://www.youtube.com/feed/trending", False),
        ("https://www.youtube.com/youtube", False),
        ("https://www.youtube.com/watch?feature=share", False),
    ],
)
def test_is_valid_youtube_video_url(url: str, ok: bool) -> None:
    assert is_valid_youtube_video_url(url) is ok


@patch("app.tools.youtube_data_api.httpx.Client")
def test_search_videos_parses_api_response(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [
            {
                "id": {"videoId": "dQw4w9WgXcQ", "kind": "youtube#video"},
                "snippet": {
                    "title": "Never Gonna",
                    "channelTitle": "Rick",
                    "description": "Full description",
                },
            }
        ]
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    hits = search_videos(
        "  test query  ",
        api_key="k",
        max_results=5,
        region_code="RU",
        relevance_language="ru",
        timeout_seconds=5.0,
    )
    assert len(hits) == 1
    assert hits[0]["title"] == "Never Gonna"
    assert "dQw4w9WgXcQ" in hits[0]["link"]
    call_kw = mock_client.get.call_args
    assert call_kw[0][0] == "https://www.googleapis.com/youtube/v3/search"
    params = call_kw[1]["params"]
    assert params["q"] == "test query"
    assert params["key"] == "k"
    assert params["regionCode"] == "RU"
    assert params["relevanceLanguage"] == "ru"


def test_search_youtube_videos_uses_official_api_first(
    monkeypatch: pytest.MonkeyPatch,
    reset_settings_cache,
) -> None:
    monkeypatch.setenv("YOUTUBE_DATA_API_KEY", "test-key")

    calls: list[str] = []

    def fake_search_videos(q: str, **kwargs: object) -> list[dict[str, str]]:
        calls.append(q)
        return [
            {
                "title": "API Video",
                "link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "snippet": "Канал: X | desc",
            }
        ]

    monkeypatch.setattr("app.tools.youtube_tool.search_videos", fake_search_videos)

    out = search_youtube_videos("LangGraph tutorial")
    assert "API Video" in out
    assert "watch?v=dQw4w9WgXcQ" in out
    assert calls, "должен вызваться YouTube Data API"
    assert any("LangGraph" in c for c in calls)

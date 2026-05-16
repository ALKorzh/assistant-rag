"""Роутер Gemini: извлечение текста из content=list/dict, не str()."""
from unittest.mock import MagicMock, patch

import pytest

from app.agent.llm import _stringify_model_content, invoke_router_text


def test_stringify_list_of_gemini_blocks() -> None:
    raw = [
        {"type": "text", "text": "LangGraph tutorial for beginners", "extras": {"x": 1}}
    ]
    assert _stringify_model_content(raw) == "LangGraph tutorial for beginners"


def test_stringify_nested_string() -> None:
    assert _stringify_model_content(" plain ") == "plain"


@patch("app.agent.llm.get_router_llm")
def test_invoke_router_text_extracts_block_text(mock_get_router: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [
        {"type": "text", "text": "only the query", "extras": {}},
    ]
    mock_resp.text = None
    mock_llm.invoke.return_value = mock_resp
    mock_get_router.return_value = mock_llm

    out = invoke_router_text("system irrelevant")
    assert out == "only the query"
    mock_llm.invoke.assert_called_once()


@pytest.mark.parametrize(
    ("blob", "want"),
    [
        (
            "[{'type': 'text', 'text': 'hello world', 'extras': {}}]",
            "hello world",
        ),
    ],
)
def test_strip_tool_query_noise(blob: str, want: str) -> None:
    from app.tools.youtube_tool import _strip_tool_query_noise

    assert _strip_tool_query_noise(blob) == want

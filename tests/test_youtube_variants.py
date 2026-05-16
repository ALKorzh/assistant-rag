"""YouTube helper logic (variants for empty-provider fallbacks)."""
from app.tools.youtube_tool import _search_query_variants


def test_variants_include_tutorial_for_langgraph() -> None:
    v = _search_query_variants("LangGraph")
    joined = " ".join(v).lower()
    assert "langgraph" in joined
    assert "tutorial" in joined


def test_variants_langchain_combo() -> None:
    v = _search_query_variants("LangGraph")
    assert any("langchain" in x.lower() for x in v)


def test_variants_casual_no_tutorial_spam() -> None:
    v = _search_query_variants("котики")
    joined = " ".join(v).lower()
    assert "котик" in joined
    assert "tutorial" not in joined
    assert any("funny" in x.lower() or "видео" in x.lower() for x in v)


def test_variants_cats_english_synonyms() -> None:
    v = _search_query_variants("котики")
    flat = " | ".join(v).lower()
    assert "cute cats" in flat or "funny cats" in flat

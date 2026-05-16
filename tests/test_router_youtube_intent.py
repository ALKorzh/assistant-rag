"""Router must force YouTube branch when the user clearly asks for videos."""
from app.agent.nodes import _user_intent_youtube_video


def test_youtube_intent_russian_video_find() -> None:
    assert _user_intent_youtube_video("Найди видео про LangGraph на ютубе для начинающих")


def test_youtube_intent_explicit_platform() -> None:
    assert _user_intent_youtube_video("LangGraph на YouTube tutorial")


def test_youtube_intent_links_and_video() -> None:
    assert _user_intent_youtube_video("Дай ссылки на видео по LangGraph")


def test_youtube_intent_english() -> None:
    assert _user_intent_youtube_video("Find LangGraph tutorial videos for beginners")


def test_exact_user_phrase_langgraph_youtube() -> None:
    assert _user_intent_youtube_video("Найди видео по LangGraph на YouTube")


def test_youtube_intent_plain_question_not_forced() -> None:
    assert not _user_intent_youtube_video("Что такое LangGraph в двух словах?")

"""После ветки calculator финальный ответ — только число или текст ошибки, без обёртки LLM."""

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import _deterministic_calculator_answer


def test_calculator_success_returns_numeric_string_only() -> None:
    state = {
        "messages": [
            HumanMessage(content="Сколько 17*3?"),
            AIMessage(content="РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ: 51"),
        ],
    }
    assert _deterministic_calculator_answer(state) == "51"


def test_calculator_syntax_error_returns_message() -> None:
    state = {
        "messages": [
            HumanMessage(content="x +"),
            AIMessage(content="РЕЗУЛЬТАТ ВЫЧИСЛЕНИЯ: Синтаксическая ошибка в выражении: unmatched"),
        ],
    }
    out = _deterministic_calculator_answer(state)
    assert out is not None
    assert "Синтаксическая ошибка" in out


def test_non_calculator_last_message_returns_none() -> None:
    state = {
        "messages": [HumanMessage(content="Привет"), AIMessage(content="Здравствуйте!")],
    }
    assert _deterministic_calculator_answer(state) is None

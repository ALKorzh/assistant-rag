import pytest

from app.tools.calculator import evaluate_expression


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 2", "4"),
        ("10 / 4", "2.5"),
        ("2^3", "8"),
        ("(1 + 2) * 3", "9"),
    ],
)
def test_evaluate_expression(expr: str, expected: str) -> None:
    assert evaluate_expression(expr) == expected


def test_division_by_zero() -> None:
    assert "ноль" in evaluate_expression("1/0").lower()

from __future__ import annotations

import ast
import operator
from typing import Any

_OPS_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_OPS_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> str:
    """Safely evaluate a numeric expression using a restricted Python AST subset."""
    raw = expression.strip().replace("^", "**")
    if not raw:
        return "Пустое выражение."

    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as exc:
        return f"Синтаксическая ошибка в выражении: {exc}"

    if not isinstance(tree, ast.Expression):
        return "Некорректное выражение."

    try:
        value = _eval_node(tree.body, depth=0)
    except ValueError as exc:
        return str(exc)

    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _eval_node(node: ast.AST, *, depth: int) -> Any:
    if depth > 64:
        raise ValueError("Слишком глубокое выражение.")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Разрешены только числовые константы.")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS_UNARY:
        return _OPS_UNARY[type(node.op)](_eval_node(node.operand, depth=depth + 1))

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS_BIN:
        left = _eval_node(node.left, depth=depth + 1)
        right = _eval_node(node.right, depth=depth + 1)
        if type(node.op) is ast.Div and right == 0:
            raise ValueError("Деление на ноль.")
        return _OPS_BIN[type(node.op)](left, right)

    raise ValueError("В выражении есть неподдерживаемые операции.")

"""Herramienta de cálculo: evalúa aritmética sin usar eval, recorriendo el árbol sintáctico."""
from __future__ import annotations

import ast
import operator

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max}
MAX_EXPONENT = 100


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponente demasiado grande (máximo {MAX_EXPONENT})")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("expresión no permitida: usa solo números, + - * / // % ** y paréntesis")


def format_number(value: float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def calculate(expression: str) -> str:
    """Devuelve el resultado como texto. Lanza ValueError si la expresión no es aritmética válida."""
    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("$", "")
    if "=" in expr:
        expr = expr.split("=", 1)[0].strip()
    if not expr:
        raise ValueError("expresión vacía")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"sintaxis inválida: {e.msg}") from e
    try:
        return format_number(_eval(tree.body))
    except ZeroDivisionError as e:
        raise ValueError("división entre cero") from e
    except OverflowError as e:
        raise ValueError("resultado fuera de rango") from e

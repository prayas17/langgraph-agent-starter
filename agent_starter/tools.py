"""Example tools. Swap in your real ones — HTTP APIs, DB queries, whatever.

Tools are the seams where agents touch the world. Keep each one narrow
and pure (same input → same output), so agents can reason about them.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports + - * / % ** and parentheses.

    Args:
        expression: a Python-style math expression, e.g. "2 * (3 + 4)".
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
    except Exception as e:  # broad on purpose — feed the error back to the LLM
        return f"Error: {e}"
    return str(result)


@tool
def current_datetime() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.utcnow().isoformat() + "Z"


@tool
def web_search_stub(query: str) -> str:
    """Placeholder for a real web-search tool.

    Replace this with a Tavily / SerpAPI / Brave Search integration in
    production. Returning canned data keeps the graph runnable in tests
    and demos without any API keys.

    Args:
        query: what the agent wants to search for.
    """
    canned = {
        "python": "Python is a high-level, general-purpose programming language.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor LLM applications.",
        "rag": "RAG (Retrieval-Augmented Generation) grounds LLM outputs in external documents.",
    }
    key = query.lower().strip()
    for k, v in canned.items():
        if k in key:
            return v
    return f"[stub search] No canned result for: {query!r}. Replace web_search_stub with a real tool."


DEFAULT_TOOLS = [calculator, current_datetime, web_search_stub]


def _safe_eval(node: ast.AST) -> float:
    """Whitelist AST evaluator — no builtins, no attribute access, no calls."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"Unsupported node: {type(node).__name__}")

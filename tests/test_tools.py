"""Unit tests for the calculator tool — no LLM required."""

import pytest

from agent_starter.tools import calculator, current_datetime, web_search_stub


def test_calculator_basic() -> None:
    assert calculator.invoke({"expression": "2 + 2"}) == "4"
    assert calculator.invoke({"expression": "10 * (3 + 5)"}) == "80"
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"
    assert calculator.invoke({"expression": "-5 + 3"}) == "-2"


def test_calculator_rejects_unsafe() -> None:
    # No function calls, no attribute access, no imports
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert result.startswith("Error")

    result = calculator.invoke({"expression": "open('/etc/passwd')"})
    assert result.startswith("Error")


def test_calculator_malformed() -> None:
    assert calculator.invoke({"expression": "not an expression"}).startswith("Error")
    assert calculator.invoke({"expression": ""}).startswith("Error")


def test_current_datetime_iso8601() -> None:
    result = current_datetime.invoke({})
    # Roughly: 2026-07-18T07:30:00.000000Z
    assert "T" in result
    assert result.endswith("Z")


def test_web_search_stub_returns_canned() -> None:
    result = web_search_stub.invoke({"query": "what is langgraph"})
    assert "LangGraph" in result

    result = web_search_stub.invoke({"query": "unrelated topic"})
    assert "stub search" in result


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1 + 2 * 3", "7"),
        ("(1 + 2) * 3", "9"),
        ("100 / 4", "25.0"),
        ("17 % 5", "2"),
    ],
)
def test_calculator_precedence(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected

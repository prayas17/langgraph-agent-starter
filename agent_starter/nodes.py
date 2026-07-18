"""Agent nodes — each is a pure function `state -> partial state update`.

Nodes should be small, testable, and free of framework glue where possible.
The graph in `graph.py` wires them together.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent_starter.state import AgentState

_ROUTER_SYSTEM = """You are a router. Read the user's latest message and pick exactly one label:

- "researcher"  — needs external information, facts, or web lookup
- "calculator"  — needs arithmetic or math
- "responder"   — can be answered directly with common sense or from context

Return JSON: {"route": "<label>"}"""

_RESEARCHER_SYSTEM = """You are a research assistant. Use the available tools to
gather information, then reply concisely with cited findings."""

_RESPONDER_SYSTEM = """You are a helpful assistant. Answer directly and concisely.
If you're not confident, say so."""


def make_router_node(llm: BaseChatModel):
    """A router node reads the latest user message and returns a route."""

    def router(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=_ROUTER_SYSTEM), *state["messages"]]
        raw = llm.invoke(messages).content
        try:
            data = json.loads(raw if isinstance(raw, str) else str(raw))
            route = str(data.get("route", "responder")).strip().lower()
        except (json.JSONDecodeError, TypeError):
            route = "responder"
        if route not in {"researcher", "calculator", "responder"}:
            route = "responder"
        return {"route": route, "iterations": state.get("iterations", 0) + 1}

    return router


def make_researcher_node(llm: BaseChatModel, tools: list[BaseTool]):
    """Researcher issues tool calls until it has enough, then answers."""
    llm_with_tools = llm.bind_tools(tools)

    def researcher(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=_RESEARCHER_SYSTEM), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return researcher


def make_tool_node(tools: list[BaseTool]):
    """Execute any tool calls in the last message. Returns ToolMessages."""
    by_name = {t.name: t for t in tools}

    def tool_executor(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        results: list[ToolMessage] = []
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                results.append(
                    ToolMessage(
                        content=f"Unknown tool: {call['name']}",
                        tool_call_id=call["id"],
                    )
                )
                continue
            try:
                out = tool.invoke(call["args"])
            except Exception as e:  # feed the error back so the agent can recover
                out = f"Tool error: {e}"
            results.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
        return {"messages": results}

    return tool_executor


def make_responder_node(llm: BaseChatModel):
    """Terminal node: produce a user-facing answer."""

    def responder(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=_RESPONDER_SYSTEM), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return responder


def make_calculator_node(llm: BaseChatModel, calculator_tool: BaseTool):
    """A specialized node that always uses the calculator tool once."""
    llm_with_calc = llm.bind_tools([calculator_tool], tool_choice="required")

    def calculator_agent(state: AgentState) -> dict[str, Any]:
        # Ask the LLM to formulate the expression as a tool call
        messages = [
            SystemMessage(content="Compute the requested value using the calculator tool."),
            *state["messages"],
        ]
        return {"messages": [llm_with_calc.invoke(messages)]}

    return calculator_agent


# ---------- helpers ----------


def has_tool_calls(state: AgentState) -> bool:
    """Router predicate: does the last message have pending tool calls?"""
    if not state["messages"]:
        return False
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def make_user_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)

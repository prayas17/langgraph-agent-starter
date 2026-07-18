"""Graph wiring. Compose nodes into a StateGraph and compile.

Shape:

    START -> router -> {researcher, calculator, responder}
                         │             │
                         ▼             ▼
                       tools        (direct)
                         │
                         └─── loop back to researcher until no more tool calls
                                             │
                                             ▼
                                         responder -> END
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent_starter.nodes import (
    has_tool_calls,
    make_calculator_node,
    make_researcher_node,
    make_responder_node,
    make_router_node,
    make_tool_node,
)
from agent_starter.state import AgentState
from agent_starter.tools import DEFAULT_TOOLS, calculator

MAX_ITERATIONS = 8


def build_graph(
    llm: BaseChatModel,
    tools: list | None = None,
    *,
    checkpointer: MemorySaver | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Callable:
    """Build and compile the multi-agent graph.

    Args:
        llm: any LangChain-compatible chat model (OpenAI, Anthropic, ...).
        tools: tool set for the researcher. Defaults to DEFAULT_TOOLS.
        checkpointer: optional persistence layer (MemorySaver, SqliteSaver, ...).
        max_iterations: hard cap on graph traversals — protects against loops.

    Returns:
        A compiled graph. Call `.invoke(state, config)` or `.stream(...)`.
    """
    tools = tools or DEFAULT_TOOLS

    graph = StateGraph(AgentState)

    graph.add_node("router", make_router_node(llm))
    graph.add_node("researcher", make_researcher_node(llm, tools))
    graph.add_node("tools", make_tool_node(tools))
    graph.add_node("calculator", make_calculator_node(llm, calculator))
    graph.add_node("responder", make_responder_node(llm))

    graph.add_edge(START, "router")

    def _route_from_router(state: AgentState) -> str:
        if state.get("iterations", 0) > max_iterations:
            return "responder"
        route = state.get("route", "responder")
        return route if route in {"researcher", "calculator", "responder"} else "responder"

    graph.add_conditional_edges(
        "router",
        _route_from_router,
        {"researcher": "researcher", "calculator": "calculator", "responder": "responder"},
    )

    # Researcher may want tools. If yes, run them and loop; if no, hand to responder.
    graph.add_conditional_edges(
        "researcher",
        lambda s: "tools" if has_tool_calls(s) else "responder",
        {"tools": "tools", "responder": "responder"},
    )
    graph.add_edge("tools", "researcher")

    # Calculator always uses its one tool then hands off
    graph.add_conditional_edges(
        "calculator",
        lambda s: "tools" if has_tool_calls(s) else "responder",
        {"tools": "tools", "responder": "responder"},
    )

    graph.add_edge("responder", END)

    return graph.compile(checkpointer=checkpointer)

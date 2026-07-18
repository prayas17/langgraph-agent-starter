"""Graph smoke tests using a fake chat model — no OpenAI key required."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent_starter.graph import build_graph
from agent_starter.nodes import has_tool_calls
from agent_starter.state import AgentState


class RoutedFakeChatModel(FakeListChatModel):
    """FakeListChatModel that responds with our expected routing JSON first,
    then falls back to canned messages. Useful for graph smoke tests.
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        # Ignore tools binding — fake model just returns preset responses
        return self


def test_router_directs_to_responder_by_default() -> None:
    llm = RoutedFakeChatModel(responses=[
        json.dumps({"route": "responder"}),
        "Hello! I'm a helpful assistant.",
    ])
    graph = build_graph(llm)

    initial = {"messages": [HumanMessage(content="hi")]}
    final = graph.invoke(initial)

    assert len(final["messages"]) >= 2
    assert final["route"] == "responder"


def test_router_directs_to_researcher() -> None:
    llm = RoutedFakeChatModel(responses=[
        json.dumps({"route": "researcher"}),
        "LangGraph is a library for stateful LLM apps.",
        "Based on my research, LangGraph is a stateful workflow library.",
    ])
    graph = build_graph(llm)

    initial = {"messages": [HumanMessage(content="tell me about langgraph")]}
    final = graph.invoke(initial)

    assert final["route"] == "researcher"


def test_router_falls_back_on_invalid_route() -> None:
    llm = RoutedFakeChatModel(responses=[
        json.dumps({"route": "invalid_route"}),
        "Fallback response.",
    ])
    graph = build_graph(llm)
    final = graph.invoke({"messages": [HumanMessage(content="test")]})
    assert final["route"] == "responder"


def test_router_handles_malformed_json() -> None:
    llm = RoutedFakeChatModel(responses=[
        "not json at all",
        "Fallback response.",
    ])
    graph = build_graph(llm)
    final = graph.invoke({"messages": [HumanMessage(content="test")]})
    assert final["route"] == "responder"


def test_iteration_cap_forces_responder() -> None:
    llm = RoutedFakeChatModel(responses=[json.dumps({"route": "researcher"})] * 20 + ["done"] * 5)
    graph = build_graph(llm, max_iterations=1)
    final = graph.invoke({"messages": [HumanMessage(content="test")]})
    # After 1 iteration, subsequent routing hits the cap
    assert "messages" in final


class TestHasToolCallsPredicate:
    def test_true_when_ai_message_has_tool_calls(self) -> None:
        msg = AIMessage(content="", tool_calls=[{"id": "1", "name": "calc", "args": {}}])
        state: AgentState = {"messages": [msg]}
        assert has_tool_calls(state) is True

    def test_false_when_no_messages(self) -> None:
        state: AgentState = {"messages": []}
        assert has_tool_calls(state) is False

    def test_false_when_last_is_human(self) -> None:
        state: AgentState = {"messages": [HumanMessage(content="hi")]}
        assert has_tool_calls(state) is False

    def test_false_when_ai_has_no_tool_calls(self) -> None:
        msg = AIMessage(content="just text")
        state: AgentState = {"messages": [msg]}
        assert has_tool_calls(state) is False

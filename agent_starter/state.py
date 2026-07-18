"""Shared state — the single source of truth passed between nodes.

Keep this small. Every field here gets serialized on every checkpoint;
bloat kills performance.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State shared across all agent nodes.

    - `messages` uses LangGraph's `add_messages` reducer, so returning a
      partial list from a node *appends* rather than replacing.
    - `route` is the last routing decision (used by the supervisor).
    - `scratchpad` is a place for intermediate agent notes — keep small.
    - `iterations` guards against runaway loops.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    scratchpad: dict[str, str]
    iterations: int

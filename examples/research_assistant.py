"""Research-assistant example.

The router sends "look this up" queries to the researcher, which uses the
stub web-search tool. Replace `web_search_stub` in `agent_starter.tools`
with a real search integration for production.

Prereqs:
    pip install -e ".[dev]"
    export OPENAI_API_KEY=sk-...

    python examples/research_assistant.py
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent_starter.graph import build_graph
from agent_starter.nodes import make_user_message


def main() -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Copy .env.example -> .env and fill it in.")
        return 1

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    graph = build_graph(llm, checkpointer=MemorySaver())

    queries = [
        "What is LangGraph?",
        "What is 42 * 17 + 100?",
        "Say hi.",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        config = {"configurable": {"thread_id": f"demo-{hash(q)}"}}
        final = graph.invoke({"messages": [make_user_message(q)]}, config=config)
        print(f"Route: {final.get('route')}")
        print(f"Assistant: {final['messages'][-1].content}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Small CLI to drive the graph from a terminal.

    agent "What is 27 * 43?"
    agent "Tell me about LangGraph"
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent_starter.graph import build_graph
from agent_starter.nodes import make_user_message


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the agent starter graph on one query.")
    parser.add_argument("query", nargs="+", help="User query")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("AGENT_TEMPERATURE", "0.2")))
    parser.add_argument("--stream", action="store_true", help="Stream events as they happen")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Copy .env.example -> .env and fill it in.", file=sys.stderr)
        return 1

    llm = ChatOpenAI(model=args.model, temperature=args.temperature)
    graph = build_graph(llm, checkpointer=MemorySaver())

    query = " ".join(args.query)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = {"messages": [make_user_message(query)]}

    if args.stream:
        for event in graph.stream(initial, config=config):
            for node_name, update in event.items():
                if messages := update.get("messages"):
                    last = messages[-1]
                    print(f"[{node_name}] {last.content}")
    else:
        final = graph.invoke(initial, config=config)
        last = final["messages"][-1]
        print(last.content)

    return 0


if __name__ == "__main__":
    sys.exit(main())

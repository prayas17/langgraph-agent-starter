<h1 align="center">langgraph-agent-starter</h1>

<p align="center">
  <b>Clone-and-run template for building multi-agent LLM systems with LangGraph.</b><br/>
  Router + specialist agents · tool calling · streaming · memory · guards against runaway loops.
</p>

<p align="center">
  <a href="https://github.com/prayas17/langgraph-agent-starter/actions/workflows/tests.yml"><img src="https://github.com/prayas17/langgraph-agent-starter/actions/workflows/tests.yml/badge.svg" alt="tests"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-blue"/>
</p>

---

## What this gives you

Most LangGraph tutorials stop at a two-node "hello world". This template ships the shape you actually want in production:

- **Router → specialists** pattern (supervisor picks which agent handles each turn)
- **Tool loop** with graceful error recovery (tool errors go back to the LLM as messages)
- **Hard iteration cap** so the graph can never loop forever
- **Explicit shared state** with `add_messages` reducer (partial updates append)
- **Checkpointer-ready** (swap `MemorySaver` for `SqliteSaver` for persistence)
- **Fake-model tests** — CI runs without touching any LLM API
- **Small CLI** so you can drive it from the terminal in one line

Total: ~500 lines across `state`, `tools`, `nodes`, `graph`, `cli`. Read the whole thing in an hour.

## Install & run

```bash
git clone https://github.com/prayas17/langgraph-agent-starter.git
cd langgraph-agent-starter
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env                                  # fill in OPENAI_API_KEY

# ask a question
agent "What is 27 * 43 + 100?"
agent "Tell me about LangGraph" --stream
```

Or as a library:

```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from agent_starter import build_graph
from agent_starter.nodes import make_user_message

llm = ChatOpenAI(model="gpt-4o-mini")
graph = build_graph(llm, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(
    {"messages": [make_user_message("What is LangGraph?")]},
    config=config,
)
print(result["messages"][-1].content)
```

## Graph shape

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         ▼
                    ┌─────────┐
                    │ router  │──── picks route via JSON
                    └────┬────┘
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        researcher   calculator   responder
              │          │          │
              ▼          ▼          │
            tools     tools         │
              │          │          │
              └──┐    ┌──┘          │
                 ▼    ▼             │
              researcher/calc       │
                 │                  │
                 └───────► responder ──── END
```

Each node is a **pure function** `state -> partial update`. That means you can:

- Test each node in isolation without a graph
- Swap the LLM without rewriting anything
- Replace `researcher` with your own agent that follows the same signature

## Extending it

**Add a new specialist.** Add a `foo` label in `_ROUTER_SYSTEM`, add a `make_foo_node()` in `nodes.py`, add a node + edge in `graph.py`.

**Add a real tool.** Replace `web_search_stub` with a Tavily/SerpAPI wrapper (return a plain string; that's the whole contract).

**Persist conversations.** Swap `MemorySaver()` for `SqliteSaver.from_conn_string("checkpoints.sqlite")`. Same interface, real durability.

**Change the model.** Swap `ChatOpenAI` for `ChatAnthropic`, `ChatVertexAI`, etc. — any LangChain-compatible chat model works.

**Stream events.** Use `graph.stream(...)` instead of `.invoke(...)` — each node's update becomes an event.

## Design decisions

**Why a router node instead of the LLM picking a tool?** Cleaner reasoning trace, easier to swap models per-branch, and cheaper (router uses a small model, specialists use the big one).

**Why does the researcher loop with tools?** Real research is iterative — one search, refine query, another search. Forcing a single tool call cripples the agent.

**Why cap iterations?** Because I've seen a supervisor + researcher + tool loop chew through $80 of API calls in 12 minutes when a bug loops the graph.

**Why fake-model tests?** So CI runs in seconds without an API key. Real LLM output is non-deterministic anyway — test the *plumbing*, not the responses.

## Roadmap

- [ ] Anthropic model example
- [ ] Sqlite checkpointer example (persistent conversations)
- [ ] Human-in-the-loop interrupt example
- [ ] Streaming to a websocket endpoint
- [ ] Retrieval-aware researcher (integrates with rag-toolkit)

## License

MIT — see [LICENSE](./LICENSE).

## Author

Built by [Prayas Jain](https://github.com/prayas17) — Forward Deployed AI Engineer.
Open to freelance & contract work on agentic AI systems: [prayas1711@gmail.com](mailto:prayas1711@gmail.com) · [Portfolio](https://prayas17.github.io/portfolio)

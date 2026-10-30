# Retrieval tester

This folder exists so you can see what the graph gives an agent. It loads stored Context for one user and chats using only that. It never writes to Neo4j or MongoDB, and it never calls ingestion.

The browser page and the terminal chat live in `ui/`. They only render. Retrieval, the agent, and saving the chat are here.

```bash
uv run python -m streamlit run ui/streamlit_app.py
uv run python -m ui.chat_with_agent --user U-praveen
```

Each chat is saved under `data/agent_sessions/` as JSON in the ingestion input shape, plus a `retrieval_tester` section that records the model, the Context that was retrieved, and the retrieval diagnostics. Ingestion ignores that extra section. Loading a saved chat into the graph is a separate step you take yourself:

```bash
uv run context-graph --input data/agent_sessions/<file>.json
```

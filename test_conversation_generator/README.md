# Test conversation generator

This folder exists to build conversations that exercise the ingestion pipeline. It is not part of a run. `context-graph` never imports it.

Each entry in `scenarios.py` is one behaviour the pipeline must handle: a lasting preference, a value that later changes, a revocation, a message too long for one episode, acknowledgements the relevance gate should skip, and similar cases. `ConversationDatasetGenerator` asks the chat model for the dialogue, then assigns session ids, message ids, and timestamps itself, so the same seed produces the same file. Every session is checked against `IngestionRequest` before it is written.

The default output path is `data/synthetic_sessions.json`. Load it the same way as any other sessions file:

```bash
uv run context-graph --input data/synthetic_sessions.json
```

Generation settings use the `SYNTH_` prefix (`SYNTH_SEED`, `SYNTH_OUTPUT_PATH`, `SYNTH_USER_COUNT`). The chat model is the one already configured for the pipeline.

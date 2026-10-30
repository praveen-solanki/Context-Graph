# ConvoMem benchmark converter

This folder turns the raw ConvoMem benchmark into session files the pipeline can ingest. It is not part of a run. `context-graph` never imports it.

ConvoMem stores evidence as nested JSON under `data/convomem/`. That shape is not an ingestion request. `prepare.py` picks a small demo set and writes each session in the shape `context-graph` already accepts.

The demo set is users who appear in all four evidence folders, the top 5 of those users, and the 3 longest sessions per user in each folder. That is 60 sessions. Output goes to `data/convomem_demo/`, and the combined file is `data/convomem_demo/sessions.json`, which is the default `--input`.

```bash
uv run python -m convomem_benchmark_converter.prepare
uv run context-graph
```

# Data

Input files for the Context Graph pipeline. Everything here is read by a command; nothing in this folder is code. Only this README and `org_structure.json` are tracked by git, because the datasets are large and can be rebuilt or downloaded.

| Path | What it is | Used by |
| --- | --- | --- |
| `convomem/` | The raw ConvoMem benchmark, about 830 MB | `convomem_benchmark_converter` only |
| `convomem_demo/` | 60 sessions selected from ConvoMem, already in the ingestion input shape | `context-graph` (default input) |
| `org_structure.json` | Organization, departments, teams, people and projects | `context-graph --org-structure` |

## `convomem/` - raw benchmark

ConvoMem is a benchmark of long user-assistant conversations, each written so that a later question can only be answered by remembering something said earlier. It is the source the demo set is built from. The pipeline cannot read it directly, because it is not in the ingestion input shape.

There is one folder per kind of memory the benchmark tests:

| Folder | What the conversations test |
| --- | --- |
| `preference_evidence/` | Preferences the user states, such as how they like things done |
| `changing_evidence/` | Facts that change during or between conversations, where the latest value must win |
| `user_evidence/` | Facts the user states about themselves, their work or their plans |
| `implicit_connection_evidence/` | Information that only answers a question when connected to something said elsewhere |

Inside each one, `N_evidence/` groups items by how many messages hold the evidence (1 to 6). Each file is one person, named `<person_id>_<Job_Title>.json`. It holds `evidence_items`, and each item has the benchmark `question`, the expected `answer`, the `message_evidences` that support it, and the full `conversations`.

## `convomem_demo/` - prepared demo set

Built from `convomem/` by:

```bash
uv run python -m convomem_benchmark_converter.prepare
```

The selection is the 5 users present in all four folders with the most long sessions, and for each of them the 3 longest sessions in each folder: 5 users x 4 folders x 3 sessions = 60 sessions. Each message gets a deterministic id and timestamp, users become `U-<first 8 characters of person_id>`, and the assistant becomes `agent-1`.

| File | Contents |
| --- | --- |
| `sessions.json` | All 60 sessions in one file. This is the default `CG_INPUT`, so `uv run context-graph` loads it |
| `<folder>/sessions.json` | The same sessions split by benchmark folder, for loading one kind of memory at a time |
| `manifest.json` | One row per session: `folder`, `person_id`, `user_id`, `job_title`, `session_id`, `message_count`, and the `median_threshold` used to select it |

Load a single folder with:

```bash
uv run context-graph --input data/convomem_demo/changing_evidence/sessions.json
```

## `org_structure.json` - organization chart

Who is on which team and which projects belong to which team. Conversation can never create these relationships, so this file is their only source. It is applied only when you ask for it:

```bash
uv run context-graph --org-structure data/org_structure.json
```

The file currently contains an empty template. Applying a file makes the graph's org relationships match it exactly, so an empty file removes every existing team, department, project and assignment link. Fill it in before applying it to a graph that already has an org structure.

## Written here by other tools

These do not exist until the tool that writes them runs:

| Path | Written by |
| --- | --- |
| `synthetic_sessions.json` | `test_conversation_generator` (generated test conversations) |
| `agent_sessions/` | `retrieval_tester` (saved chats, which can be loaded with `--input`) |

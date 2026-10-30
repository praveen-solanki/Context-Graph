# Context Graph — Ingestion & Personalization Layer (POC)

Implements the Context Graph ingestion specification: conversations become
immutable Episodes in Neo4j, and durable personalization Context is derived from
them with explicit provenance, scope and write-time governance. Request-time
scope applicability and authorization belong to the separate retrieval design
and are deliberately **not** implemented here.

## Layout

| Module | Responsibility |
| --- | --- |
| `config.py` | Single source of settings, budgets, thresholds and promotion policy |
| `domain/` | Models and controlled vocabulary: enums, conversation (Episode, SourceSpan), extraction envelope, context |
| `graph/` | Normative Neo4j schema, property mapping, idempotent graph store |
| `ledger/` | MongoDB run record, session checkpoints, extraction batch results |
| `providers/` | Protocols plus the OpenAI-compatible chat and embedding client |
| `pipeline/` | Gate, batching, rolling state, extraction, resolution, policy, writer, promotion, orchestration |
| `prompts.py` | All prompt text and its version |
| `tool.py` | `ingest_conversation` — the single public entry point |

Modules above live in `src/context_graph/`. Around the pipeline:

| Folder | Responsibility |
| --- | --- |
| `convomem_benchmark_converter/` | Converts the raw ConvoMem benchmark into session files. See its README |
| `test_conversation_generator/` | Builds conversations that exercise pipeline behaviour. See its README |
| `retrieval_tester/` | Checks what the graph returns to an agent, without writing. See its README |
| `ui/` | Browser and terminal screens for the retrieval tester |
| `tests/` | `unit/`, `e2e/`, `integration/`, `ui/` suites plus shared factories and doubles |

## Pipeline

```
conversation input
  → episode segmentation (strategy-selectable: see Episodes below)
  → Episode persistence (Neo4j, immutable, idempotent on session_id+episode_id)
  → cheap relevance gate
  → MongoDB checkpoint → only unprocessed / incomplete Episodes
  → token-based batching (+ overlap and temporary rolling session state)
  → LLM extraction → schema validation → validated result stored in MongoDB
  → embedding + concept resolution (deterministic → embedding → adjudication)
  → writer policy → canonical Context + evidence + provenance + lineage
  → commit batch → advance checkpoint → asynchronous promotion pass
```

Episode persistence is never blocked by extraction; a failed batch keeps its
Episodes eligible for retry and stops the checkpoint at the gap.

## Episodes

An Episode is a **bounded group of related conversation content**, not a single
message. `CG_SEGMENTATION__STRATEGY` picks how a session is grouped:

- `llm_boundary` (default) - token-bounded windows where the model proposes
  topic boundaries over ordered source references, and
  [`pipeline/segmentation.py`](src/context_graph/pipeline/segmentation.py)
  reconstructs episodes from the original records. The model supplies
  boundaries only, never text, identity, timestamps or metadata.
- `similarity` - character-size windows split where embedding similarity dips,
  with no LLM calls. Implemented in
  [`pipeline/similarity_segmentation.py`](src/context_graph/pipeline/similarity_segmentation.py)
  and tuned via `CG_SEGMENTATION__SIMILARITY__*` (see `.env.example`).

Each episode holds `SourceSpan`s: an exact slice of one message with its
character offsets, speaker, timestamp and original metadata. A short reply stays
with the discussion it answers; a message too large for one episode is split
deterministically into bounded spans so a boundary can fall inside a message
covering two topics.

Both strategies are enforced by code, not trusted from the model or the
heuristic: every span belongs to exactly one finalized episode, spans stay in
order with exact text and offsets, episodes respect a hard size limit and
continue into further episodes when a discussion outgrows it, and the
session's remaining content is finalized when the input ends. Content is never
discarded. `llm_boundary` additionally carries a window's unfinished tail into
the next window with bounded read-only context, and falls back to
deterministic size-based grouping on an invalid proposal, a truncated
response or a provider failure.

Spans live as one JSON property on the Episode node, so grouping adds no node
per message. Episodes written before this change carry no spans and are read
back as a single whole-message span, so existing graphs keep working. Downstream
code is unchanged: `Episode.text`, `.role`, `.sender_id`, `.timestamp` and
`.source_type` are derived from the first span, while every span keeps its own
speaker identity.

`llm_boundary` grouping is model-assisted and not bit-stable across runs
(`similarity` grouping is deterministic given the same embeddings), so either
way each run's segmentation is stored in the ledger and reused on resume;
episode identity is otherwise derived deterministically from the spans it
covers.

## Guarantees

- **Idempotent**: Episodes MERGE on `(session_id, episode_id)`, concepts on
  `concept_key`, context on a deterministic fingerprint, evidence on its source
  Episode. Retries never duplicate nodes or inflate evidence counts.
- **Append-only values**: the same value refreshes evidence; a changed value
  creates a new instance that `SUPERSEDES` the old one at the same scope.
- **Governed shared scope**: conversation can create private context, but
  Team/Department/Organization signals are stored as `candidate` and only an
  asynchronous, type- and scope-aware promotion pass can activate them.
- **No self-granted access**: the LLM may detect that a statement concerns a
  Team or Organization, but authorization edges are never written from
  conversation, and signals about unknown subjects or unidentifiable scopes stay
  unresolved rather than creating an incorrect entity.
- **Rebuildable**: derived context carries run, extractor, prompt, policy and
  embedding-model lineage, so it can be rebuilt from Episodes.

## Setup

```bash
uv sync                     # add --system-certs behind a TLS-inspecting proxy
cp .env.example .env        # fill in Neo4j, MongoDB and NVIDIA_API_KEY
uv run context-graph
```

`context-graph` is the only way to load data: one command, two settings, each
overridable from the command line or from `.env`.

| Setting | CLI flag | `.env` | Default |
| --- | --- | --- | --- |
| Which conversations to load | `--input` | `CG_INPUT` | `data/convomem_demo/sessions.json` |
| Which org chart to load | `--org-structure` | `CG_ORG_STRUCTURE` | unset - org structure is skipped |

```bash
uv run context-graph                                          # ingest the default dataset only
uv run context-graph --input other.json                       # any file of the same shape
uv run context-graph --org-structure data/org_structure.json  # fill that file first, then apply it and ingest
```

Neo4j, MongoDB and the chat/embedding model stay where they already are, in
`.env`/`config.py` - this command never takes them as arguments.

The default dataset is ConvoMem. `data/convomem_demo/sessions.json` is built
from the raw benchmark in `data/convomem/` by `convomem_benchmark_converter/`.
That step only writes a JSON file and is not part of a run:

```bash
uv run python -m convomem_benchmark_converter.prepare   # raw ConvoMem -> data/convomem_demo/
```

Extraction, adjudication, segmentation and the retrieval tester all run on the chat
backend configured in `.env`. The default is **NVIDIA NIM**
(`nvidia/nemotron-3.5-bolt-e4b` at `https://integrate.api.nvidia.com/v1`).
Another OpenAI-compatible host is the same client with different settings.

| Setting | Default |
| --- | --- |
| `LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL` | `nvidia/nemotron-3.5-bolt-e4b` |
| `NVIDIA_API_KEY` | chat and embedding credential |

Token budgets assume a **10240-token context window**
(`CG_LLM__CONTEXT_LIMIT_TOKENS`) with a 4000-token reply
(`CG_LLM__MAX_OUTPUT_TOKENS`), and concurrency defaults to 8 in-flight
requests. The model accepts far more; the window is kept small because it sets
the batch size, and the reply cap must hold a full batch of signals or the
batch is truncated and re-run in halves.

With `json_object`, the JSON schema is placed in the prompt and the reply is
validated here, with a bounded repair attempt that shows the model its own
error - a reply that never validates is a failure, never a partial result.

Embeddings use the same NVIDIA host. The default model is
`nvidia/nemotron-3.5-embed-1b` at **2048 dimensions**, sent as passage
embeddings (`input_type=passage`). Vectors are matched back by index.

Changing `EMBEDDING_DIMENSION` invalidates the Neo4j vector indexes. Because
`CREATE VECTOR INDEX … IF NOT EXISTS` silently keeps an existing index, startup
checks the stored dimension and refuses to run against indexes built for another
embedding size rather than searching across two incompatible vector spaces.

Programmatic use:

```python
from context_graph.tool import ContextGraphIngestion, IngestionRequest

async with ContextGraphIngestion.from_settings() as ingestion:
    report = await ingestion.ingest_conversation(IngestionRequest(session=session))
```

One tool handles new, incremental, bulk and historical input; the caller never
selects a processing mode. Each run has its own `run_id`, graph target and
MongoDB processing state — set `CG_RUN_ID` to resume an interrupted run.

## Tests

```bash
uv run pytest                                   # everything that needs no live service
uv run pytest -m unit                           # one suite: unit | e2e | ui | integration
CG_INTEGRATION=1 uv run pytest -m integration   # live Neo4j + MongoDB (writes data)
uv run pyright
```

| Suite | What it covers |
| --- | --- |
| `tests/unit/` | Single components with in-memory doubles, no I/O beyond `tmp_path` |
| `tests/e2e/` | The whole ingestion pipeline against in-memory graph and ledger stores |
| `tests/ui/` | The Streamlit page (rendered headless) and the terminal chat |
| `tests/integration/` | Real Neo4j, MongoDB and model calls |

Each folder's name is also its marker, applied automatically in
`tests/conftest.py`, and markers are strict, so a typo fails the run. Shared
builders live in `tests/factories.py` and `tests/agent_doubles.py`, never in
`conftest.py`. Async tests run under pytest-asyncio in `auto` mode.

Integration tests are opt-in because they write to real databases; point
`CG_NEO4J__*` and `CG_MONGO__*` at a disposable instance before enabling them.
Without `CG_INTEGRATION=1` they are reported as skipped, never silently dropped.

## Test conversation generator

[`test_conversation_generator/README.md`](test_conversation_generator/README.md)
says why this folder exists. `ConversationDatasetGenerator` builds realistic
multi-turn sessions for every behaviour the pipeline implements, using the chat
backend configured in `.env`. A previously generated set can be ingested like
any other file:

```bash
uv run context-graph --input data/synthetic_sessions.json
```

The model supplies conversation text only; `session_id`, `message_id`, sequence
and timestamps are assigned deterministically, so the same seed produces the
same dataset. Every session is validated against `IngestionRequest` before it is
written, and the sidecar `*.manifest.json` records which behaviour each session
exercises (`shared_candidate`, `supersession`, `revocation`, `oversized_episode`,
...). Scenario definitions live in [`test_conversation_generator/scenarios.py`](test_conversation_generator/scenarios.py).

Concept resolution thresholds (`CG_RESOLUTION__ACCEPT_SIMILARITY`,
`CG_RESOLUTION__AMBIGUOUS_SIMILARITY`) are raw cosine, calibrated for
`nvidia/nemotron-3.5-embed-1b`: genuine synonyms score 0.76-0.90 while
related-but-distinct concepts reach 0.73, so only near-identical text
auto-accepts (0.95) and the 0.70-0.95 band is decided by LLM adjudication.
Recalibrate them if you change the embedding model.

Token batching and episode chunking only trigger at the configured budget, so to
exercise them against this data, lower it — e.g. `CG_LLM__CONTEXT_LIMIT_TOKENS=6000`
with `CG_BATCHING__MAX_EPISODES_PER_BATCH=4`.

## Organizational structure

Who is on which team, and which projects belong to which team. This is the
governed writer the spec requires for scope membership: conversation ingestion
cannot create these edges, because a user saying "I'm on the Search team" is a
claim, not an authorization.

```bash
uv run context-graph --org-structure data/org_structure.json
```

`data/org_structure.json` is shipped empty. Fill in the organization, departments, teams, people, and projects before running that command. Until `organization_id` and `name` are set, the command stops and writes nothing.

This applies the structure once, in the same run, before ingesting
conversations - there is no separate script for it.

It writes `MEMBER_OF` (user → team), `BELONGS_TO` (team → department →
organization), `PART_OF` (project → team) and `WORKS_ON` (user → project).
MERGE keeps an unchanged edge untouched, so re-applying the same snapshot
changes nothing; a membership, team, project or assignment that changed, or
an entity dropped from the file entirely, has its stale edge removed so the
graph's org edges always match the file. It writes structure and nothing
else - no Context, no Episodes, no ledger, and it never deletes a `User` node,
even one that exists only because a conversation created it.

These edges are what make the graph act like a graph: shared Team context only
reaches a user through `MEMBER_OF`, and questions like "which Aman is on my
team" become a traversal rather than a remembered string.

## Retrieval tester

[`retrieval_tester/README.md`](retrieval_tester/README.md) says why this folder
exists. It reads Context for a user and chats with it. It never writes to the
graph and never calls ingestion. The screens live in `ui/`.

```bash
uv run python -m streamlit run ui/streamlit_app.py   # browser UI (default)
uv run python -m ui.chat_with_agent --user U-praveen  # same thing in the terminal
```

The Streamlit page has the user_id box and session controls in the sidebar, the
chat in the middle, and the retrieved Context, retrieval debug and the exact
system prompt on the right. Both UIs are rendering layers over the same
[`ChatController`](retrieval_tester/chat.py); retrieval and session logic exist once.

The agent also has one **read-only graph lookup tool**
([`retrieval_tester/graph_tools.py`](retrieval_tester/graph_tools.py)). Pre-fetched
retrieval answers "what is true about this user"; structural questions - which
teams exist, who is on Payments, who works on Atlas - are open-ended, so rather
than pre-fetching every path the agent asks the graph itself, and only when a
question needs it. The model never writes a query: it picks one of four fixed
lookups (`teams`, `team_members`, `person`, `project`) and supplies a name, and
every query is parameterised, read-only and confined to the directory - personal
Context is unreachable through it. Turn it off with the sidebar toggle, or
`--no-graph-tool` in the terminal, to compare against pre-fetched retrieval
alone. Each lookup the agent runs is shown in the UI and saved with the session.

Commands: `/context` (what was retrieved), `/debug` (why), `/prompt` (the exact
system prompt), `/new` (new session, same user, retrieval re-run), `/user ID`,
`/sessions`, `/quit`. Each session is saved to `data/agent_sessions/*.json`
after every turn.

The saved file is written in the ingestion tool's own input shape, so you can
ingest a retrieval-tester conversation later **by choice**:

```bash
uv run --system-certs context-graph --input data/agent_sessions/DA-U-praveen-<stamp>.json
```

`session` is what the pipeline reads; the sibling `retrieval_tester` key records the
model, the Context that was retrieved and the retrieval diagnostics, and is
ignored by ingestion. The application itself never does this - it never writes
to the graph and never calls the pipeline; ingesting a transcript is always a
deliberate step you take. Note that agent turns are ingested as
`agent_message`, so they carry derived trust rather than first-party trust.

Retrieval ([`retrieval_tester/retrieval.py`](retrieval_tester/retrieval.py)) answers four
kinds of question, in one round of concurrent read-only queries:

| Capability | What it gives the agent |
| --- | --- |
| Durable context | `active` Context inside its validity window: the user's own USER/TASK/PROJECT scopes, plus shared Context reachable by walking `MEMBER_OF` then `BELONGS_TO` up to the department and organization |
| Directory | the user's team, department, organization, their projects, and their teammates' names, roles and projects |
| Episodic recall | the user's recent Sessions with times, message counts and the concepts derived from each, so "what did we discuss last time" is answerable |
| Provenance | the first supporting Episode's text on each item, so the agent can say when and how something was said |
| Value history | superseded values via `SUPERSEDES`, so "what did I used to prefer" is answerable |

Visibility follows the scope the data sits at: directory facts (who is on a team,
what it owns) are shared with that team; ORGANIZATION and DEPARTMENT context
reaches every member beneath it; and personal context - preferences, goals,
project notes - is returned only to the user who owns it. A colleague is exposed
as name, role and projects only, never their preferences.

Selection ranks the user's own Context against the current message - embeddings
are compared locally rather than through the shared vector index, so one user's
Context can never be crowded out by another's. Standing instructions
(Preference, Constraint, WorkingStyle) are always included whatever the topic,
and relevance only excludes anything once there is more Context than fits.
SESSION-scoped Context is never carried into a new session (A.9). Queries run
with READ routing and the client exposes no write operation; a test asserts the
package never imports `context_graph.tool`, `pipeline`, `ledger` or the graph
store, so there is no path back into ingestion.

This is a stand-in for the separate retrieval design, not an implementation of
it: it is deliberately minimal and sits behind a `ContextRetriever` protocol, so
a real retrieval service can replace it without touching the agent or the UI.

### Planned Context Graph retrieval (`context_graph_retrieval`)

A second, independent retrieval tool
([`retrieval_tester/context_tool.py`](retrieval_tester/context_tool.py)) that takes a
`query` and a `user_id` and plans a traversal instead of running a fixed one.
It is reachable from the **Context Graph retrieval** tab of the Streamlit page;
the chat tool above is unchanged and unaffected.

The graph schema is **discovered, never assumed**: labels, relationship types,
their endpoints and every property name are read from the database itself with
`db.schema.nodeTypeProperties()` and `db.schema.visualization()`
([`retrieval_tester/schema_catalog.py`](retrieval_tester/schema_catalog.py)) and cached for
`DA_SCHEMA_CACHE_SECONDS`. Embedding properties are stripped before the model
ever sees the schema.

Planning is progressive, and costs only the calls it needs:

| Level | What the model is given | What it returns |
| --- | --- | --- |
| 1 | the query and the label/relationship outline, no properties | where the answer lives, keywords, and whether the user's own Context already settles it |
| 2 | properties and edges of **only** the labels level 1 chose | a bounded traversal plan: anchor, hops, optional filters |
| 3 | the schema around the plan's last label | the completed plan - only when level 2 said it must go deeper |

A question about the asking user stops at level 1, so the common case is a single
model call. Every reply is a validated Pydantic model, obtained through the same
structured-output provider the pipeline uses.

A route must pass through `Context` exactly **once** - that is what the tool
returns and what the access guard is applied to - but it does not have to *end*
there. Reaching the Context and carrying on is how a stored fact is traced back
to where it came from:

    (User)-[:MEMBER_OF]->(Team)<-[:APPLIES_TO]-(Context)-[:SUPPORTED_BY]->(Episode)<-[:CONTAINS]-(Session)-[:INVOLVES_AGENT]->(Agent)
      hop 0            hop 1                 hop 2                        hop 3                    hop 4                   hop 5

Which route reaches the answer is a judgement, so the model makes it. How a fact
links back to its evidence is not - it is one fixed chain the pipeline writes on
every ingest - so when level 1 says the question asks where something came from,
`DA_PROVENANCE_CHAIN` (`SUPPORTED_BY`, `CONTAINS`, `INVOLVES_AGENT`) is walked
mechanically, with **no extra model call**. Each link is resolved against the
discovered schema - direction and target label come from the database - and the
walk stops at the first link the graph does not have.

The model produces a plan, never a query. Before anything runs, the plan is
checked against the discovered schema - real labels, real properties, real
relationships in the direction they actually exist, within
`DA_PLANNER_MAX_HOPS` (5, enough for the whole chain above) - and a hop written
backwards is corrected rather than invented. Optional filters naming a scope this user may not read are dropped,
because access is not the model's to decide: the compiled query always carries
the same lifecycle and authorization guard as standard retrieval (`active`,
inside its validity window, the user's own private scopes or a shared scope
resolved from `MEMBER_OF`/`BELONGS_TO`; SESSION scope never applies). A
deliberately greedy plan therefore returns only what that user could already
read. Values always travel as parameters, the query runs with READ routing, and
results are bounded by `DA_PLANNER_MAX_ROWS` and a graph timeout.

Failure is always answerable: an unknown user is reported, an empty result after
the model's filters is retried once on the same path without them, and an
unavailable schema, an invalid plan, a model failure or a graph timeout falls
back to standard retrieval with the reason recorded in the retrieval debug. The
result is an ordinary `RetrievalResult`, so the UI renders it with the same
components as pre-fetched retrieval.

The tab shows three things in order.

**The answer.** The retrieved Context is handed to the same agent prompt the
chat uses, with the lookup tool switched off, so the reply can only come from
what retrieval returned. Because one explicit question makes unrelated context
noise rather than background, the planned path drops items below
`DA_MIN_SIMILARITY` before answering - standing instructions (Preference,
Constraint, WorkingStyle) always stay, exactly as pre-fetch keeps them. This is
the one place the two paths deliberately differ.

**The journey through the graph.** The compiled query returns the path it walked,
not just the rows, so the page draws the real subgraph the way a graph browser
does: circles coloured per label, arrows captioned with the relationship type,
drag, zoom, and click a node to see every property stored on it. A multi-level
traversal is laid out left to right **by hop**, so the levels are the columns:

    (User) Praveen Solanki  --MEMBER_OF-->  (Team) Search  <--APPLIES_TO--  (Context) thursday
         hop 0                                  hop 1                            hop 2

A thick blue ring marks where the walk started, a green ring marks the Context
that answered. *Replay the walk* reveals it one hop at a time, and the slider and
arrows step through it by hand - all inside the viewer, so stepping never re-runs
the query. A static Graphviz picture sits under it for when the viewer's CDN is
unreachable. When level 1 answered directly there are no hops, and the page says
so instead of drawing a journey that never happened.

Two repairs keep real routes runnable instead of throwing them away, and both are
recorded in the trace. A relationship written in the wrong direction is flipped
when the schema allows only one direction, and an anchor property that does not
exist is pointed at that label's identity property from the ontology - so
`(User {MEMBER_OF: 'U-praveen'})`, which a small model does produce, becomes
`(User {user_id: 'U-praveen'})`. Neither can reach anything the schema does not
have: an unknown label, an invented relationship or a label with no identity
property is still rejected, and the plan falls back.

**What happened, step by step.** Each step is one sentence anyone can read -
*"Worked out what U007 is allowed to see: their own private context, plus the
shared context of team platform-infra"*, *"The model proposed a route through the
graph: (Team)<-[:APPLIES_TO]-(Context)"*, *"Checked the model's route against the
real graph and corrected a connection it had pointing the wrong way"* - labelled
with who did it (the model 🧠, the database 🗄️, safety rules 🛡️, relevance 📐,
the safety net ↩️), how long it took, and whether its outcome was used. Every
step keeps the exact material behind a *show the exact detail* fold: the
discovered schema, the resolved scopes, the full system and user prompt of each
planning level, the validated JSON the model replied with, the plan as proposed
against the plan as accepted, the compiled Cypher with its parameters, and the
row counts of each attempt.

## Deliberately out of scope

- Retrieval, request-time scope applicability and authorization (separate design).
- `CONTRADICTS` edges — the spec marks them optional and not a source of truth.
- Episode embeddings — optional until an evidence-search consumer exists.
- `MEMBER_OF`, `BELONGS_TO`, `PART_OF` — org structure and scope membership are
  authorization metadata with no authoritative source in a conversation.
  Ingestion takes conversation as its only input and never writes these edges: a
  user saying "I'm on the Search team" is stored as ordinary private Context (a
  `Fact` at USER scope with its episode evidence), never as an access grant.
- Out-of-order and late-arriving sequences: the POC assumes session input is
  sequential and complete.

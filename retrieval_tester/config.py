"""Configuration for the retrieval tester.

Connection and credential settings are reused from the ingestion package's
configuration so there is one source of truth for Neo4j and the chat backend; only
agent-specific knobs live here, under the ``DA_`` prefix.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from context_graph.config import (
    ChatBackendSettings,
    EmbeddingSettings,
    LLMSettings,
    Neo4jSettings,
    RetrySettings,
    get_settings,
)


class AgentSettings(BaseSettings):
    """Behaviour of the retrieval tester and its local session storage."""

    model_config = SettingsConfigDict(env_prefix="DA_", env_file=".env", extra="ignore")

    #: Directory for saved retrieval-tester sessions (never read by ingestion).
    sessions_dir: Path = Path("data/agent_sessions")
    #: Maximum Context items injected into the agent's system prompt.
    max_context_items: int = Field(default=25, gt=0)
    #: Conversation turns kept in the model request, most recent first.
    max_history_turns: int = Field(default=20, gt=0)
    max_output_tokens: int = Field(default=1_200, gt=0)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    #: Include governed shared-scope Context the user is a member of.
    include_shared_scopes: bool = True
    #: Past conversations recalled for "what did we discuss" questions.
    max_sessions_recalled: int = Field(default=5, ge=0)
    #: Superseded values recalled, so the agent knows what changed.
    max_value_changes: int = Field(default=5, ge=0)
    #: Context nodes fetched from the vector index when ranking by relevance.
    relevance_top_k: int = Field(default=15, gt=0)
    #: Minimum similarity for non-standing Context to be included.
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Let the agent query the graph itself for structural questions.
    enable_graph_tool: bool = True
    #: Lookup rounds allowed before the agent must answer.
    max_tool_iterations: int = Field(default=3, ge=0)
    #: Teammates listed from the organizational directory.
    max_colleagues: int = Field(default=15, ge=0)
    #: Characters kept from a supporting episode as provenance.
    evidence_quote_chars: int = Field(default=140, ge=0)

    # --- planned Context Graph retrieval (context_graph_retrieval tool) ---
    #: Planning levels allowed for one query, including the first.
    planner_max_levels: int = Field(default=3, ge=1)
    #: Hard ceiling on the traversal depth a plan may request. Eight hops covers
    #: membership into a shared scope, the Context on that scope, and the full
    #: provenance chain (Episode → Session → Agent) without rejecting valid plans.
    planner_max_hops: int = Field(default=8, ge=0)
    #: Rows read from the graph before ranking and capping.
    planner_max_rows: int = Field(default=200, gt=0)
    #: Keywords carried into value matching.
    planner_max_keywords: int = Field(default=8, gt=0)
    #: Tokens allowed for one planning reply.
    planner_max_output_tokens: int = Field(default=700, gt=0)
    #: Wall-clock limit for one planning call.
    planner_timeout_seconds: float = Field(default=45.0, gt=0)
    #: Wall-clock limit for one graph read made by the planned tool.
    graph_timeout_seconds: float = Field(default=20.0, gt=0)
    #: How a stored fact links back to its source, in order. Walked without a
    #: model call and only as far as the discovered schema actually goes.
    provenance_chain: tuple[str, ...] = ("SUPPORTED_BY", "CONTAINS", "INVOLVES_AGENT")
    #: Properties carried back per node for inspection in the traversal view.
    max_node_properties: int = Field(default=10, gt=0)
    #: Nodes drawn in the traversal picture before it stops being readable.
    max_graph_nodes: int = Field(default=60, gt=0)
    #: How long a discovered graph schema is reused before it is read again.
    schema_cache_seconds: float = Field(default=300.0, ge=0)


class RetrievalTesterConfig(BaseModel):
    """Everything the application needs, assembled once at startup."""

    neo4j: Neo4jSettings
    chat: ChatBackendSettings
    #: Model budgets, reused by the planned retrieval tool's structured calls.
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings
    retry: RetrySettings
    agent: AgentSettings

    @classmethod
    def load(cls) -> RetrievalTesterConfig:
        """Build configuration from the environment and ``.env``."""
        pipeline = get_settings()
        return cls(
            neo4j=pipeline.neo4j,
            chat=pipeline.chat,
            llm=pipeline.llm,
            embedding=pipeline.embedding,
            retry=pipeline.retry,
            agent=AgentSettings(),
        )

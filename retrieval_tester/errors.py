"""Errors surfaced by the retrieval tester, kept separate from pipeline errors."""

from __future__ import annotations


class RetrievalTesterError(Exception):
    """Base class for retrieval-tester failures."""


class RetrievalError(RetrievalTesterError):
    """Reading Context from the graph failed."""


class AgentError(RetrievalTesterError):
    """The agent could not produce a reply."""


class SessionStoreError(RetrievalTesterError):
    """A retrieval-tester session could not be saved."""

"""``context-graph``: the only way to load data into the Context Graph.

One command, two settings (``input`` and ``org_structure``), both overridable
from the command line or from ``.env``. Everything else - Neo4j, MongoDB, the
chat and embedding model - stays where it already is, in :mod:`context_graph.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from context_graph.config import DEFAULT_INPUT, Settings, get_settings
from context_graph.org import OrgStructure, OrgStructureStore
from context_graph.tool import ContextGraphIngestion, IngestionRequest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load organizational structure and/or ingest conversations.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"JSON file of sessions to ingest (default: CG_INPUT, {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--org-structure",
        type=Path,
        default=None,
        help=(
            "JSON file of org structure to apply in this run before ingesting "
            "conversations (default: CG_ORG_STRUCTURE; leave unset to skip it)."
        ),
    )
    return parser.parse_args(argv)


def load_requests(path: Path) -> list[IngestionRequest]:
    """Read and validate a file of one ingestion request or a list of them."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [IngestionRequest.model_validate(item) for item in _as_list(payload)]


async def apply_org_structure(path: Path, settings: Settings) -> None:
    """Load and apply one org structure snapshot, authoritatively."""
    structure = OrgStructure.model_validate_json(path.read_text(encoding="utf-8"))
    async with OrgStructureStore(settings.neo4j) as store:
        await store.apply(structure)


async def run(settings: Settings) -> int:
    """Apply the org structure if one is set, then ingest every request in ``settings.input``.

    Both files are checked, and the sessions file is validated, before anything
    is written. A missing or unfilled file leaves the graph untouched.
    """
    if settings.org_structure is not None and not settings.org_structure.is_file():
        print(f"org structure not found: {settings.org_structure}", file=sys.stderr)
        return 1
    if not settings.input.is_file():
        print(f"input not found: {settings.input}", file=sys.stderr)
        return 1
    try:
        requests = load_requests(settings.input)
        if settings.org_structure is not None:
            OrgStructure.model_validate_json(
                settings.org_structure.read_text(encoding="utf-8")
            ).validate_references()
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if settings.org_structure is not None:
        print(f"applying org structure {settings.org_structure}")
        await apply_org_structure(settings.org_structure, settings)

    print(f"ingesting {settings.input}")
    async with ContextGraphIngestion.from_settings(settings) as ingestion:
        # Sessions are independent, so they are ingested concurrently; the
        # windows inside one session stay strictly ordered.
        for report in await ingestion.ingest_conversations(requests):
            print(report.model_dump_json(indent=2))
    return 0


def _as_list(payload: object) -> list[object]:
    return payload if isinstance(payload, list) else [payload]


def _resolved_settings(arguments: argparse.Namespace) -> Settings:
    """CLI flags override the corresponding setting for this run only."""
    overrides: dict[str, object] = {}
    if arguments.input is not None:
        overrides["input"] = arguments.input
    if arguments.org_structure is not None:
        overrides["org_structure"] = arguments.org_structure
    settings = get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


def main(argv: list[str] | None = None) -> int:
    """Entry point registered as the ``context-graph`` console script."""
    settings = _resolved_settings(_parse_args(argv))
    return asyncio.run(run(settings))


if __name__ == "__main__":
    raise SystemExit(main())

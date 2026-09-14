"""Loading and validating the benchmark query set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agentpath.paths import evaluation_dir


class QuerySetError(RuntimeError):
    """The query set on disk is unusable."""


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """One benchmark case.

    ``supported`` is derived, not declared twice: a query either names expected
    procedures or expects abstention, never both.
    """

    id: str
    query: str
    type: str
    language: str
    expected: tuple[str, ...]
    expect_abstain: bool

    @property
    def supported(self) -> bool:
        return bool(self.expected)


def queries_path() -> Path:
    return evaluation_dir() / "queries.yaml"


def load_queries(path: Path | None = None) -> list[BenchmarkQuery]:
    """Load the query set, rejecting contradictory expectations."""
    path = path or queries_path()
    if not path.is_file():
        raise QuerySetError(f"query set not found at {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise QuerySetError(f"query set is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict) or "queries" not in raw:
        raise QuerySetError("query set must be a mapping containing 'queries'")

    queries: list[BenchmarkQuery] = []
    seen: set[str] = set()
    for entry in raw["queries"]:
        if not isinstance(entry, dict):
            raise QuerySetError(f"malformed query entry: {entry!r}")
        for field in ("id", "query", "type", "language"):
            if not entry.get(field):
                raise QuerySetError(f"query entry missing {field!r}: {entry!r}")

        query_id = str(entry["id"])
        if query_id in seen:
            raise QuerySetError(f"duplicate query id {query_id}")
        seen.add(query_id)

        expected = tuple(str(item) for item in (entry.get("expected") or ()))
        expect_abstain = bool(entry.get("expect_abstain", False))

        if expected and expect_abstain:
            raise QuerySetError(f"{query_id}: a query cannot both expect procedures and expect abstention")
        if not expected and not expect_abstain:
            raise QuerySetError(f"{query_id}: a query with no expected procedures must set expect_abstain: true")

        queries.append(
            BenchmarkQuery(
                id=query_id,
                query=str(entry["query"]),
                type=str(entry["type"]),
                language=str(entry["language"]),
                expected=expected,
                expect_abstain=expect_abstain,
            )
        )

    if not queries:
        raise QuerySetError("query set is empty")
    return queries


def split_by_support(queries: list[BenchmarkQuery]) -> tuple[list[BenchmarkQuery], list[BenchmarkQuery]]:
    """Separate the two scoring populations."""
    supported = [query for query in queries if query.supported]
    unsupported = [query for query in queries if not query.supported]
    return supported, unsupported

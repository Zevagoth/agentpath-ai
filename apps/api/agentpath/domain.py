"""Typed domain models shared by every layer.

These are plain frozen dataclasses rather than pydantic models on purpose: they
are internal vocabulary, not a wire format. The API layer converts them to
pydantic schemas at the edge, which keeps transport concerns out of the
retrieval logic and lets the retrieval code stay usable from the CLI and the
evaluation harness without dragging FastAPI along.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Language(StrEnum):
    """Detected script of the incoming query."""

    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RetrievalMode(StrEnum):
    """Which retrieval path produced a result."""

    HYBRID = "hybrid"
    """AgentPath AI: query understanding + RAGFlow hybrid retrieval + reranking."""

    LEXICAL_BASELINE = "baseline"
    """Exact/keyword matching only. Benchmark reference, never a production path."""


class EvidenceLevel(StrEnum):
    """How well the retrieved evidence supports answering at all.

    Deliberately coarse. A percentage would imply a calibration we do not have,
    and the UI must not suggest certainty the retrieval scores cannot justify.
    """

    STRONG = "strong"
    """One procedure clearly outranks the rest and scores well."""

    PARTIAL = "partial"
    """Something relevant was found, but the agent must verify before acting."""

    INSUFFICIENT = "insufficient"
    """Nothing retrieved is good enough to answer from. The system abstains."""


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    """A lexicon concept recognised in the query.

    Carries the reason it matched so the UI and the logs can explain an
    expansion instead of presenting it as magic.
    """

    concept_id: str
    canonical: str
    matched_surface: str
    language: Language


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    """The result of query understanding, before any retrieval happens.

    ``original`` is preserved verbatim because it is what the agent typed and
    what we show back to them. ``search_text`` is what the engine sees: the
    original plus deterministic English equivalents, so exact policy terms and
    Arabic phrasings both have a path to the right procedure.
    """

    original: str
    normalized: str
    language: Language
    search_text: str
    expansions: tuple[str, ...] = ()
    corrections: tuple[tuple[str, str], ...] = ()
    concepts: tuple[ConceptMatch, ...] = ()
    scenario_tags: tuple[str, ...] = ()
    detected_product_line: str | None = None
    requested_product_line: str | None = None

    @property
    def effective_product_line(self) -> str | None:
        """An explicit filter from the agent always wins over detection."""
        return self.requested_product_line or self.detected_product_line

    @property
    def is_empty(self) -> bool:
        return not self.normalized.strip()


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrieved chunk, normalized across engines.

    ``lexical_score`` and ``semantic_score`` are kept separate from ``score`` so
    the benchmark can show *why* hybrid retrieval beat the baseline on a given
    query rather than only that it did.
    """

    chunk_id: str
    document_id: str
    procedure_id: str
    title: str
    content: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    highlight: str | None = None
    product_line: str | None = None
    version: str | None = None
    effective_date: str | None = None
    scenario_tags: tuple[str, ...] = ()
    image_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def snippet(self) -> str:
        """Text to show the agent: the engine's highlight when available."""
        return self.highlight or self.content


@dataclass(frozen=True, slots=True)
class ProcedureHit:
    """A procedure, assembled from the passages that matched inside it."""

    procedure_id: str
    title: str
    score: float
    passages: tuple[Passage, ...]
    product_line: str | None = None
    version: str | None = None
    effective_date: str | None = None
    scenario_tags: tuple[str, ...] = ()
    rank: int = 0

    @property
    def best_passage(self) -> Passage:
        return self.passages[0]

    @property
    def matched_passage_count(self) -> int:
        return len(self.passages)


@dataclass(frozen=True, slots=True)
class RetrievedPassages:
    """Raw engine output plus the timings and diagnostics it reported."""

    passages: tuple[Passage, ...]
    mode: RetrievalMode
    engine_ms: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Everything the API returns for one search.

    ``abstained`` is derived rather than set independently: there is exactly one
    rule for refusing to answer, and it lives in ``agentpath.evidence``.
    """

    query: PreparedQuery
    mode: RetrievalMode
    hits: tuple[ProcedureHit, ...]
    evidence: EvidenceLevel
    evidence_reason: str
    timings_ms: Mapping[str, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.evidence is EvidenceLevel.INSUFFICIENT

    @property
    def top_hit(self) -> ProcedureHit | None:
        return self.hits[0] if self.hits else None


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """Reachability of one external dependency."""

    name: str
    reachable: bool
    detail: str
    latency_ms: float | None = None


def rank_hits(hits: Sequence[ProcedureHit]) -> tuple[ProcedureHit, ...]:
    """Sort hits by score and stamp their 1-based rank.

    Ties break on procedure ID so results are stable across runs, which matters
    for reproducible benchmark numbers.
    """
    ordered = sorted(hits, key=lambda hit: (-hit.score, hit.procedure_id))
    return tuple(
        ProcedureHit(
            procedure_id=hit.procedure_id,
            title=hit.title,
            score=hit.score,
            passages=hit.passages,
            product_line=hit.product_line,
            version=hit.version,
            effective_date=hit.effective_date,
            scenario_tags=hit.scenario_tags,
            rank=index,
        )
        for index, hit in enumerate(ordered, start=1)
    )

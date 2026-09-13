"""Wire format for the AgentPath AI API.

These pydantic models are the contract with the web app. They are deliberately
separate from the internal dataclasses in ``agentpath.domain``: the domain can be
refactored without breaking the UI, and the UI gets field names chosen for
clarity at the edge rather than for internal convenience.

Two shape decisions worth stating:

* Confidence is a level plus a written reason, never a percentage. The retrieval
  scores are not calibrated probabilities, and rendering "87%" next to a
  procedure would imply an accuracy the system cannot support.
* Every recommended procedure carries its citations inline. There is no path
  through this schema that returns a recommendation without its source.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentpath.domain import EvidenceLevel, Language, ProcedureHit, RetrievalMode, SearchResult
from agentpath.evidence import ABSTENTION_MESSAGE


class SearchRequest(BaseModel):
    """A customer-service case, as typed by the agent."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="The case in the agent's own words, Arabic or English.",
        examples=["hotel has no room for the guest", "الفندق ما عنده غرف والضيف واصل"],
    )
    product_line: (
        Literal["hotels", "flights", "cancellations", "refunds", "escalations"] | None
    ) = Field(
        default=None,
        description="Optional filter. When set, only procedures in this product line are searched.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="How many procedures to return.")


class Correction(BaseModel):
    """A spelling fix or abbreviation expansion applied to the query."""

    before: str
    after: str


class ConceptTag(BaseModel):
    """A knowledge-base concept recognised in the query."""

    concept_id: str
    canonical: str
    matched_surface: str
    language: Language


class QueryUnderstanding(BaseModel):
    """What the system understood, shown to the agent so it can be corrected."""

    original: str
    normalized: str
    language: Language
    search_text: str = Field(description="What was actually sent to the retrieval engine.")
    expansions: list[str] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    concepts: list[ConceptTag] = Field(default_factory=list)
    scenario_tags: list[str] = Field(default_factory=list)
    detected_product_line: str | None = None
    requested_product_line: str | None = None


class Citation(BaseModel):
    """A retrieved passage, with the scores that put it there."""

    chunk_id: str
    document_id: str
    snippet: str = Field(description="Engine highlight when available, otherwise the chunk text.")
    score: float
    lexical_score: float | None = Field(
        default=None, description="BM25 term similarity contribution."
    )
    semantic_score: float | None = Field(
        default=None, description="Vector similarity contribution."
    )


class ProcedureCard(BaseModel):
    """One recommended procedure."""

    rank: int
    procedure_id: str
    title: str
    product_line: str | None = None
    version: str | None = None
    effective_date: str | None = None
    score: float
    scenario_tags: list[str] = Field(default_factory=list)
    matched_sections: int = Field(description="How many passages of this procedure matched.")
    citations: list[Citation]

    @classmethod
    def from_hit(cls, hit: ProcedureHit) -> ProcedureCard:
        return cls(
            rank=hit.rank,
            procedure_id=hit.procedure_id,
            title=hit.title,
            product_line=hit.product_line,
            version=hit.version,
            effective_date=hit.effective_date,
            score=round(hit.score, 4),
            scenario_tags=list(hit.scenario_tags),
            matched_sections=hit.matched_passage_count,
            citations=[
                Citation(
                    chunk_id=passage.chunk_id,
                    document_id=passage.document_id,
                    snippet=passage.snippet,
                    score=round(passage.score, 4),
                    lexical_score=(
                        round(passage.lexical_score, 4) if passage.lexical_score is not None else None
                    ),
                    semantic_score=(
                        round(passage.semantic_score, 4) if passage.semantic_score is not None else None
                    ),
                )
                for passage in hit.passages
            ],
        )


class Evidence(BaseModel):
    """How far the retrieved evidence can be trusted."""

    level: EvidenceLevel
    reason: str = Field(description="Plain-language explanation of the level.")
    abstained: bool
    message: str | None = Field(
        default=None,
        description="Set only when the system abstains; the exact wording to show the agent.",
    )


class SearchResponse(BaseModel):
    """The full result of one search."""

    mode: RetrievalMode
    query: QueryUnderstanding
    evidence: Evidence
    procedures: list[ProcedureCard]
    conflicts: list[str] = Field(
        default_factory=list,
        description="Procedures that appear to cover the same scenario, for human judgement.",
    )
    timings_ms: dict[str, float]
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: SearchResult) -> SearchResponse:
        prepared = result.query
        return cls(
            mode=result.mode,
            query=QueryUnderstanding(
                original=prepared.original,
                normalized=prepared.normalized,
                language=prepared.language,
                search_text=prepared.search_text,
                expansions=list(prepared.expansions),
                corrections=[Correction(before=before, after=after) for before, after in prepared.corrections],
                concepts=[
                    ConceptTag(
                        concept_id=concept.concept_id,
                        canonical=concept.canonical,
                        matched_surface=concept.matched_surface,
                        language=concept.language,
                    )
                    for concept in prepared.concepts
                ],
                scenario_tags=list(prepared.scenario_tags),
                detected_product_line=prepared.detected_product_line,
                requested_product_line=prepared.requested_product_line,
            ),
            evidence=Evidence(
                level=result.evidence,
                reason=result.evidence_reason,
                abstained=result.abstained,
                message=ABSTENTION_MESSAGE if result.abstained else None,
            ),
            procedures=[ProcedureCard.from_hit(hit) for hit in result.hits],
            conflicts=list(result.conflicts),
            timings_ms=dict(result.timings_ms),
            diagnostics=dict(result.diagnostics),
        )


class DependencyStatus(BaseModel):
    """Reachability of one external dependency."""

    name: str
    reachable: bool
    detail: str
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Service health, including every dependency the search path needs."""

    status: Literal["ok", "degraded"]
    version: str
    mode: RetrievalMode
    dataset: str
    dependencies: list[DependencyStatus]

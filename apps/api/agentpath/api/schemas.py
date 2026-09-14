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
from agentpath.screenshots import SCREENSHOT_MEDIA_TYPE, Screenshot, ScreenshotCatalogue


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


class HighlightBox(BaseModel):
    """Where the highlighted control sits in a screenshot, in CSS pixels.

    Top-left origin, ``deviceScaleFactor`` 1, measured from the live element by
    the generator. Given the image's own ``width`` and ``height``, the UI can
    scale this box to whatever size it renders the image at.
    """

    control_name: str
    x: float
    y: float
    width: float
    height: float


class LocalizedDescription(BaseModel):
    """The bilingual description of what a screenshot shows."""

    en: str
    ar: str


class ScreenshotRef(BaseModel):
    """The screenshot that illustrates a retrieved passage.

    Two identifier spaces meet here and only one of them is safe to publish.
    RAGFlow's own handle is ``img_id``, shaped ``"{dataset_id}-{chunk_id}"``,
    carried internally as ``Passage.image_id``. It is deliberately *not* in this
    model: it is meaningless outside the one RAGFlow instance that minted it, it
    changes whenever a document is re-uploaded or re-parsed, and fetching it needs
    a RAGFlow credential a browser must never be given. ``screenshot_id`` is the
    stable ``agentdesk-*`` name, which is committed in the repository and is what
    this API serves.

    ``url`` is issued by the server, already complete. No client concatenates
    anything to reach an image, which is the point: the only URLs in circulation
    are ones the server minted from its own catalogue.
    """

    model_config = ConfigDict(extra="forbid")

    screenshot_id: str = Field(description="Stable screenshot name, e.g. `agentdesk-booking-detail-status`.")
    url: str = Field(description="Path on this API that returns the PNG. Server-issued; do not construct it.")
    media_type: str = Field(description="Content type the URL returns.")
    width: int
    height: int
    bytes: int
    sha256: str = Field(description="Digest of the bytes the URL returns, verified when the service started.")
    procedure_id: str
    procedure_version: str
    step_order: int = Field(description="Which step of the procedure this screenshot illustrates.")
    screen_name: str = Field(description="The console screen shown, e.g. `Booking Detail`.")
    control_name: str = Field(description="The control the step acts on, e.g. `Booking status chip`.")
    description: LocalizedDescription
    watermark: str = Field(description="The synthetic-data banner rendered into the image.")
    language: str = Field(description="Language of the console labels in the image.")
    highlight: HighlightBox | None = None

    @classmethod
    def from_screenshot(cls, screenshot: Screenshot) -> ScreenshotRef:
        return cls(
            screenshot_id=screenshot.screenshot_id,
            url=screenshot.url_path,
            media_type=SCREENSHOT_MEDIA_TYPE,
            width=screenshot.width,
            height=screenshot.height,
            bytes=screenshot.byte_count,
            sha256=screenshot.sha256,
            procedure_id=screenshot.procedure_id,
            procedure_version=screenshot.procedure_version,
            step_order=screenshot.step_order,
            screen_name=screenshot.screen_name,
            control_name=screenshot.control_name,
            description=LocalizedDescription(en=screenshot.description_en, ar=screenshot.description_ar),
            watermark=screenshot.watermark,
            language=screenshot.language,
            highlight=(
                HighlightBox(
                    control_name=screenshot.highlight.control_name,
                    x=screenshot.highlight.x,
                    y=screenshot.highlight.y,
                    width=screenshot.highlight.width,
                    height=screenshot.highlight.height,
                )
                if screenshot.highlight is not None
                else None
            ),
        )


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
    # Additive, and optional in the strong sense: `null` whenever the passage does
    # not name a screenshot this service holds, which is the case for every
    # procedure outside the Stage 2A slice. A client written against the Stage 1
    # contract reads the fields above and is unaffected.
    screenshot: ScreenshotRef | None = Field(
        default=None,
        description=(
            "The screenshot illustrating this passage's step, when one exists. "
            "Discovery happens here rather than through a listing route: a client only "
            "ever needs the screenshot for a step it is already showing."
        ),
    )


def _screenshot_for(content: str, screenshots: ScreenshotCatalogue) -> ScreenshotRef | None:
    """Map a retrieved chunk to the screenshot it depicts, if any.

    The chunk's own text is the mapping. Three alternatives were available and
    each was rejected:

    * ``Passage.image_id`` - RAGFlow's ``"{dataset_id}-{chunk_id}"``. It says an
      image exists but not *which* screenshot it is, it is local to one RAGFlow
      instance, and ``chunk_id`` is ``xxhash64(content + document_id)`` so it moves
      whenever a document is re-uploaded. It is also the wrong direction: it
      points at RAGFlow's copy, which needs a credential.
    * The build manifest ``data/synthetic/build/visual_chunks.json``, which maps
      chunk id to screenshot id. It is gitignored, describes one machine's RAGFlow,
      and goes stale the moment a re-parse changes a chunk id - the API would then
      serve a screenshot reference for a chunk that no longer exists, or none for
      one that does.
    * The citation token ``NW-HTL-014#S01``. Present verbatim in both kinds of
      chunk, but the sidecars the catalogue is built from record ``step_order``,
      not ``step_id``, so turning a token into a screenshot would mean loading the
      procedure YAML in the API - exactly the ingestion-time work the import-graph
      guarantee exists to keep off the request path.

    The screenshot id needs none of that. It is in the rendered Markdown under the
    step heading and in ``build_visual_chunk_content``'s output, it is the
    catalogue's own key, and it is the id the serving route resolves - so the
    mapping is an exact match against a set we already hold.
    """
    if not screenshots:
        return None
    screenshot = screenshots.for_chunk_text(content)
    return ScreenshotRef.from_screenshot(screenshot) if screenshot is not None else None


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
    def from_hit(cls, hit: ProcedureHit, screenshots: ScreenshotCatalogue | None = None) -> ProcedureCard:
        """Build the card.

        *screenshots* defaults to ``None``, which means "serve no screenshot
        references". Every existing caller keeps working and gets the Stage 1
        response verbatim.
        """
        catalogue = screenshots or ScreenshotCatalogue.empty()
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
                    screenshot=_screenshot_for(passage.content, catalogue),
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
    def from_result(cls, result: SearchResult, screenshots: ScreenshotCatalogue | None = None) -> SearchResponse:
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
            procedures=[ProcedureCard.from_hit(hit, screenshots) for hit in result.hits],
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

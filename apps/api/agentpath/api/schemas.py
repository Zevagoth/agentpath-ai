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

from agentpath.case_context import CaseContext
from agentpath.domain import EvidenceLevel, Language, ProcedureHit, RetrievalMode, SearchResult
from agentpath.evidence import ABSTENTION_MESSAGE
from agentpath.generation.answer_service import AnswerResult, AnswerStatus
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
    product_line: Literal["hotels", "flights", "cancellations", "refunds", "escalations"] | None = Field(
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
    requested_procedure_id: str | None = Field(
        default=None,
        description=(
            "A procedure id found in the query. When set, the search was restricted to "
            "that procedure: an id is an exact request, not a search term."
        ),
    )


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
    lexical_score: float | None = Field(default=None, description="BM25 term similarity contribution.")
    semantic_score: float | None = Field(default=None, description="Vector similarity contribution.")
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


def _screenshot_for(content: str, screenshots: ScreenshotCatalogue, *, procedure_id: str) -> ScreenshotRef | None:
    """Map a retrieved chunk to the screenshot it depicts, if any.

    ``procedure_id`` is required, not optional, because the chunk text alone is
    not sufficient evidence. Two procedures reference the same screen: both
    ``NW-HTL-014`` and ``NW-HTL-021`` declare ``agentdesk-booking-detail-status``
    and ``agentdesk-supplier-log``, so an ``NW-HTL-021`` passage contains those
    ids and a text-only match attaches ``NW-HTL-014``'s capture to it. That was
    observed live: a search for the Arabic booking-status question returned
    ``NW-HTL-021`` at rank 2 carrying both of ``NW-HTL-014``'s screenshots.

    The image is wrong to show, not merely mismatched metadata. Every capture is
    watermarked with the procedure and step it was generated for, so the agent
    would be reading "procedure NW-HTL-014, step 1" beside an NW-HTL-021 result,
    at a different version, for a different step number. The ingestion side
    already refuses this - ``visual_steps`` reports those two as gaps rather than
    reusing the captures - and this is the same rule applied on the way out.

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
    if screenshot is None or screenshot.procedure_id != procedure_id:
        return None
    return ScreenshotRef.from_screenshot(screenshot)


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
                    lexical_score=(round(passage.lexical_score, 4) if passage.lexical_score is not None else None),
                    semantic_score=(round(passage.semantic_score, 4) if passage.semantic_score is not None else None),
                    screenshot=_screenshot_for(passage.content, catalogue, procedure_id=hit.procedure_id),
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
                requested_procedure_id=prepared.requested_procedure_id,
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


# --------------------------------------------------------------------------- #
# POST /api/answer
# --------------------------------------------------------------------------- #


class AnswerRequest(BaseModel):
    """A case, plus anything the agent already knows about it."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="The case in the agent's own words, Arabic or English.",
        examples=["the hotel says it has no room and the guest is at reception"],
    )
    product_line: Literal["hotels", "flights", "cancellations", "refunds", "escalations"] | None = Field(
        default=None,
        description="Optional filter. Takes precedence over case_context.product_type.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="How many procedures to consider.")
    case_context: CaseContext | None = Field(
        default=None,
        description=(
            "Non-sensitive facts the agent already has, used to narrow the search, pick the "
            "right branch and avoid re-asking. It never overrides the procedure. Personal and "
            "financial fields are rejected by the schema rather than ignored."
        ),
    )


class AnswerStepOut(BaseModel):
    """One instruction, with everything needed to verify it.

    ``instruction`` is the only field the model wrote. Every other field was
    attached by the server from the procedure catalogue after the model chose a
    step id, which is why none of them can be invented.
    """

    instruction: str
    procedure_id: str
    procedure_version: str
    step_id: str
    citation: str = Field(description="Human-readable, e.g. 'NW-HTL-014 v2.1 · NW-HTL-014#S01'.")
    source_section: str = Field(description="Where in the source document this step lives.")
    control_name: str | None = Field(
        default=None, description="Authoritative name of the control to operate, from the procedure."
    )
    conditions: list[str] = Field(default_factory=list, description="Structured conditions that route into this step.")
    screenshot: ScreenshotRef | None = Field(default=None, description="The screenshot for this step, when one exists.")


class ContextUsed(BaseModel):
    """What the supplied case context actually did.

    Present so the effect is visible rather than inferred: an agent who supplied a
    fact and was not asked about it should be able to see why, and a reviewer should
    be able to confirm the context narrowed the search rather than changed the advice.
    """

    provided: list[str] = Field(default_factory=list, description="Fields the caller set.")
    applied: dict[str, str] = Field(
        default_factory=dict, description="Field name to what it did, in words an agent can read."
    )
    unanswered: list[str] = Field(
        default_factory=list,
        description="Information the procedure needs that the context could not supply.",
    )
    completed_steps: list[str] = Field(default_factory=list, description="Steps the agent said they had already done.")
    product_line: str | None = Field(
        default=None, description="Product-line restriction derived from the context, if any."
    )


class AnswerResponse(BaseModel):
    """The full result of one answer attempt, whatever became of it.

    ``status`` is the field to branch on. The three failure modes are deliberately
    distinct: ``abstained`` means the knowledge base has no answer, ``generation_failed``
    means the evidence was there and the model server was not, and ``empty_query``
    means nothing was asked. An agent who cannot tell them apart learns to distrust
    the first.
    """

    status: AnswerStatus
    query: str
    detected_language: Language
    summary: str = Field(description="For a non-answer status, this is the message to show.")
    steps: list[AnswerStepOut] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list, description="The corpus's own words on when this applies.")
    escalation: str | None = None
    customer_wording: str | None = Field(
        default=None, description="Suggested wording for the customer. Carries no policy weight."
    )
    clarifying_questions: list[str] = Field(
        default_factory=list,
        description="At most one, and only when the evidence does not settle the case.",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Procedures that overlap, with versions. Surfaced, never resolved for you.",
    )
    sources: list[str] = Field(default_factory=list)
    screenshots: list[ScreenshotRef] = Field(
        default_factory=list,
        description="Every distinct screenshot referenced by a step, in step order.",
    )
    evidence: Evidence
    model_called: bool = Field(description="Whether the local model was consulted. False on every abstention.")
    failure_detail: str | None = Field(
        default=None, description="Operator-facing reason, set only when generation failed."
    )
    dropped_citations: list[str] = Field(
        default_factory=list,
        description="Citations the model returned that the evidence did not support. Should be empty.",
    )
    context_used: ContextUsed
    timings_ms: dict[str, float]
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: AnswerResult, screenshots: ScreenshotCatalogue | None = None) -> AnswerResponse:
        """Convert the service's frozen dataclass to the wire format.

        The single conversion point, as ``SearchResponse.from_result`` is for
        search. Screenshots are resolved here rather than in the service so the
        service stays free of HTTP concerns: it carries a screenshot *id*, and the
        URL is a fact about this API.
        """
        catalogue = screenshots if screenshots is not None else ScreenshotCatalogue.empty()

        steps: list[AnswerStepOut] = []
        gallery: list[ScreenshotRef] = []
        seen: set[str] = set()
        for step in result.steps:
            record = catalogue.resolve(step.screenshot_id) if step.screenshot_id else None
            # The same guard as on a search citation: a capture is watermarked with
            # the procedure and step it was made for, so it belongs to that procedure
            # and nowhere else.
            reference = (
                ScreenshotRef.from_screenshot(record)
                if record is not None and record.procedure_id == step.procedure_id
                else None
            )
            if reference is not None and reference.screenshot_id not in seen:
                seen.add(reference.screenshot_id)
                gallery.append(reference)
            steps.append(
                AnswerStepOut(
                    instruction=step.instruction,
                    procedure_id=step.procedure_id,
                    procedure_version=step.procedure_version,
                    step_id=step.step_id,
                    citation=step.citation,
                    source_section=step.source_section,
                    control_name=step.control_name,
                    conditions=list(step.conditions),
                    screenshot=reference,
                )
            )

        use = result.context_used
        return cls(
            status=result.status,
            query=result.query,
            detected_language=result.detected_language,
            summary=result.summary,
            steps=steps,
            conditions=list(result.conditions),
            escalation=result.escalation,
            customer_wording=result.customer_wording,
            clarifying_questions=list(result.clarifying_questions),
            conflicts=list(result.conflicts),
            sources=list(result.sources),
            screenshots=gallery,
            evidence=Evidence(
                level=result.evidence_level,
                reason=result.evidence_reason,
                abstained=result.abstained,
                message=ABSTENTION_MESSAGE if result.abstained else None,
            ),
            model_called=result.model_called,
            failure_detail=result.failure_detail,
            dropped_citations=list(result.dropped_citations),
            context_used=ContextUsed(
                provided=list(use.provided),
                applied=dict(use.applied),
                unanswered=list(use.unanswered),
                completed_steps=list(use.completed_steps),
                product_line=use.product_line,
            ),
            timings_ms=dict(result.timings_ms),
            retrieval_metadata=dict(result.retrieval_metadata),
        )

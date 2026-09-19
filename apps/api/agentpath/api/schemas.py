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
from agentpath.corpus_retrieval.ports import CorpusRetrievalResult, OffsetSource, RetrievalMethod
from agentpath.corpus_retrieval.service import CorpusSearchService
from agentpath.domain import EvidenceLevel, Language, ProcedureHit, RetrievalMode, SearchResult
from agentpath.evidence import ABSTENTION_MESSAGE
from agentpath.generation.answer_service import AnswerResult, AnswerStatus, StepSource
from agentpath.provenance import ImageProvenance, Provenance, SourceKind
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

    ``instruction`` is the only field the model wrote, and only when ``source`` is
    ``model``. Every other field was attached by the server from the procedure
    catalogue after the model chose a step id, which is why none of them can be
    invented.
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
    source: StepSource = Field(
        default=StepSource.MODEL,
        description=(
            "'model' means the model chose this step and wrote this instruction. 'prerequisite' means the "
            "server inserted it, verbatim from the procedure, because a cited step declares it depends on it."
        ),
    )
    requires: list[str] = Field(
        default_factory=list,
        description="Steps this one depends on, from the procedure. Explains why a prerequisite step is present.",
    )


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
        description="Information the procedure needs that the context could not supply, and that is worth asking for.",
    )
    not_collected: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Information the procedure requires that this system will not ask for, mapped to why. "
            "These are identifiers and payment data the case-context contract rejects, so the agent "
            "is never asked to supply them. Reported rather than dropped, so the policy is visible."
        ),
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
    inserted_prerequisites: list[str] = Field(
        default_factory=list,
        description=(
            "Steps the server added because a cited step declared it depends on them. "
            "Non-empty means the model's own step selection was incomplete."
        ),
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
                    source=step.source,
                    requires=list(step.requires),
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
            inserted_prerequisites=list(result.inserted_prerequisites),
            context_used=ContextUsed(
                provided=list(use.provided),
                applied=dict(use.applied),
                unanswered=list(use.unanswered),
                not_collected=dict(use.not_collected),
                completed_steps=list(use.completed_steps),
                product_line=use.product_line,
            ),
            timings_ms=dict(result.timings_ms),
            retrieval_metadata=dict(result.retrieval_metadata),
        )


# --------------------------------------------------------------------------- #
# External corpora
#
# A separate family of models rather than an extension of SearchResponse. The two
# answer different questions - one ranks versioned procedures for a case, the other
# ranks documents and images of a third-party corpus - and a union type would have
# meant every field on both sides becoming optional, which pushes the branching into
# the UI and makes "which fields are set" a matter of convention.
# --------------------------------------------------------------------------- #


class ImageProvenanceOut(BaseModel):
    """One image, with everything a reader needs to check it.

    ``source_url`` is the upstream original and is often ``null``: 299 of
    MRAMG-Manual's 2,599 images record none. The field is null rather than
    backfilled with a plausible link, and the UI says so - a fabricated "original"
    would be the exact failure the provenance contract exists to prevent.
    """

    image_id: str
    url: str | None = Field(default=None, description="Where this API serves the bytes.")
    source_url: str | None = Field(default=None, description="Upstream original, or null when the corpus has none.")
    document_id: str | None = None
    position_index: int | None = Field(default=None, description="0-based ordinal among the document's images.")
    char_offset: int | None = Field(default=None, description="Where the image belongs in the document's own text.")
    caption: str | None = Field(default=None, description="Null for every MRAMG-Manual image; see RETRIEVAL_MODES.md.")
    position_summary: str = Field(description="One line naming the position, for display.")

    @classmethod
    def from_provenance(cls, record: ImageProvenance) -> ImageProvenanceOut:
        return cls(
            image_id=record.image_id,
            url=record.served_url,
            source_url=record.source_url,
            document_id=record.document_id,
            position_index=record.position_index,
            char_offset=record.char_offset,
            caption=record.caption,
            position_summary=record.position_summary,
        )


class ProvenanceOut(BaseModel):
    """The trail for one retrieved document, and the link that opens it.

    ``open_url`` is issued by the server, complete, and may be ``null``. A client
    must not build it: the whole point of the contract is that a link either came
    from a validated upstream URL or from this API's own source registry, and a
    concatenated third option would bypass both. ``has_original`` is the field a UI
    branches on to decide whether to render `Open original source` at all.
    """

    corpus_id: str
    dataset_id: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    chunk_id: str | None = None
    kind: SourceKind = Field(description="Whether the original is an upstream URL, a local document, or absent.")
    open_url: str | None = Field(default=None, description="Complete link that opens the original at the citation.")
    has_original: bool = Field(description="False means there is genuinely no original to open.")
    position: str = Field(description="Human-readable position within the document.")
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    images: list[ImageProvenanceOut] = Field(default_factory=list)

    @classmethod
    def from_provenance(cls, record: Provenance) -> ProvenanceOut:
        return cls(
            corpus_id=record.corpus_id,
            dataset_id=record.dataset_id,
            document_id=record.document_id,
            document_title=record.document_title,
            chunk_id=record.chunk_id,
            kind=record.kind,
            open_url=record.link(),
            has_original=record.kind is not SourceKind.NONE,
            position=record.locator.summary,
            section=record.locator.section,
            char_start=record.locator.char_start,
            char_end=record.locator.char_end,
            images=[ImageProvenanceOut.from_provenance(image) for image in record.images],
        )


class CorpusSummary(BaseModel):
    """One corpus the console can select, with its attribution.

    Licence and attribution are in the *list* rather than on every result, because
    they are facts about the corpus. They are required fields: this project indexes
    someone else's Apache-2.0 data, and a UI that can render a result without being
    able to render where it came from is a UI that will.
    """

    corpus_id: str
    title: str
    licence: str
    attribution: str
    counts: dict[str, int]
    image_retrieval_mode: str = Field(
        description="How images are ranked. 'context-window' means no model saw the image."
    )
    methods: list[RetrievalMethod] = Field(description="Methods available for this corpus, in registration order.")
    default_method: RetrievalMethod


class CorpusListResponse(BaseModel):
    """Every corpus available, plus the synthetic procedures for comparison."""

    corpora: list[CorpusSummary]
    #: The Stage 2A knowledge base. Listed so the console's selector is complete, but
    #: it is served by `/api/search` and `/api/answer`, not by `/api/corpus-search`:
    #: it is procedures with versions and product lines, not documents with images.
    procedures_corpus_id: str
    problems: list[str] = Field(
        default_factory=list,
        description="Bundles that were rejected at startup. A corpus missing here is reported, not silent.",
    )


class CorpusSearchRequest(BaseModel):
    """A question, and which corpus and method to answer it with.

    ``corpus_id`` is required and has no default. A default would mean a client could
    search without stating what it searched, and the one thing this endpoint must
    never do is return results the caller cannot attribute to a corpus.
    """

    model_config = ConfigDict(extra="forbid")

    corpus_id: str = Field(
        min_length=1,
        max_length=64,
        description="Which corpus to search. Exactly one; mixed results are not available.",
        examples=["mramg-manual"],
    )
    question: str = Field(min_length=1, max_length=2000, description="The question, in the user's own words.")
    method: RetrievalMethod | None = Field(
        default=None,
        description="Which retrieval method to use. Defaults to the corpus's best available method.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="How many documents to return.")
    image_limit: int = Field(default=5, ge=0, le=20, description="How many images to return. Zero disables images.")


class CorpusPassageOut(BaseModel):
    """One matched span, with how its position was established.

    ``offset_source`` is exposed deliberately. A citation whose offsets were recovered
    by a whitespace-insensitive search is slightly less precise at its edges than one
    that matched verbatim, and ``unresolved`` means the document opens at the top. A
    reviewer can see which they are looking at instead of being told all citations are
    equal.
    """

    chunk_id: str
    text: str
    score: float
    offset_source: OffsetSource
    char_start: int | None = None
    char_end: int | None = None
    section: str | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None


class CorpusDocumentOut(BaseModel):
    """One retrieved document, its best passages, and its citation."""

    document_id: str
    title: str
    score: float
    rank: int
    passages: list[CorpusPassageOut]
    provenance: ProvenanceOut


class CorpusImageOut(BaseModel):
    """One image the image selector chose, and the reason it chose it.

    Only selected images appear here. A document's other images are not included even
    though they exist and could be rendered: showing an image because it is nearby
    rather than because it was chosen is what the image-selection step exists to stop.
    """

    image_id: str
    document_id: str
    url: str
    source_url: str | None = None
    score: float
    rank: int
    strategy: str = Field(description="What the selector read: 'context-window' or 'document-order'.")
    reason: str
    evidence: str = Field(default="", description="The text that scored. Empty for the positional baseline.")
    matched_terms: list[str] = Field(default_factory=list)
    position_summary: str = ""
    caption: str | None = None


class CorpusSearchResponse(BaseModel):
    """The full result of one corpus search.

    ``corpus_id`` and ``method`` are echoed rather than assumed, so a stored response
    records what produced it. That matters for the evaluation: a results table whose
    rows cannot prove which method wrote them is not evidence.
    """

    corpus_id: str
    corpus_title: str
    licence: str
    attribution: str
    method: RetrievalMethod
    image_strategy: str
    image_retrieval_mode: str
    question: str
    documents: list[CorpusDocumentOut]
    images: list[CorpusImageOut]
    timings_ms: dict[str, float]
    problems: list[str] = Field(
        default_factory=list,
        description="Unresolved offsets and missing image files. Part of the result, not a log line.",
    )
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: CorpusRetrievalResult, service: CorpusSearchService) -> CorpusSearchResponse:
        corpus = service.corpus
        documents: list[CorpusDocumentOut] = []
        for hit in result.documents:
            provenance = service.citation_for(hit, result)
            if provenance is None:
                continue
            documents.append(
                CorpusDocumentOut(
                    document_id=hit.document_id,
                    title=hit.title,
                    score=round(hit.score, 6),
                    rank=hit.rank,
                    passages=[
                        CorpusPassageOut(
                            chunk_id=passage.chunk_id,
                            text=passage.text,
                            score=round(passage.score, 6),
                            offset_source=passage.offset_source,
                            char_start=passage.char_start,
                            char_end=passage.char_end,
                            section=passage.section,
                            lexical_score=passage.lexical_score,
                            semantic_score=passage.semantic_score,
                        )
                        for passage in hit.passages
                    ],
                    provenance=ProvenanceOut.from_provenance(provenance),
                )
            )

        images: list[CorpusImageOut] = []
        for image_hit in result.images:
            record = corpus.image(image_hit.image_id)
            if record is None:
                continue
            trail = record.provenance()
            images.append(
                CorpusImageOut(
                    image_id=image_hit.image_id,
                    document_id=image_hit.document_id,
                    url=record.served_url,
                    source_url=trail.source_url,
                    score=round(image_hit.score, 6),
                    rank=image_hit.rank,
                    strategy=image_hit.strategy,
                    reason=image_hit.reason,
                    evidence=image_hit.evidence,
                    matched_terms=list(image_hit.matched_terms),
                    position_summary=trail.position_summary,
                    caption=trail.caption,
                )
            )

        return cls(
            corpus_id=result.corpus_id,
            corpus_title=corpus.title,
            licence=corpus.licence,
            attribution=corpus.attribution,
            method=result.method,
            image_strategy=service.image_strategy,
            image_retrieval_mode=corpus.image_retrieval_mode,
            question=result.question,
            documents=documents,
            images=images,
            timings_ms={
                "retrieval": round(result.retrieval_ms, 2),
                "image_selection": round(result.image_selection_ms, 2),
                "total": round(result.total_ms, 2),
            },
            problems=list(result.problems),
            diagnostics=dict(result.diagnostics),
        )

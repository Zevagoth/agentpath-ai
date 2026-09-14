"""Attach one image-bearing RAGFlow chunk per visual step.

A parsed procedure document already answers "what are the steps". It cannot
answer "which screen and which control", because the screenshot is not in it.
This module writes the missing half: for every step that has a usable
screenshot, one extra chunk carrying the screenshot itself plus enough text to be
retrieved on its own terms.

Three properties the design is built around.

**Written after parsing, never before.** Re-parsing a document issues an
unfiltered delete of every chunk it owns, manual ones included, so anything
attached before :meth:`~agentpath_ingestion.ingest.IngestionService._wait_for_parsing`
returns is thrown away by the parse that follows. :func:`attach_visual_chunks` is
therefore called at the end of ingestion, and the ordering is asserted in tests
rather than left to the reading order of the caller.

**Idempotent from RAGFlow's state, not from a local record.** The manifest this
writes is a report, not a source of truth. RAGFlow's own chunk listing decides
whether a write happens, because the local file cannot know that a re-parse
deleted the chunk it names - and believing it would mean skipping the write and
shipping a step whose screenshot is gone. See
:meth:`~agentpath.retrieval.ragflow_client.RAGFlowClient.upsert_chunk`.

**A missing screenshot is reported, never skipped.** Only ``NW-HTL-014`` has
generated screenshots in the Stage 2A slice. The rest are returned as gaps and
printed, so "no screenshot appeared" can never be mistaken for "this step has no
screenshot".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath_ingestion.dataset import LoadedProcedure, repo_root
from agentpath_ingestion.schema import FICTIONAL_BANNER, Procedure
from agentpath_ingestion.visual_steps import ScreenshotGap, VisualStep, collect_visual_steps

logger = structlog.get_logger(__name__)

#: Filename of the record written beside the rendered documents.
VISUAL_CHUNK_MANIFEST = "visual_chunks.json"

#: How much of the screenshot's DOM text is carried into the chunk. See
#: :func:`select_dom_excerpt` for why this is a small number.
DOM_EXCERPT_MAX_LINES = 6
DOM_EXCERPT_MAX_CHARS = 420

#: A DOM line shorter than this is a bare label, a numeral or a separator - the
#: console's furniture rather than its wording.
DOM_EXCERPT_MIN_LINE_CHARS = 16


class VisualChunkError(RuntimeError):
    """A visual chunk cannot be written safely."""


@dataclass(frozen=True, slots=True)
class VisualChunkPlan:
    """Everything needed to write one visual chunk, resolved before any call.

    Built without touching RAGFlow, so the content, the identity and the image
    bytes are all testable on their own.
    """

    visual: VisualStep
    content: str
    identity_prefix: str
    important_keywords: tuple[str, ...]
    tag_keywords: tuple[str, ...]
    image: bytes

    @property
    def citation(self) -> str:
        return self.visual.citation

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VisualChunkOutcome:
    """What happened to one visual chunk, for reporting and for the manifest.

    ``action`` is RAGFlow's own verdict on a write path (``created``,
    ``unchanged``, ``image_attached``) or the read-only verdict of
    :func:`inspect_visual_chunks` (``attached``, ``missing``, ``image_missing``,
    ``blocked``). ``image_blocked`` is the loud one: the text landed, the
    screenshot did not, and no amount of re-running fixes it.
    """

    citation: str
    procedure_id: str
    step_id: str
    step_order: int
    screenshot_id: str
    image_path: str
    image_sha256: str
    content_sha256: str
    chunk_id: str
    image_id: str | None
    action: str
    image_blocked: bool = False
    superseded_chunk_ids: tuple[str, ...] = ()

    @property
    def has_image(self) -> bool:
        return bool(self.image_id)


@dataclass
class VisualChunkReport:
    """Result of attaching, or of inspecting, the visual chunks."""

    outcomes: list[VisualChunkOutcome] = field(default_factory=list)
    gaps: list[ScreenshotGap] = field(default_factory=list)
    #: Citations whose procedure has no ingested document to attach them to.
    orphaned: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> list[VisualChunkOutcome]:
        return [item for item in self.outcomes if item.image_blocked or item.action == "blocked"]

    @property
    def missing(self) -> list[VisualChunkOutcome]:
        return [item for item in self.outcomes if item.action in {"missing", "image_missing"}]

    @property
    def with_image(self) -> list[VisualChunkOutcome]:
        return [item for item in self.outcomes if item.has_image]

    @property
    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.outcomes:
            counts[item.action] = counts.get(item.action, 0) + 1
        return counts

    @property
    def ok(self) -> bool:
        """True when every usable screenshot is attached and carries its image.

        Gaps do not make this false: the corpus declares screenshots that have
        not been generated yet, and that is a known shortfall of the slice rather
        than a failure of this run. Everything else does.
        """
        return not self.blocked and not self.missing and not self.orphaned

    @property
    def remedy(self) -> str | None:
        """What to do about a blocked image, in one line, or ``None``."""
        if not self.blocked:
            return None
        return (
            "an image is already stored under these chunks' object keys, left behind by a "
            "deleted or re-parsed chunk, and RAGFlow would concatenate onto it. Re-upload the "
            "documents so the chunk ids move to free keys: make ingest-force"
        )


# --------------------------------------------------------------------------- #
# Chunk content
# --------------------------------------------------------------------------- #


def select_dom_excerpt(
    visual: VisualStep,
    *,
    max_lines: int = DOM_EXCERPT_MAX_LINES,
    max_chars: int = DOM_EXCERPT_MAX_CHARS,
) -> list[str]:
    """Pick the DOM lines that are about *this* control, and only those.

    The whole DOM text is deliberately not used. Two measured reasons.

    *Volume.* The three committed captures carry 2.3k, 2.4k and 2.7k characters
    of extracted text, against 1.7k-2.0k characters for everything else in the
    chunk. Appending all of it would make the console's furniture 54-58% of the
    text the embedder sees, so the chunk's vector would describe an AgentDesk
    screen in general rather than this step.

    *Discrimination.* The furniture is identical across all three screens - the
    same sidebar (``Cases``, ``Booking Detail``, ``Case Notes``, ``Supplier
    Contact``, ``Supplier Log``, ``Relocation Finder``, ``Cost Recovery``,
    ``Messages``), the same header, the same watermark, the same booking
    reference block. Those shared lines pull the three visual chunks *towards*
    each other, which is the opposite of what a query naming one control needs.

    What the DOM uniquely contributes is the literal on-screen wording for this
    control, which exists nowhere in the YAML and which the lexical half of
    hybrid search can only match verbatim - including the generator's own callout
    ("Step 1 - this Booking status chip must read Confirmed"). So the excerpt is
    the lines that mention this step's screen, its control, or the highlighted
    element, in document order, deduplicated and bounded. For the committed
    screenshots that is two or three lines each.

    Lines shorter than :data:`DOM_EXCERPT_MIN_LINE_CHARS` are dropped because at
    that length a match is a bare navigation label rather than wording, and the
    watermark is dropped because it is on every capture and is stated once at the
    end of the chunk anyway.
    """
    anchors = [
        text.casefold() for text in (visual.control_name, visual.highlight.control_name, visual.screen_name) if text
    ]
    kept: list[str] = []
    seen: set[str] = set()
    budget = max_chars

    for raw in visual.dom_text.splitlines():
        # Tabs separate table cells in the extracted text; collapse them so a row
        # reads as one line rather than as alignment.
        line = " ".join(raw.split())
        if len(line) < DOM_EXCERPT_MIN_LINE_CHARS or FICTIONAL_BANNER in line:
            continue
        lowered = line.casefold()
        if lowered in seen or not any(anchor in lowered for anchor in anchors):
            continue
        seen.add(lowered)
        if len(line) > budget:
            continue
        kept.append(line)
        budget -= len(line)
        if len(kept) >= max_lines:
            break
    return kept


def build_visual_chunk_content(visual: VisualStep, *, procedure_title: str) -> str:
    """Compose the text of one visual chunk.

    Shape, and why each part is there:

    * The **citation token first, verbatim**. The answer layer proves a citation
      is real by finding it in the evidence text, so the token has to survive
      into the chunk unaltered; putting it first also makes it usable as the
      identity prefix, which is how a re-run recognises this chunk as the same
      logical thing.
    * **Procedure id, title and version**, so a chunk retrieved on its own can be
      labelled and attributed without depending on which other chunk happened to
      carry the document heading.
    * **Step order, screen and control**, which is the question this chunk exists
      to answer and the vocabulary an agent actually types.
    * **The instruction in English and Arabic.** Arabic is not a translation
      convenience here: it is the only way an Arabic query reaches this chunk
      through the vector half of hybrid search, since the rest of the text is
      English UI wording that stays English.
    * **The bilingual screenshot description**, which is what makes the image
      itself searchable - the picture contributes no tokens.
    * **A bounded DOM excerpt** (see :func:`select_dom_excerpt`).
    * **The fictional-data banner**, so a chunk quoted out of context still says
      it is synthetic.

    No paths appear: the screenshot is named by its id, which is also how the
    serving route will resolve it, so nothing here can leak a filesystem layout.
    ``questions`` is left unset on the chunk for the same reason - a plausible
    question would be invented content, not data.
    """
    lines = [
        f"{visual.citation} — screenshot for step {visual.step_order} of "
        f"{visual.procedure_id} {procedure_title} (version {visual.procedure_version})",
        f"Screen: {visual.screen_name} · Control: {visual.control_name}",
        f"Instruction (English): {visual.instruction}",
    ]
    if visual.instruction_ar:
        lines.append(f"التعليمات بالعربية: {visual.instruction_ar}")
    if visual.expected_result:
        lines.append(f"Expected result: {visual.expected_result}")
    lines += [
        f"Screenshot {visual.screenshot_id} ({visual.image_width}x{visual.image_height}) shows: "
        f"{visual.image_description.en}",
        f"وصف لقطة الشاشة: {visual.image_description.ar}",
    ]
    excerpt = select_dom_excerpt(visual)
    if excerpt:
        lines.append(f"On-screen text at {visual.control_name}: " + " · ".join(excerpt))
    lines.append(f"{FICTIONAL_BANNER} · synthetic AgentDesk capture, not a real console.")
    return "\n".join(lines)


def visual_chunk_keywords(visual: VisualStep) -> tuple[str, ...]:
    """Lexical handles for this chunk, taken only from what the record holds.

    RAGFlow appends ``important_keywords`` to the text it embeds and scores, so
    these are the terms worth repeating: the control and screen an agent names,
    the citation token, the procedure id, and the screenshot id the serving route
    resolves. Nothing is derived or guessed.
    """
    candidates = (
        visual.control_name,
        visual.screen_name,
        visual.citation,
        visual.procedure_id,
        visual.screenshot_id,
    )
    seen: set[str] = set()
    keywords: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            keywords.append(candidate)
    return tuple(keywords)


def visual_chunk_tags(procedure: Procedure) -> tuple[str, ...]:
    """The procedure's own scenario tags, unchanged.

    These are the corpus's existing taxonomy - the same strings that already
    travel as document metadata - so a visual chunk is tagged exactly like the
    document it belongs to. No new vocabulary is introduced here.
    """
    return tuple(procedure.scenario_tags)


def plan_visual_chunks(
    procedures: Sequence[LoadedProcedure],
    *,
    directory: Path | None = None,
    root: Path | None = None,
) -> tuple[list[VisualChunkPlan], list[ScreenshotGap]]:
    """Resolve every writable visual chunk, and the screenshots still missing.

    Reads the PNG from disk here so the write path holds bytes rather than a
    path: nothing downstream needs to know where the repository is.
    """
    base = root if root is not None else repo_root()
    visuals, gaps = collect_visual_steps(procedures, directory=directory, root=base)
    titles = {item.procedure_id: item.procedure for item in procedures}

    _assert_identities_do_not_shadow_each_other(visuals)

    plans: list[VisualChunkPlan] = []
    for visual in visuals:
        procedure = titles[visual.procedure_id]
        image = (base / visual.image_path).read_bytes()
        plans.append(
            VisualChunkPlan(
                visual=visual,
                content=build_visual_chunk_content(visual, procedure_title=procedure.title),
                # The identity is exactly the thing being cited, so a chunk and
                # the token an answer quotes for it cannot drift apart.
                identity_prefix=visual.citation,
                important_keywords=visual_chunk_keywords(visual),
                tag_keywords=visual_chunk_tags(procedure),
                image=image,
            )
        )
    return plans, gaps


def _assert_identities_do_not_shadow_each_other(visuals: Sequence[VisualStep]) -> None:
    """Refuse a corpus where one citation token is a prefix of another.

    Identity is matched with ``startswith``, so ``NW-HTL-014#S01`` would also
    match a chunk for ``NW-HTL-014#S010`` and delete it as a superseded version
    of itself. Two-digit step ids and the forty-step cap make that impossible
    today; this is here so widening either one fails loudly instead of quietly
    eating a chunk.
    """
    citations = sorted({visual.citation for visual in visuals})
    for index, citation in enumerate(citations):
        for other in citations[index + 1 :]:
            if other.startswith(citation):
                raise VisualChunkError(
                    f"citation {citation!r} is a prefix of {other!r}, so the two cannot be told "
                    "apart by identity; pin distinct step ids in the procedure YAML"
                )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


async def attach_visual_chunks(
    client: RAGFlowClient,
    dataset_id: str,
    *,
    document_ids: Mapping[str, str],
    procedures: Sequence[LoadedProcedure],
    directory: Path | None = None,
    root: Path | None = None,
) -> VisualChunkReport:
    """Write one image-bearing chunk per usable visual step.

    *document_ids* maps a procedure id to the RAGFlow document holding it. A
    procedure with a usable screenshot but no document is reported as orphaned
    rather than ignored, because that combination means ingestion did not do what
    the caller thinks it did.

    Must be called only after every affected document has finished parsing. The
    write itself is delegated to
    :meth:`~agentpath.retrieval.ragflow_client.RAGFlowClient.upsert_chunk`, which
    owns the "never post an image onto an occupied object key" rule.
    """
    plans, gaps = plan_visual_chunks(procedures, directory=directory, root=root)
    report = VisualChunkReport(gaps=gaps)

    for plan in plans:
        visual = plan.visual
        document_id = document_ids.get(visual.procedure_id)
        if not document_id:
            report.orphaned.append(visual.citation)
            logger.warning(
                "ingest.visual_chunk_orphaned",
                citation=visual.citation,
                reason="the procedure has no ingested document to attach the screenshot to",
            )
            continue

        result = await client.upsert_chunk(
            dataset_id,
            document_id,
            content=plan.content,
            identity_prefix=plan.identity_prefix,
            important_keywords=plan.important_keywords,
            tag_keywords=plan.tag_keywords,
            image=plan.image,
        )
        report.outcomes.append(
            VisualChunkOutcome(
                citation=visual.citation,
                procedure_id=visual.procedure_id,
                step_id=visual.step_id,
                step_order=visual.step_order,
                screenshot_id=visual.screenshot_id,
                image_path=visual.image_path,
                image_sha256=visual.image_sha256,
                content_sha256=plan.content_sha256,
                chunk_id=result.chunk_id,
                image_id=result.image_id,
                action=result.action,
                image_blocked=result.image_blocked,
                superseded_chunk_ids=result.superseded_chunk_ids,
            )
        )
        logger.info(
            "ingest.visual_chunk",
            citation=visual.citation,
            action=result.action,
            chunk_id=result.chunk_id,
            image=bool(result.image_id),
            image_blocked=result.image_blocked,
        )

    return report


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


async def inspect_visual_chunks(
    client: RAGFlowClient,
    dataset_id: str,
    *,
    document_ids: Mapping[str, str],
    procedures: Sequence[LoadedProcedure],
    directory: Path | None = None,
    root: Path | None = None,
) -> VisualChunkReport:
    """Report the state of every visual chunk without writing anything.

    Four verdicts, and the distinction between the last two is the whole point:

    * ``attached`` - the chunk is there and carries its screenshot.
    * ``missing`` - no chunk for this step, so ingestion has not run since the
      step gained a screenshot.
    * ``image_missing`` - the text is there, the image is not, and the object key
      is free, so the next ``make ingest`` will attach it.
    * ``blocked`` - the text is there, the image is not, and the key is occupied
      by an orphan. Re-running will not fix it; the document has to be
      re-uploaded so the chunk id moves.
    """
    plans, gaps = plan_visual_chunks(procedures, directory=directory, root=root)
    report = VisualChunkReport(gaps=gaps)

    listings: dict[str, list[dict[str, object]]] = {}
    for plan in plans:
        visual = plan.visual
        document_id = document_ids.get(visual.procedure_id)
        if not document_id:
            report.orphaned.append(visual.citation)
            continue

        if document_id not in listings:
            data = await client.list_chunks(dataset_id, document_id)
            chunks = data.get("chunks") if isinstance(data, dict) else None
            listings[document_id] = [chunk for chunk in (chunks or []) if isinstance(chunk, dict)]

        match = next(
            (
                chunk
                for chunk in listings[document_id]
                if str(chunk.get("content") or "").startswith(plan.identity_prefix)
            ),
            None,
        )
        if match is None:
            report.outcomes.append(_outcome(plan, chunk_id="", image_id=None, action="missing"))
            continue

        chunk_id = str(match.get("id") or "")
        # By id, not from the listing: a PATCH-set image id reaches the paged
        # listing only after the doc store refreshes, so the listing can say
        # "no image" about a chunk that has one.
        stored = await client.get_chunk(dataset_id, document_id, chunk_id) or {}
        image_id = str(stored.get("img_id") or match.get("image_id") or "") or None
        if image_id:
            report.outcomes.append(_outcome(plan, chunk_id=chunk_id, image_id=image_id, action="attached"))
            continue

        occupied = await client.fetch_chunk_image(f"{dataset_id}-{chunk_id}") is not None
        report.outcomes.append(
            _outcome(
                plan,
                chunk_id=chunk_id,
                image_id=None,
                action="blocked" if occupied else "image_missing",
                image_blocked=occupied,
            )
        )
    return report


def _outcome(
    plan: VisualChunkPlan,
    *,
    chunk_id: str,
    image_id: str | None,
    action: str,
    image_blocked: bool = False,
) -> VisualChunkOutcome:
    visual = plan.visual
    return VisualChunkOutcome(
        citation=visual.citation,
        procedure_id=visual.procedure_id,
        step_id=visual.step_id,
        step_order=visual.step_order,
        screenshot_id=visual.screenshot_id,
        image_path=visual.image_path,
        image_sha256=visual.image_sha256,
        content_sha256=plan.content_sha256,
        chunk_id=chunk_id,
        image_id=image_id,
        action=action,
        image_blocked=image_blocked,
    )


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def write_visual_chunk_manifest(report: VisualChunkReport, target_dir: Path) -> Path:
    """Write the attached chunk and image ids beside the rendered documents.

    Deliberately *not* the source of truth for idempotence - see the module
    docstring - and deliberately in ``data/synthetic/build/``, which is
    gitignored. The ids in it belong to one local RAGFlow instance and mean
    nothing anywhere else, so committing them would be committing another
    machine's state. What it is for: letting a human see what the last run
    attached, and letting a report quote chunk ids without another round trip.

    Every path in it is repository-relative, enforced upstream by
    :class:`~agentpath_ingestion.visual_steps.VisualStep`.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "banner": FICTIONAL_BANNER,
        "note": (
            "Written by `make ingest`. RAGFlow is the source of truth for whether a chunk "
            "exists; this file is a report of the last run. Ids are local to one RAGFlow "
            "instance, which is why this directory is not committed."
        ),
        "visual_chunk_count": len(report.outcomes),
        "actions": report.actions,
        "visual_chunks": [
            {
                "citation": item.citation,
                "procedure_id": item.procedure_id,
                "step_id": item.step_id,
                "step_order": item.step_order,
                "screenshot_id": item.screenshot_id,
                "image_path": item.image_path,
                "image_sha256": item.image_sha256,
                "content_sha256": item.content_sha256,
                "chunk_id": item.chunk_id,
                "image_id": item.image_id,
                "action": item.action,
                "image_blocked": item.image_blocked,
            }
            for item in report.outcomes
        ],
        "screenshot_gaps": [
            {
                "procedure_id": gap.procedure_id,
                "step_id": gap.step_id,
                "step_order": gap.step_order,
                "screenshot_id": gap.screenshot_id,
                "problems": list(gap.problems),
            }
            for gap in report.gaps
        ],
        "orphaned_citations": list(report.orphaned),
    }
    path = target_dir / VISUAL_CHUNK_MANIFEST
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

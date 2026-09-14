"""The grounded answer service: evidence first, model second, validation last.

This is the service layer only. It owns the whole decision sequence for one
answer and knows nothing about HTTP; the route that exposes it is T10.

The order of operations is the design, and it is deliberately hostile to the
model:

1. **Search, then grade.** :mod:`agentpath.evidence` already owns the single
   abstention rule, and it runs before anything is generated. On ``INSUFFICIENT``
   the service returns the fixed message and **never calls Ollama** -
   :attr:`AnswerResult.model_called` proves it, so a test can assert the absence
   of a call rather than the presence of a message.
2. **A blank query is a third case.** Nothing was searched, so saying "no
   confirmed procedure was found in the available knowledge base" would be a
   false statement about the knowledge base. It gets its own status and its own
   wording.
3. **Rebuild the evidence.** Retrieved chunks are mapped to step ids and each
   step is rebuilt whole from :mod:`agentpath.procedures`. If nothing can be
   rebuilt - the index knows a procedure the corpus does not - the service
   abstains rather than generating from a pointer it cannot resolve.
4. **Constrain the decoder.** ``step_id`` is an ``enum`` over this request's
   citations, and the model is asked for wording and step ids only.
5. **Validate anyway.** Every returned citation is looked up in the assembled
   evidence. ``enum`` should make a bad one impossible; a prompt edit, a schema
   edit or a model swap could each quietly remove that guarantee, and this check
   is what would still catch it. Invalid steps are dropped and counted; if
   nothing valid survives, the service abstains rather than answering.
6. **Attach the rest server-side.** Procedure id, version, control name,
   screenshot reference and source section are read from the catalogue, so they
   are facts about the corpus rather than model output.

Three states that must never be confused, and are not:

* :attr:`AnswerStatus.ABSTAINED` - we looked and the knowledge base has no
  answer.
* :attr:`AnswerStatus.GENERATION_FAILED` - we had the evidence and the model
  server failed. An agent who cannot tell these apart learns to distrust the
  first one.
* :attr:`AnswerStatus.EMPTY_QUERY` - nothing was asked, so nothing was looked up.

Conflicting procedures are surfaced, never resolved.
:func:`~agentpath.evidence.detect_scenario_overlap` computes the notice and its
docstring deferred resolution "until the answer layer exists". This is that
layer, and the resolution it makes is to refuse: both procedures' steps go into
the evidence, both notices go into the response, and the status becomes
:attr:`AnswerStatus.CLARIFICATION_NEEDED` so the UI cannot present one version as
the answer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from agentpath.config import Settings
from agentpath.domain import EvidenceLevel, Language, SearchResult
from agentpath.evidence import ABSTENTION_MESSAGE
from agentpath.generation.evidence_assembly import (
    MAX_EVIDENCE_BLOCKS,
    AssembledEvidence,
    assemble_evidence,
    clarifying_questions,
)
from agentpath.generation.ollama_client import OllamaClient, OllamaError, StructuredOutputError
from agentpath.generation.prompt import DraftAnswer, build_answer_schema, build_prompt
from agentpath.procedures import ProcedureCatalogue
from agentpath.search_service import SearchService

logger = structlog.get_logger(__name__)

#: Shown when the agent submitted nothing. Deliberately *not*
#: :data:`~agentpath.evidence.ABSTENTION_MESSAGE`: that sentence is a claim about
#: the knowledge base, and no search was performed here.
EMPTY_QUERY_MESSAGE = "No question was provided, so nothing was searched."

#: Shown when evidence existed but generation failed. Says which half broke,
#: because the remedy differs and because "the model server is down" must never
#: read as "there is no procedure for this".
GENERATION_FAILED_MESSAGE = (
    "The procedure was found, but the local answer model did not respond, so no answer was generated. "
    "This is not the same as finding no procedure."
)

#: Shown when the model answered but nothing it cited survived validation.
UNGROUNDED_MESSAGE = (
    "A procedure was found, but no part of the generated answer could be traced to a step in it, "
    "so the answer was discarded."
)


class AnswerStatus(StrEnum):
    """What happened, in the one word the UI branches on."""

    ANSWERED = "answered"
    """Evidence was sufficient, the model answered, every citation validated."""

    CLARIFICATION_NEEDED = "clarification_needed"
    """Evidence is partial, or two procedures cover the same scenario. Ask first."""

    ABSTAINED = "abstained"
    """We looked and found nothing good enough. The model was not consulted."""

    EMPTY_QUERY = "empty_query"
    """Nothing was asked. Not a statement about the knowledge base."""

    GENERATION_FAILED = "generation_failed"
    """Evidence was there; the model server was not. Distinct from abstention."""


@dataclass(frozen=True, slots=True)
class AnswerStep:
    """One instruction, with everything needed to verify it.

    ``instruction`` is the model's wording. Every other field on this record was
    attached by the server from the catalogue after the model chose ``step_id``,
    which is why none of them can be invented.
    """

    instruction: str
    procedure_id: str
    procedure_version: str
    step_id: str
    source_section: str
    control_name: str | None = None
    screenshot_id: str | None = None
    screenshot_url: str | None = None
    conditions: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        """The human-readable citation an agent reads under the instruction."""
        return f"{self.procedure_id} v{self.procedure_version} · {self.step_id}"


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """One answer attempt, whatever became of it."""

    status: AnswerStatus
    query: str
    detected_language: Language
    summary: str
    steps: tuple[AnswerStep, ...] = ()
    conditions: tuple[str, ...] = ()
    escalation: str | None = None
    customer_wording: str | None = None
    clarifying_questions: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    evidence_level: EvidenceLevel = EvidenceLevel.INSUFFICIENT
    evidence_reason: str = ""
    #: Whether Ollama was called. The abstention path must leave this ``False``,
    #: and ``tests/test_answer_service.py`` asserts it against a client that
    #: raises if it is touched.
    model_called: bool = False
    #: Set only for :attr:`AnswerStatus.GENERATION_FAILED`. Operator-facing.
    failure_detail: str | None = None
    #: Citations the model returned that the evidence did not support. Should
    #: always be empty while the ``enum`` holds; counted so a regression is data.
    dropped_citations: tuple[str, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    retrieval_metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status in {AnswerStatus.ANSWERED, AnswerStatus.CLARIFICATION_NEEDED}

    @property
    def abstained(self) -> bool:
        return self.status is AnswerStatus.ABSTAINED


class AnswerService:
    """Turns a question into a grounded, cited answer, or into an honest refusal."""

    def __init__(
        self,
        *,
        search: SearchService,
        catalogue: ProcedureCatalogue,
        client: OllamaClient,
        settings: Settings,
        max_blocks: int = MAX_EVIDENCE_BLOCKS,
        screenshot_url_prefix: str | None = None,
    ) -> None:
        self._search = search
        self._catalogue = catalogue
        self._client = client
        self._settings = settings
        self._max_blocks = max_blocks
        # Imported lazily through a parameter rather than from
        # agentpath.screenshots so this module keeps working when screenshots are
        # not configured, and so a test can pin the prefix.
        self._screenshot_prefix = screenshot_url_prefix

    @property
    def model(self) -> str:
        return self._client.model

    async def answer(
        self,
        query: str,
        *,
        product_line: str | None = None,
        limit: int | None = None,
    ) -> AnswerResult:
        """Run the whole sequence for one question.

        Every stage is timed separately, because "the answer took nine seconds"
        is not actionable and "prompt evaluation took seven of them" is.
        """
        total_started = time.perf_counter()
        timings: dict[str, float] = {
            "query_processing": 0.0,
            "retrieval": 0.0,
            "evidence_validation": 0.0,
            "prompt_construction": 0.0,
            "model_generation": 0.0,
            "total": 0.0,
        }

        result = await self._search.search(query, product_line=product_line, limit=limit)
        search_timings = dict(result.timings_ms)
        timings["query_processing"] = search_timings.get("understanding", 0.0)
        # Aggregation is part of "getting results out of the engine": it groups
        # the engine's passages into procedures and does no I/O. Reported inside
        # retrieval_metadata as well, so the split stays inspectable.
        timings["retrieval"] = round(search_timings.get("retrieval", 0.0) + search_timings.get("aggregation", 0.0), 2)
        metadata: dict[str, object] = {
            "mode": result.mode.value,
            "hits": [hit.procedure_id for hit in result.hits],
            "search_timings_ms": search_timings,
            **dict(result.diagnostics),
        }

        if result.query.is_empty:
            return self._finish(
                AnswerResult(
                    status=AnswerStatus.EMPTY_QUERY,
                    query=query,
                    detected_language=result.query.language,
                    summary=EMPTY_QUERY_MESSAGE,
                    evidence_level=result.evidence,
                    evidence_reason=result.evidence_reason,
                    retrieval_metadata=metadata,
                ),
                timings=timings,
                started=total_started,
            )

        if result.evidence is EvidenceLevel.INSUFFICIENT:
            # No Ollama call on this path, by construction: there is no branch
            # between here and the return that touches the client.
            return self._finish(
                self._abstention(query, result, ABSTENTION_MESSAGE, metadata),
                timings=timings,
                started=total_started,
            )

        validation_started = time.perf_counter()
        evidence = assemble_evidence(
            result,
            self._catalogue,
            conflict_score=self._settings.evidence_partial_score,
            max_blocks=self._max_blocks,
        )
        metadata.update(
            {
                "evidence_blocks": len(evidence.blocks),
                "citable_steps": list(evidence.citations),
                "recovered_citations": list(evidence.recovered_citations),
                "retrieved_orphan_chars": evidence.orphan_chars,
                "dropped_evidence_blocks": evidence.dropped_blocks,
                "unknown_procedures": list(evidence.unknown_procedures),
            }
        )
        timings["evidence_validation"] = round((time.perf_counter() - validation_started) * 1000, 2)

        if evidence.is_empty:
            # Retrieval found something, but no step of it could be rebuilt from
            # the structured source. Answering would mean generating from text we
            # have declared untrustworthy, so this abstains - and says why,
            # because it means the index and the corpus disagree.
            reason = (
                f"{result.evidence_reason} No step could be rebuilt from the procedure source"
                + (f" for {', '.join(evidence.unknown_procedures)}" if evidence.unknown_procedures else "")
                + "."
            )
            return self._finish(
                self._abstention(query, result, ABSTENTION_MESSAGE, metadata, reason=reason),
                timings=timings,
                started=total_started,
            )

        prompt_started = time.perf_counter()
        messages = build_prompt(question=result.query.original, language=result.query.language, evidence=evidence)
        schema = build_answer_schema(evidence.citations)
        timings["prompt_construction"] = round((time.perf_counter() - prompt_started) * 1000, 2)

        generation_started = time.perf_counter()
        try:
            structured = await self._client.chat_structured(messages=messages, schema=schema)
        except OllamaError as exc:
            timings["model_generation"] = round((time.perf_counter() - generation_started) * 1000, 2)
            logger.error("answer.generation_failed", error=str(exc), model=self._client.model)
            return self._finish(
                AnswerResult(
                    status=AnswerStatus.GENERATION_FAILED,
                    query=query,
                    detected_language=result.query.language,
                    summary=GENERATION_FAILED_MESSAGE,
                    conflicts=result.conflicts,
                    sources=_sources(evidence),
                    evidence_level=result.evidence,
                    evidence_reason=result.evidence_reason,
                    model_called=True,
                    # The raw text of a StructuredOutputError is deliberately not
                    # carried into the response: it is model output, and the
                    # response is shown to an agent. It is in the log line above.
                    failure_detail=_failure_detail(exc),
                    retrieval_metadata=metadata,
                ),
                timings=timings,
                started=total_started,
            )
        timings["model_generation"] = round((time.perf_counter() - generation_started) * 1000, 2)

        draft = structured.value
        metadata.update(
            {
                "model": structured.completion.model,
                "prompt_tokens": structured.completion.prompt_tokens,
                "generation_tokens": structured.completion.generation_tokens,
                "done_reason": structured.completion.done_reason,
            }
        )

        steps, dropped = self._validate_steps(draft, evidence)
        if not steps:
            logger.warning(
                "answer.no_valid_citation",
                dropped=list(dropped),
                offered=list(evidence.citations),
            )
            return self._finish(
                AnswerResult(
                    status=AnswerStatus.ABSTAINED,
                    query=query,
                    detected_language=result.query.language,
                    summary=UNGROUNDED_MESSAGE,
                    conflicts=result.conflicts,
                    sources=_sources(evidence),
                    evidence_level=result.evidence,
                    evidence_reason=result.evidence_reason,
                    model_called=True,
                    dropped_citations=dropped,
                    retrieval_metadata=metadata,
                ),
                timings=timings,
                started=total_started,
            )

        needs_clarification = result.evidence is EvidenceLevel.PARTIAL or bool(result.conflicts)
        questions: tuple[str, ...] = ()
        if needs_clarification:
            model_question = (draft.clarifying_question or "").strip()
            structured_questions = clarifying_questions(evidence, conflicts=result.conflicts)
            questions = tuple(dict.fromkeys(([model_question] if model_question else []) + list(structured_questions)))

        return self._finish(
            AnswerResult(
                status=AnswerStatus.CLARIFICATION_NEEDED if needs_clarification else AnswerStatus.ANSWERED,
                query=query,
                detected_language=result.query.language,
                summary=draft.summary.strip(),
                steps=steps,
                conditions=_conditions(evidence),
                escalation=(draft.escalation or "").strip() or None,
                customer_wording=draft.customer_wording.strip(),
                clarifying_questions=questions,
                conflicts=result.conflicts,
                sources=_sources(evidence),
                evidence_level=result.evidence,
                evidence_reason=result.evidence_reason,
                model_called=True,
                dropped_citations=dropped,
                retrieval_metadata=metadata,
            ),
            timings=timings,
            started=total_started,
        )

    # -- pieces ------------------------------------------------------------ #

    def _validate_steps(
        self,
        draft: DraftAnswer,
        evidence: AssembledEvidence,
    ) -> tuple[tuple[AnswerStep, ...], tuple[str, ...]]:
        """Bind each drafted step to its evidence block, dropping what does not bind.

        This is the belt to the schema's braces, and it is also where the server
        attaches everything the model was not allowed to write. A step whose id
        is not in the evidence is dropped rather than repaired: repairing would
        mean choosing a step on the model's behalf, which is the guess this whole
        design exists to remove.
        """
        steps: list[AnswerStep] = []
        dropped: list[str] = []
        for drafted in draft.steps:
            citation = (drafted.step_id or "").strip()
            block = evidence.block(citation)
            instruction = (drafted.instruction or "").strip()
            if block is None or not instruction:
                dropped.append(citation or "(empty)")
                continue
            steps.append(
                AnswerStep(
                    instruction=instruction,
                    # Every field below is read from the catalogue-backed block,
                    # never from the model.
                    procedure_id=block.procedure_id,
                    procedure_version=block.procedure_version,
                    step_id=block.step_id,
                    source_section=block.source_section,
                    control_name=block.control_name,
                    screenshot_id=block.screenshot_id,
                    screenshot_url=self._screenshot_url(block.screenshot_id),
                    conditions=block.conditions,
                )
            )
        return tuple(steps), tuple(dict.fromkeys(dropped))

    def _screenshot_url(self, screenshot_id: str | None) -> str | None:
        if not screenshot_id or not self._screenshot_prefix:
            return None
        return f"{self._screenshot_prefix}/{screenshot_id}"

    def _abstention(
        self,
        query: str,
        result: SearchResult,
        message: str,
        metadata: Mapping[str, object],
        *,
        reason: str | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            status=AnswerStatus.ABSTAINED,
            query=query,
            detected_language=result.query.language,
            summary=message,
            conflicts=result.conflicts,
            evidence_level=result.evidence,
            evidence_reason=reason or result.evidence_reason,
            model_called=False,
            retrieval_metadata=dict(metadata),
        )

    def _finish(self, answer: AnswerResult, *, timings: dict[str, float], started: float) -> AnswerResult:
        """Stamp the total, log once, and return.

        One log line per answer, with the stage timings and the citations, so a
        slow or ungrounded answer is diagnosable from the log alone. The query
        text is included only when ``AGENTPATH_LOG_QUERY_TEXT`` is on, matching
        the search service.
        """
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        stamped = AnswerResult(
            status=answer.status,
            query=answer.query,
            detected_language=answer.detected_language,
            summary=answer.summary,
            steps=answer.steps,
            conditions=answer.conditions,
            escalation=answer.escalation,
            customer_wording=answer.customer_wording,
            clarifying_questions=answer.clarifying_questions,
            conflicts=answer.conflicts,
            sources=answer.sources,
            evidence_level=answer.evidence_level,
            evidence_reason=answer.evidence_reason,
            model_called=answer.model_called,
            failure_detail=answer.failure_detail,
            dropped_citations=answer.dropped_citations,
            timings_ms=dict(timings),
            retrieval_metadata=answer.retrieval_metadata,
        )
        logger.info(
            "answer.completed",
            status=stamped.status.value,
            language=stamped.detected_language.value,
            evidence=stamped.evidence_level.value,
            model_called=stamped.model_called,
            steps=[step.step_id for step in stamped.steps],
            dropped=list(stamped.dropped_citations),
            conflicts=len(stamped.conflicts),
            clarifications=len(stamped.clarifying_questions),
            timings_ms=dict(timings),
            query=stamped.query if self._settings.log_query_text else None,
        )
        return stamped


def _sources(evidence: AssembledEvidence) -> tuple[str, ...]:
    """One line per procedure the evidence came from, with its version.

    Present on every non-empty result, including the failure states, so an agent
    can go and read the procedure even when the model produced nothing.
    """
    return tuple(
        f"{record.procedure_id} v{record.version} (effective {record.effective_date}) — {record.title}"
        for record in evidence.procedures
    )


def _conditions(evidence: AssembledEvidence) -> tuple[str, ...]:
    """Every structured condition the evidence carried, deduplicated.

    Returned alongside the answer rather than folded into it: these are the
    corpus's own words, and an agent deciding whether a procedure applies should
    read them rather than a paraphrase.
    """
    return tuple(dict.fromkeys(condition for block in evidence.blocks for condition in block.conditions))


def _failure_detail(exc: OllamaError) -> str:
    """An operator-facing reason for a generation failure, with no model text."""
    if isinstance(exc, StructuredOutputError):
        return f"{exc.__class__.__name__}: the model's reply did not satisfy the answer schema"
    return f"{exc.__class__.__name__}: {exc}"


__all__ = [
    "EMPTY_QUERY_MESSAGE",
    "GENERATION_FAILED_MESSAGE",
    "UNGROUNDED_MESSAGE",
    "AnswerResult",
    "AnswerService",
    "AnswerStatus",
    "AnswerStep",
]

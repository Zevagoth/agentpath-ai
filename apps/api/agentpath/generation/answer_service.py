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
from dataclasses import dataclass, field, replace
from enum import StrEnum

import structlog

from agentpath.case_context import (
    CaseContext,
    ContextUse,
    answers_requirement,
    describe_use,
    refused_reason,
    render_for_prompt,
)
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
from agentpath.procedures import ProcedureCatalogue, StepRecord
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


def _already_known(question: str, context: CaseContext | None) -> bool:
    """Whether the supplied case context already answers *question*.

    Matched through :func:`~agentpath.case_context.answers_requirement`, the same
    function the response's ``context_used`` reports from, so what the agent is
    asked and what they are told about their context cannot disagree.
    """
    if context is None or context.is_empty:
        return False
    return bool(answers_requirement(context, question))


def _must_not_be_asked(question: str) -> bool:
    """Whether *question* asks for something the case-context contract refuses.

    Applied to the model's own question as well as to the structured ones. The
    structured list is already filtered at the source, but the model writes its
    question freely and can ask for a booking reference in wording of its own -
    and that question has no valid answer, because the field an agent would type it
    into is rejected by design.
    """
    return refused_reason(question) is not None


#: How many clarifying questions reach the agent. One, by product decision: an
#: agent mid-call can act on one question, and answering it narrows the case so the
#: next turn can ask the next one. The evidence layer still ranks several - see
#: :func:`~agentpath.generation.evidence_assembly.clarifying_questions` - and this
#: is the cut applied at the edge, so the ordering there decides *which* one.
MAX_QUESTIONS_SHOWN = 1

#: Shown when the model answered but nothing it cited survived validation.
UNGROUNDED_MESSAGE = (
    "A procedure was found, but no part of the generated answer could be traced to a step in it, "
    "so the answer was discarded."
)


class StepSource(StrEnum):
    """Who wrote the wording of a step, so the response stays honest about it.

    The two are not interchangeable to a reader. ``MODEL`` wording was generated
    for this question and phrased for it. ``PREREQUISITE`` wording is the corpus's
    own sentence, inserted by the server because a cited step declares it depends
    on this one - nobody asked for it, and it is shown verbatim rather than
    paraphrased precisely because the model never considered it.
    """

    MODEL = "model"
    """The model chose this step and wrote this instruction."""

    PREREQUISITE = "prerequisite"
    """The server inserted this step, verbatim from the corpus, to satisfy a dependency."""


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
    #: Whether the model wrote this instruction or the server inserted it to
    #: satisfy a declared dependency. See :class:`StepSource`.
    source: StepSource = StepSource.MODEL
    #: Steps this one depends on, as bare ids, from the corpus. Carried so a
    #: reader can see *why* an inserted step is present.
    requires: tuple[str, ...] = ()
    #: Position in the procedure, used to present steps in the order they are
    #: performed rather than in the order the model happened to list them.
    step_order: int = 0

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
    #: Steps the server added because a cited step declared it depends on them.
    #: Non-empty means the model's own selection was incomplete, which is worth
    #: measuring rather than hiding: it is the difference between "the procedure
    #: was followed" and "the procedure was followed because we enforced it".
    inserted_prerequisites: tuple[str, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    retrieval_metadata: Mapping[str, object] = field(default_factory=dict)
    #: What the supplied case context actually did, so its effect is visible rather
    #: than inferred: which fields narrowed the search, which stopped a question
    #: being asked, and which required information is still missing.
    context_used: ContextUse = field(default_factory=ContextUse)

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
        context: CaseContext | None = None,
    ) -> AnswerResult:
        """Run the whole sequence for one question.

        Every stage is timed separately, because "the answer took nine seconds"
        is not actionable and "prompt evaluation took seven of them" is.

        ``context`` carries non-sensitive facts the agent already has - see
        :mod:`agentpath.case_context`. It narrows and informs, and it never
        overrides: it can restrict the search, tell the model which branch of the
        procedure applies, and stop a question being asked about something the agent
        already stated. It contributes no words to the evidence and cannot change
        what a step says.
        """
        total_started = time.perf_counter()
        # An explicit product line wins over the one implied by the case: the
        # dropdown is a deliberate act, the context is ambient.
        effective_product_line = product_line or (context.product_type if context else None)
        timings: dict[str, float] = {
            "query_processing": 0.0,
            "retrieval": 0.0,
            "evidence_validation": 0.0,
            "prompt_construction": 0.0,
            "model_generation": 0.0,
            "total": 0.0,
        }

        result = await self._search.search(query, product_line=effective_product_line, limit=limit)
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
        # An explicit context language overrides detection: the agent knows the
        # customer better than a script detector does. Decided once, because a
        # server-inserted prerequisite has to be written in the same language the
        # model was told to answer in.
        answer_language = context.language if context and context.language else result.query.language
        messages = build_prompt(
            question=result.query.original,
            language=answer_language,
            evidence=evidence,
            case_facts=render_for_prompt(context),
        )
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

        # Step ordering and step dependencies are enforced here, after validation
        # and before the response is built, so no answer can present a step whose
        # declared prerequisite is absent.
        steps, inserted = self._with_prerequisites(steps, language=answer_language)

        needs_clarification = result.evidence is EvidenceLevel.PARTIAL or bool(result.conflicts)
        questions: tuple[str, ...] = ()
        if needs_clarification:
            # Exactly one question reaches the agent. Someone mid-call with a guest
            # at the desk can act on one question; five is a form to fill in, and
            # the specification asks for *a focused* question. Answering it narrows
            # the case, and the next turn asks the next one.
            #
            # The model's question is preferred when it produced one, because it is
            # the only candidate that saw *this* case: the structured questions are
            # derived from the procedure and would be identical for every query
            # against it. The structured list is the fallback, ordered so that the
            # fallback is the most decisive thing left to ask - a fork that changes
            # the path before a fact that is merely missing.
            #
            # Conflicts deliberately do not compete for the slot. They are already
            # surfaced in their own field with both versions named, and spending the
            # question on "which of these two applies" would bury the fact that the
            # knowledge base contains an unresolved overlap.
            model_question = (draft.clarifying_question or "").strip()
            candidates = [model_question] if model_question else []
            candidates += list(clarifying_questions(evidence, conflicts=()))
            # Drop anything the agent already told us. Asking whether the guest has
            # arrived when `customer_arrived` was supplied is the exact waste the
            # case-context contract exists to remove.
            candidates = [q for q in candidates if not _already_known(q, context)]
            # And drop anything we would refuse to accept an answer to. Asking for
            # a booking reference is not a clarification, it is a dead end.
            candidates = [q for q in candidates if not _must_not_be_asked(q)]
            questions = tuple(dict.fromkeys(candidates))[:MAX_QUESTIONS_SHOWN]

        return self._finish(
            AnswerResult(
                status=AnswerStatus.CLARIFICATION_NEEDED if needs_clarification else AnswerStatus.ANSWERED,
                query=query,
                detected_language=result.query.language,
                summary=draft.summary.strip(),
                steps=steps,
                conditions=_conditions(evidence),
                escalation=_usable_escalation(draft.escalation),
                customer_wording=draft.customer_wording.strip(),
                clarifying_questions=questions,
                conflicts=result.conflicts,
                sources=_sources(evidence),
                evidence_level=result.evidence,
                evidence_reason=result.evidence_reason,
                model_called=True,
                dropped_citations=dropped,
                inserted_prerequisites=inserted,
                retrieval_metadata=metadata,
                context_used=describe_use(
                    context,
                    requirements=_requirements(evidence),
                    product_line_applied=(
                        effective_product_line if context and context.product_type and not product_line else None
                    ),
                ),
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
            # The dependency list is read from the catalogue rather than added to
            # the evidence contract: the model is not shown dependencies and must
            # not be, or "S06 requires S05" becomes another instruction it can
            # decline to follow. It is a server fact, resolved server-side.
            record = self._catalogue.step(block.step_id)
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
                    # From the catalogue record, not the evidence block. The block
                    # deliberately carries the *procedure's* scope rules on its
                    # earliest step so the model reads them once; attaching those to
                    # a step in the response made the first step of every answer
                    # recite eleven `applies_when` lines that the response already
                    # reports at procedure level. The record holds only the decision
                    # branches that route into this step, which is what "conditions
                    # on this step" should mean. Nothing is lost: `conditions` on the
                    # response still carries the gate rules.
                    conditions=record.conditions if record is not None else block.conditions,
                    source=StepSource.MODEL,
                    requires=record.requires if record is not None else (),
                    step_order=block.step_order,
                )
            )
        return tuple(steps), tuple(dict.fromkeys(dropped))

    def _with_prerequisites(
        self,
        steps: tuple[AnswerStep, ...],
        *,
        language: Language,
    ) -> tuple[tuple[AnswerStep, ...], tuple[str, ...]]:
        """Add every step the cited steps depend on, then put them in procedure order.

        This is the fix for a measured failure that no prompt wording repaired.
        The model answered "how do I relocate a guest" by citing ``S06`` -
        *present the shortlisted option* - without ``S05``, the step that produces
        the shortlist. It was deterministic across repetitions, and S05 was in the
        evidence the whole time: the model was not missing information, it was
        making a selection, and asking it more nicely to select better is not a
        guarantee. So the dependency is declared in the corpus and satisfied here.

        Two choices worth stating. The dependent step is **kept** - dropping S06
        would satisfy the constraint by removing the answer, which is worse than
        the defect. And the inserted step carries the corpus's own sentence rather
        than a generated one: the model never chose this step, so it has no
        wording for it, and inventing wording on its behalf would be the guess
        this design exists to remove. The Arabic side is used for an Arabic
        answer, because the corpus is bilingual and a language gate is not
        something to fail on text we inserted ourselves.

        Ordering becomes the procedure's rather than the model's. A prerequisite
        that appeared after its dependent would satisfy the letter of the
        dependency and none of its purpose. Procedures keep their order of first
        appearance, so a conflict answer still leads with the procedure the model
        led with.
        """
        by_procedure: dict[str, list[str]] = {}
        for step in steps:
            by_procedure.setdefault(step.procedure_id, []).append(step.step_id)

        present = {(step.procedure_id, step.step_id) for step in steps}
        additions: list[AnswerStep] = []
        for procedure_id, cited in by_procedure.items():
            record = self._catalogue.procedure(procedure_id)
            if record is None:
                continue
            for required in record.dependency_closure(cited):
                if (procedure_id, required.citation) in present:
                    continue
                additions.append(self._prerequisite_step(required, language=language))

        if not additions:
            return tuple(sorted(steps, key=self._presentation_key(by_procedure))), ()

        combined = tuple(sorted([*steps, *additions], key=self._presentation_key(by_procedure)))
        inserted = tuple(step.step_id for step in additions)
        logger.info("answer.prerequisites_inserted", steps=list(inserted))
        return combined, inserted

    @staticmethod
    def _presentation_key(by_procedure: Mapping[str, list[str]]):
        """Order by the procedure's first appearance, then by step number."""
        rank = {procedure_id: index for index, procedure_id in enumerate(by_procedure)}
        return lambda step: (rank.get(step.procedure_id, len(rank)), step.step_order)

    def _prerequisite_step(
        self,
        step: StepRecord,
        *,
        language: Language,
    ) -> AnswerStep:
        """Build one server-inserted step from the corpus.

        Read from the catalogue rather than from the evidence, because a dependency
        is a fact about the procedure and not about what retrieval happened to
        return - the step may never have been offered as evidence at all.
        """
        instruction = step.instruction
        if language is Language.ARABIC and step.instruction_ar:
            instruction = step.instruction_ar
        return AnswerStep(
            instruction=instruction,
            procedure_id=step.procedure_id,
            procedure_version=step.procedure_version,
            step_id=step.citation,
            source_section=step.source_section,
            control_name=step.control_name,
            screenshot_id=step.screenshot_id,
            screenshot_url=self._screenshot_url(step.screenshot_id),
            conditions=step.conditions,
            source=StepSource.PREREQUISITE,
            requires=step.requires,
            step_order=step.step_order,
        )

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
        # `replace` rather than a field-by-field rebuild. The rebuild silently
        # dropped `context_used` when it was added - the field defaulted to empty
        # and every response reported that no context had been supplied, while the
        # context had in fact been applied. A constructor call that has to be
        # edited whenever a field is added will eventually not be.
        stamped = replace(answer, timings_ms=dict(timings))
        logger.info(
            "answer.completed",
            status=stamped.status.value,
            language=stamped.detected_language.value,
            evidence=stamped.evidence_level.value,
            model_called=stamped.model_called,
            steps=[step.step_id for step in stamped.steps],
            dropped=list(stamped.dropped_citations),
            inserted_prerequisites=list(stamped.inserted_prerequisites),
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


def _requirements(evidence: AssembledEvidence) -> tuple[str, ...]:
    """Information the procedures in this evidence say they need, deduplicated.

    Taken from the corpus rather than from the question, so "what is still missing"
    is a property of the procedure and not a guess about the conversation.
    """
    return tuple(dict.fromkeys(item for record in evidence.procedures for item in record.required_information))


#: The shortest thing that could be an escalation instruction. "Escalate now." is
#: 13 characters; a one-word non-answer is never more than a handful.
MIN_ESCALATION_CHARS = 12

#: Ways the model says "not applicable" when asked for an escalation route, in both
#: languages. Compared after stripping and lowercasing.
ESCALATION_NON_ANSWERS = frozenset(
    {
        "no",
        "none",
        "n/a",
        "na",
        "not applicable",
        "not needed",
        "لا",
        "لا يوجد",
        "غير مطلوب",
        "لا حاجة",
    }
)


def _usable_escalation(text: str | None) -> str | None:
    """The model's escalation route, or ``None`` when it did not give one.

    ``escalation`` is optional in the contract, and the model sometimes treats the
    field as a yes/no question rather than a request for a route. Measured on the
    Arabic relocation case, it returned exactly ``"لا"`` - "no". Rendering that
    under a heading like "if this needs escalating" presents a non-answer as
    guidance, which is worse than showing nothing: an agent who reads it learns
    that the section is noise.

    So a reply too short to be an instruction, or one that is a plain negation, is
    treated as absent. Deliberately narrow - it cannot discard a real route, since
    naming a queue or a team takes more than a few characters - and it changes only
    whether the section appears, never its wording.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.strip(".!؟?").lower() in ESCALATION_NON_ANSWERS:
        return None
    if len(stripped) < MIN_ESCALATION_CHARS:
        return None
    return stripped


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
    "StepSource",
]

"""Rebuild generation evidence from the structured source, not from chunk text.

The pipeline this implements::

    retrieve  ->  map each chunk to (procedure_id, step_id)  ->  rebuild each
    step whole from the procedure catalogue  ->  hand those blocks to the model

RAGFlow keeps the job it is good at. It decides which procedure is relevant and
which parts of it matched, and it does that over the rendered bilingual document
where an Arabic query can reach an English procedure. What changes is that its
output is treated as a *pointer* rather than as text: the bytes the model reads
come from :mod:`agentpath.procedures`.

Why, in one measurement
-----------------------
T7 scored four unsupported claims, all of them the single fabricated token
``NW-HTL-014#S10``, and 0/12 correct attributions on the ``en-cost`` case. The
evidence for that case contained this chunk, captured verbatim and committed as
``tests/fixtures/ragflow_chunks_nw_htl_014.json``::

    Present the shortlisted option to the guest and get their agreement before
    booking anything.
    **عربي:** ...
    **Expected result:** The guest has agreed to a specific alternative.
    ### Step 7 — Screen: Relocation Finder — Control: Rebook and Link
    **Step ID:** `NW-HTL-014#S07`
    ...

The first three lines are step *6*. Its heading and its ``**Step ID:**`` line
were cut away by the chunker, so the first token the model could see named step
7. Pairing them was a reasonable reading of the bytes it was given, and every
citation after that shifted by one until it ran off the end of the procedure.

Two failure modes have to be handled, and the second one is the fix
------------------------------------------------------------------
*The clean case* is a substring match. The rendered Markdown writes ``**Step
ID:** `NW-HTL-014#S01``` verbatim and T5's visual chunks open with the citation
token, so a chunk that contains a step's heading contains the token that names
it.

*The truncated case* has no token for the step it starts inside. That is exactly
the case T7 broke on, so it is resolved by matching the orphaned text against the
catalogue's own step text - not by assuming it belongs to the step before the
first visible token, which is the same class of guess that caused the bug, only
in the other direction. The match decides; the position is used at most to break
a tie.

What this buys beyond correct citations
---------------------------------------
Text that arrives in a chunk never reaches the model. A hostile instruction
pasted into the knowledge base can, at most, influence which *step ids* are
selected; its words are discarded, because the words the model sees are read from
the committed YAML. The prompt keeps T7's untrusted-evidence framing anyway -
belt and braces - and the harness keeps measuring injection resistance, because
"structurally impossible" claims are worth exactly as much as the tests that
check them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from agentpath.domain import Passage, ProcedureHit, SearchResult
from agentpath.procedures import (
    CITATION_TOKEN_PATTERN,
    ProcedureCatalogue,
    ProcedureRecord,
    StepRecord,
)

logger = structlog.get_logger(__name__)

#: The heading that owns the first citation token in a truncated chunk. Text
#: from this point onward belongs to the step the token names, so it is not
#: orphaned and must be removed before the orphan is matched.
STEP_HEADING = "### Step"

#: Below this, the text before a chunk's first token is a section heading or a
#: stray bullet rather than an orphaned instruction. Measured on the real
#: retrieved chunks committed as a fixture: genuine orphans are 193, 237 and 253
#: characters, and the one non-orphan head is 52 (``- Number of guests and rooms
#: on the booking\n## Steps``). 60 sits in that gap.
MIN_ORPHAN_CHARS = 60

#: How much of the smaller word set has to be shared before a truncated fragment
#: is accepted as belonging to a step. Chunk text is copied verbatim out of the
#: rendered document, so a genuine match overlaps almost totally and a wrong one
#: barely at all. Measured on the three real truncations in the fixture: the
#: correct step scores 1.00, 0.91 and 0.88, and the best wrong step in each case
#: scores 0.17, 0.19 and 0.19. Nothing sits near this line.
ORPHAN_MATCH_THRESHOLD = 0.5

#: A token-free chunk is not demonstrably truncated - it may simply be the "Do
#: not" list, the summary or the document header, which belong to no step - so
#: recovery there has to clear a much higher bar before it invents a step
#: association. The three token-free chunks in the fixture score 0.28, 0.37 and
#: 0.39 against their best candidate, so all three correctly contribute no step.
UNANCHORED_MATCH_THRESHOLD = 0.75

#: The winning step must beat the runner-up by this much. Without it, a fragment
#: that two steps share equally would resolve to whichever sorted first.
ORPHAN_MATCH_MARGIN = 0.1

#: Words shorter than this carry no discriminating signal in either script.
MIN_WORD_LENGTH = 3

#: Ceiling on the evidence handed to one generation.
#:
#: Twelve, and the history is worth keeping because the first attempt at the
#: latency problem cut this to six and broke correctness.
#:
#: The measurement that mattered: at a p95 of 14.7 s, prompt evaluation was a few
#: hundred milliseconds and decoding was ~10.5 s. The cost was how much the model
#: *wrote*, not how much it read - it had been answering by reciting the whole
#: procedure. Capping the evidence looked like the same lever and is not: it caps
#: what the model can *see*.
#:
#: What that broke, caught by the necessary-steps check in
#: ``scripts/measure_model_fitness.py``: asked in Arabic how to relocate a guest and
#: record the new booking, the answer cited S01, S02, S03, S06, S09 and S10 - and
#: lost S05 (find an equivalent stay) and S07 (Rebook and Link), which are the
#: question. Step ranking is derived from chunk scores, and RAGFlow returns whole
#: chunks, so every step inside one chunk shares a score; the cap then cannot tell
#: which of them the query actually matched.
#:
#: So the evidence stays wide and :data:`MAX_ANSWER_STEPS` does the bounding. The
#: model sees every relevant step and is limited only in how many it writes, which
#: is the thing that was actually expensive.
MAX_EVIDENCE_BLOCKS = 12

#: Ceiling on how many steps one answer may contain.
#:
#: Ten, which is the length of the longest procedure in the corpus - so in practice
#: this bounds nothing and the evidence does the bounding. That is deliberate, and
#: it is the second correction to the same latency problem.
#:
#: Six seemed reasonable and was wrong. Forced to choose six steps out of ten, the
#: model took the first six in procedure order, and the necessary-steps gate caught
#: what that costs: asked how to handle the price difference, it cited S01 through
#: S07 and never reached S08 - the Cost Recovery tab, which *is* the answer. A
#: shorter answer that omits the step the question was about is not a saving.
#:
#: So length is bounded per instruction instead - see
#: :data:`~agentpath.generation.prompt.MAX_INSTRUCTION_CHARS`. An agent scanning a
#: step list needs the action, not the procedure's full paragraph, and trimming
#: each line removes the same output tokens without removing a step.
MAX_ANSWER_STEPS = 10

#: Ceiling on clarifying questions. The specification asks for *a focused*
#: question; the first implementation emitted every ``required_information`` entry
#: and every decision branch, which came to eleven for a single answer.
MAX_CLARIFYING_QUESTIONS = 3

_WORD_SPLIT = re.compile(r"[^\w\u0600-\u06FF]+")


def _words(text: str) -> set[str]:
    """Discriminating words in *text*, English or Arabic.

    Case-folded, punctuation-split, short words dropped. Deliberately the same
    normalisation the T7 harness uses for its attribution metric, so the mapping
    and the measurement of the mapping cannot disagree about what "the same text"
    means.
    """
    return {word for word in _WORD_SPLIT.split(text.casefold()) if len(word) >= MIN_WORD_LENGTH}


# --------------------------------------------------------------------------- #
# Chunk -> step
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChunkMapping:
    """Which steps one retrieved chunk points at, and how each was decided.

    ``explicit`` and ``recovered`` are kept apart rather than merged because they
    have different confidence and different diagnostic value: a corpus change
    that starts producing recoveries where there were none is a chunking change
    worth noticing.
    """

    chunk_id: str
    procedure_id: str
    score: float
    explicit: tuple[str, ...] = ()
    recovered: tuple[str, ...] = ()
    #: Characters of step text this chunk carries before its first citation
    #: token. The T7 evidence-hygiene number, measured per chunk.
    orphan_chars: int = 0
    #: Why a recovery was or was not made, for logs and for test assertions.
    detail: str = ""

    @property
    def citations(self) -> tuple[str, ...]:
        return (*self.explicit, *self.recovered)

    @property
    def truncated(self) -> bool:
        """Whether this chunk began inside a step whose id was cut away."""
        return bool(self.recovered) and self.orphan_chars > 0


def citation_positions(text: str, catalogue: ProcedureCatalogue) -> tuple[tuple[int, str], ...]:
    """Every catalogue-known citation token in *text*, with where it occurs.

    Tokens the catalogue does not know are ignored rather than reported: chunk
    text is retrieved content, so a token-shaped string in it is only ever
    allowed to *select* from the set we loaded, never to describe a step to go
    and find. That is the same direction of trust
    :meth:`agentpath.screenshots.ScreenshotCatalogue.for_chunk_text` uses.
    """
    found: list[tuple[int, str]] = []
    for match in CITATION_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        if catalogue.step(token) is not None:
            found.append((match.start(), token))
    return tuple(found)


def leading_orphan(text: str, first_token_position: int) -> str:
    """The step text a chunk carries before its first citation token.

    The heading that introduces the token's own step is removed first: ``### Step
    7 — Screen: ... — Control: ...`` sits between the orphan and the token, and it
    describes step 7, so leaving it in would pull the match towards the step that
    is already identified.
    """
    head = text[:first_token_position]
    heading = head.rfind(STEP_HEADING)
    if heading != -1:
        head = head[:heading]
    return head.strip()


def match_step(
    region: str,
    procedure: ProcedureRecord,
    *,
    threshold: float,
    exclude: Sequence[str] = (),
) -> tuple[StepRecord | None, float, str]:
    """Which step of *procedure* the text in *region* belongs to.

    Scored as containment of the smaller word set in the larger, which is the
    right shape for this problem: the region is a *fragment* of one step, so
    asking "how much of the fragment does this step account for" works whether
    the chunker kept the whole step or only its last line. Dividing by the step's
    own length instead would penalise a step for being long, and dividing by the
    region's length would let a one-word fragment match anything.

    Returns the step, its score, and a sentence explaining the decision.
    ``exclude`` holds steps already identified by their own token, which cannot
    also be the orphan: a chunk that shows a step's id did not lose it.
    """
    region_words = _words(region)
    if not region_words:
        return None, 0.0, "no matchable words in the fragment"

    skip = set(exclude)
    scores: list[tuple[float, StepRecord]] = []
    for step in procedure.ordered_steps:
        if step.citation in skip or step.step_id in skip:
            continue
        step_words = _words(step.match_text)
        if not step_words:
            continue
        shared = len(region_words & step_words)
        scores.append((shared / min(len(region_words), len(step_words)), step))

    if not scores:
        return None, 0.0, "the procedure offers no other step to match against"

    scores.sort(key=lambda item: (-item[0], item[1].step_order))
    best_score, best = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0

    if best_score < threshold:
        return None, best_score, f"best match {best.citation} scored {best_score:.2f}, under the {threshold:.2f} floor"
    if best_score - runner_up < ORPHAN_MATCH_MARGIN:
        return (
            None,
            best_score,
            f"{best.citation} scored {best_score:.2f} but only {best_score - runner_up:.2f} ahead of the next step",
        )
    return best, best_score, f"matched {best.citation} at {best_score:.2f}, {best_score - runner_up:.2f} clear"


def map_chunk(
    passage: Passage,
    procedure: ProcedureRecord,
    catalogue: ProcedureCatalogue,
) -> ChunkMapping:
    """Map one retrieved chunk onto the steps it points at.

    Three outcomes, and the middle one is the T7 fix:

    * tokens present, nothing orphaned - the tokens are the answer;
    * tokens present *and* step text before the first one - the chunk began
      inside a step whose id was cut away, so that step is recovered by matching
      its text;
    * no tokens at all - the chunk is a summary, a "Do not" list or a decision
      branch, which belong to no step. A high-confidence whole-text match is
      still attempted, and normally finds nothing, which is correct: the chunk
      still identifies the procedure and contributes no step.
    """
    text = passage.content or ""
    positions = citation_positions(text, catalogue)
    explicit = tuple(dict.fromkeys(token for _, token in positions if token.startswith(procedure.procedure_id)))

    if not positions:
        step, _score, detail = match_step(text, procedure, threshold=UNANCHORED_MATCH_THRESHOLD)
        return ChunkMapping(
            chunk_id=passage.chunk_id,
            procedure_id=procedure.procedure_id,
            score=passage.score,
            explicit=(),
            recovered=(step.citation,) if step is not None else (),
            orphan_chars=0,
            detail=f"no citation token in this chunk; {detail}",
        )

    orphan = leading_orphan(text, positions[0][0])
    if len(orphan) < MIN_ORPHAN_CHARS:
        return ChunkMapping(
            chunk_id=passage.chunk_id,
            procedure_id=procedure.procedure_id,
            score=passage.score,
            explicit=explicit,
            orphan_chars=len(orphan),
            detail=f"{len(orphan)} chars before the first token, under the {MIN_ORPHAN_CHARS}-char floor",
        )

    # The similarity score is what `match_step` thresholds on internally; only the
    # step it settled on and the reason are reported outward.
    step, _score, detail = match_step(orphan, procedure, threshold=ORPHAN_MATCH_THRESHOLD, exclude=explicit)
    return ChunkMapping(
        chunk_id=passage.chunk_id,
        procedure_id=procedure.procedure_id,
        score=passage.score,
        explicit=explicit,
        recovered=(step.citation,) if step is not None else (),
        orphan_chars=len(orphan),
        detail=f"{len(orphan)} orphaned chars before {positions[0][1]}; {detail}",
    )


# --------------------------------------------------------------------------- #
# Evidence blocks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    """One whole step, as the model will see it.

    Exactly the eight fields the answer contract needs, and no others:
    ``procedure_id``, ``procedure_version``, ``step_id``, ``instruction``,
    ``conditions``, ``control_name``, ``screenshot_id``, ``source_section``.

    Two notes on what is *not* here. ``expected_result`` exists on the step and is
    deliberately left out: the field list is exact, and the answer contract has
    nowhere to put an expected result, so carrying it would only spend prompt
    tokens on text the model would have to invent a home for. ``screen_name`` is
    likewise absent - the block names the control, which is the thing an agent
    clicks, and the screen is already inside the instruction text.

    ``instruction`` carries both language sides because the corpus is bilingual
    and the answer language is decided per request: an Arabic answer that is
    faithful to the evidence needs the Arabic instruction, and withholding it
    would leave the model translating English policy text itself.
    """

    procedure_id: str
    procedure_version: str
    step_id: str
    instruction: str
    conditions: tuple[str, ...]
    control_name: str | None
    screenshot_id: str | None
    source_section: str
    #: The Arabic half of ``instruction``. One contract field, two language sides.
    instruction_ar: str | None = None
    #: Presentation order, from the procedure. Not part of the evidence contract;
    #: used to keep the blocks in the order the agent will act in.
    step_order: int = 0
    #: Which retrieved chunks pointed here, for logging and for the response's
    #: retrieval metadata. Never rendered into the prompt.
    from_chunks: tuple[str, ...] = ()

    def render(self) -> str:
        """The block as the prompt carries it: one labelled field per line."""
        lines = [
            f"procedure_id: {self.procedure_id}",
            f"procedure_version: {self.procedure_version}",
            f"step_id: {self.step_id}",
            f"control_name: {self.control_name or '(this step operates no named control)'}",
            f"screenshot_id: {self.screenshot_id or '(no screenshot for this step)'}",
            f"source_section: {self.source_section}",
        ]
        if self.conditions:
            lines.append("conditions:")
            lines += [f"  - {condition}" for condition in self.conditions]
        else:
            lines.append("conditions: (none recorded for this step)")
        lines.append(f"instruction (English): {self.instruction}")
        if self.instruction_ar:
            lines.append(f"instruction (العربية): {self.instruction_ar}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class AssembledEvidence:
    """Everything one generation is allowed to know, and how it was decided."""

    blocks: tuple[EvidenceBlock, ...]
    procedures: tuple[ProcedureRecord, ...]
    mappings: tuple[ChunkMapping, ...]
    #: Procedures that were retrieved but are not in the catalogue, so no step of
    #: theirs could be rebuilt. Reported rather than ignored: it means the corpus
    #: and the index disagree.
    unknown_procedures: tuple[str, ...] = ()
    dropped_blocks: int = 0

    @property
    def citations(self) -> tuple[str, ...]:
        """The citation tokens the model may choose from, in presentation order.

        This is the ``enum`` the answer schema is built over, so a citation
        outside this set is unreachable during decoding.
        """
        return tuple(block.step_id for block in self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    @property
    def procedure_ids(self) -> tuple[str, ...]:
        return tuple(record.procedure_id for record in self.procedures)

    @property
    def orphan_chars(self) -> int:
        """Orphaned step text in the *retrieved* chunks, summed.

        Kept and reported because it is the T7 diagnostic that explained the
        citation failures, and it does not go to zero just because the evidence
        the model reads is now rebuilt - the chunker still truncates. What goes to
        zero is the effect: see
        :meth:`recovered_citations`.
        """
        return sum(mapping.orphan_chars for mapping in self.mappings)

    @property
    def recovered_citations(self) -> tuple[str, ...]:
        """Steps that only reached the evidence because a truncation was repaired."""
        return tuple(dict.fromkeys(citation for mapping in self.mappings for citation in mapping.recovered))

    def block(self, citation: str) -> EvidenceBlock | None:
        return next((block for block in self.blocks if block.step_id == citation), None)

    def procedure(self, procedure_id: str) -> ProcedureRecord | None:
        return next((record for record in self.procedures if record.procedure_id == procedure_id), None)

    def render(self) -> str:
        """Every block, separated and numbered, for the prompt's EVIDENCE section."""
        return "\n\n".join(
            f"[evidence block {index} of {len(self.blocks)}]\n{block.render()}"
            for index, block in enumerate(self.blocks, start=1)
        )

    def render_procedure_context(self) -> str:
        """Procedure-level facts: scope, unbound forks, escalation, and identity.

        Separate from the blocks because these are properties of the procedure
        rather than of any step. Repeating them inside every block would spend
        several hundred prompt tokens saying the same thing eight times and would
        imply that a scope rule is a step precondition.
        """
        sections: list[str] = []
        for record in self.procedures:
            lines = [
                f"procedure_id: {record.procedure_id}",
                f"procedure_version: {record.version}",
                f"effective_date: {record.effective_date}",
                f"title: {record.title}",
            ]
            if record.title_ar:
                lines.append(f"title (العربية): {record.title_ar}")
            lines += [f"  - {item}" for item in record.gate_conditions]
            lines += [f"  - {item}" for item in record.escalation]
            lines += [f"  - required_information: {item}" for item in record.required_information]
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def select_procedures(
    result: SearchResult,
    *,
    conflict_score: float,
) -> tuple[ProcedureHit, ...]:
    """The retrieved procedures whose steps may be rebuilt into evidence.

    The top hit, plus any procedure that scores at least *conflict_score* and
    shares a scenario tag with it. That second clause is not generosity: two
    procedures covering the same scenario is the retrieval-stage shadow of a
    policy conflict, and
    :func:`~agentpath.evidence.detect_scenario_overlap` already surfaces it as a
    notice. Building evidence from only the higher-scoring one would be exactly
    the silent resolution the specification forbids - the answer layer has to be
    able to cite either procedure in order to present both.

    An unrelated second procedure is *not* included. It is prompt tokens the
    answer does not need and an invitation to cite across procedures.
    """
    if not result.hits:
        return ()
    top = result.hits[0]
    selected = [top]
    top_tags = set(top.scenario_tags)
    for hit in result.hits[1:]:
        if hit.score >= conflict_score and top_tags & set(hit.scenario_tags):
            selected.append(hit)
    return tuple(selected)


def assemble_evidence(
    result: SearchResult,
    catalogue: ProcedureCatalogue,
    *,
    conflict_score: float,
    max_blocks: int = MAX_EVIDENCE_BLOCKS,
) -> AssembledEvidence:
    """Turn a search result into whole, structured evidence blocks.

    The only place raw chunk text is read is :func:`map_chunk`, and all it takes
    out is a set of step ids. Everything that reaches the model afterwards comes
    from the catalogue.
    """
    hits = select_procedures(result, conflict_score=conflict_score)
    records: list[ProcedureRecord] = []
    unknown: list[str] = []
    mappings: list[ChunkMapping] = []
    #: citation -> (best passage score that pointed here, chunk ids, hit rank)
    candidates: dict[str, tuple[float, list[str], int]] = {}

    for rank, hit in enumerate(hits):
        record = catalogue.procedure(hit.procedure_id)
        if record is None:
            unknown.append(hit.procedure_id)
            continue
        records.append(record)
        for passage in hit.passages:
            mapping = map_chunk(passage, record, catalogue)
            mappings.append(mapping)
            for citation in mapping.citations:
                score, chunks, existing_rank = candidates.get(citation, (0.0, [], rank))
                chunks.append(passage.chunk_id)
                candidates[citation] = (max(score, passage.score), chunks, min(existing_rank, rank))

    # Rank by the best evidence that pointed at each step, then cap, then restore
    # execution order for presentation. Capping on score rather than on order
    # matters: if the cap bites, the step the query actually matched should
    # survive, not whichever step happens to be numbered lowest.
    ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], item[1][2], item[0]))
    kept = ordered[:max_blocks]
    dropped = len(ordered) - len(kept)

    blocks: list[EvidenceBlock] = []
    gate_used: set[str] = set()
    for citation, (_score, chunks, _rank) in sorted(
        kept,
        key=lambda item: (item[1][2], catalogue.step(item[0]).step_order if catalogue.step(item[0]) else 0),
    ):
        step = catalogue.step(citation)
        if step is None:  # pragma: no cover - candidates only ever hold known steps
            continue
        record = catalogue.procedure(step.procedure_id)
        conditions = list(step.conditions)
        if record is not None and record.procedure_id not in gate_used:
            # The procedure's scope and its unbound forks ride on the earliest
            # step of that procedure in this evidence, so they are stated once.
            gate_used.add(record.procedure_id)
            conditions = [*record.gate_conditions, *conditions]
        blocks.append(
            EvidenceBlock(
                procedure_id=step.procedure_id,
                procedure_version=step.procedure_version,
                step_id=step.citation,
                instruction=step.instruction,
                conditions=tuple(conditions),
                control_name=step.control_name,
                screenshot_id=step.screenshot_id,
                source_section=step.source_section,
                instruction_ar=step.instruction_ar,
                step_order=step.step_order,
                from_chunks=tuple(dict.fromkeys(chunks)),
            )
        )

    assembled = AssembledEvidence(
        blocks=tuple(blocks),
        procedures=tuple(records),
        mappings=tuple(mappings),
        unknown_procedures=tuple(dict.fromkeys(unknown)),
        dropped_blocks=dropped,
    )
    logger.info(
        "evidence.assembled",
        procedures=list(assembled.procedure_ids),
        blocks=len(assembled.blocks),
        citations=list(assembled.citations),
        recovered=list(assembled.recovered_citations),
        orphan_chars=assembled.orphan_chars,
        dropped=dropped,
        unknown_procedures=list(assembled.unknown_procedures),
    )
    return assembled


def clarifying_questions(evidence: AssembledEvidence, *, conflicts: Sequence[str] = ()) -> tuple[str, ...]:
    """Focused questions, composed only from structured data.

    Used when the evidence is partial or when two procedures cover the same
    scenario: the specification says ask, not guess. Every question is built from
    a field in the corpus - ``required_information``, a decision branch's
    ``condition``, or the conflicting procedures' own ids and versions - so a
    clarification cannot smuggle in an invented requirement.
    """
    questions: list[str] = []
    questions.extend(conflicts)

    # An unbound decision branch is the most decisive thing to ask about: it
    # changes which path the agent takes, so guessing it wrong wastes the whole
    # answer. Asked before the missing-information items for that reason.
    for record in evidence.procedures:
        for condition in record.unbound_conditions:
            question = _branch_question(condition)
            if question is not None:
                questions.append(question)

    for block in evidence.blocks:
        for condition in block.conditions:
            question = _branch_question(condition)
            if question is not None:
                questions.append(question)

    for record in evidence.procedures:
        for item in record.required_information:
            questions.append(_information_question(item))

    # Capped, because the specification asks for *a focused* question and an agent
    # handed eleven of them at once is being interrogated rather than helped. The
    # ordering above is the priority: a conflict first, then the fork that changes
    # the path, then the fact most likely to be missing.
    return tuple(dict.fromkeys(questions))[:MAX_CLARIFYING_QUESTIONS]


def _information_question(item: str) -> str:
    """Turn a ``required_information`` entry into a grammatical question.

    ``"What is the ..."`` does not fit every entry. The corpus writes some of them
    as yes/no facts - ``"Whether the guest is physically at the property right
    now"`` - and the naive template produced *"what is the whether the guest is
    physically at the property right now?"*, which is the kind of sentence that
    makes an agent stop trusting the tool. Those become a confirmation instead.
    """
    text = item.strip().rstrip("?.")
    if not text:
        return ""
    lowered = text[0].lower() + text[1:]
    if lowered.startswith(("whether ", "if ")):
        return f"Please confirm {lowered}."
    return f"What is the {lowered}?"


def _branch_question(condition: str) -> str | None:
    """Turn a rendered decision branch into a yes/no question, or ``None``.

    Only the ``If ... then ...`` shape carries a testable condition; anything else
    is prose that would make a confusing question, so it is skipped rather than
    reworded into something the corpus did not say.
    """
    prefix = "decision_branch: If "
    if not condition.startswith(prefix):
        return None
    clause = condition[len(prefix) :].split(" then ", 1)[0].strip().rstrip(".")
    if not clause:
        return None
    return f"Does this apply: {clause[0].lower() + clause[1:]}?"


def evidence_fields() -> tuple[str, ...]:
    """The exact field set an evidence block carries.

    Named here so a test can assert the contract rather than reading the
    dataclass, and so the prompt and the contract cannot drift.
    """
    return (
        "procedure_id",
        "procedure_version",
        "step_id",
        "instruction",
        "conditions",
        "control_name",
        "screenshot_id",
        "source_section",
    )


__all__ = [
    "MAX_EVIDENCE_BLOCKS",
    "AssembledEvidence",
    "ChunkMapping",
    "EvidenceBlock",
    "assemble_evidence",
    "citation_positions",
    "clarifying_questions",
    "evidence_fields",
    "leading_orphan",
    "map_chunk",
    "match_step",
    "select_procedures",
]

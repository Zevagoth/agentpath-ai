"""The structured procedure source, resolved once at startup.

Why this module exists
---------------------
T7 measured four unsupported claims in the generated answers and all four were
the same fabricated token, ``NW-HTL-014#S10``. The cause was not the model. It
was the evidence: RAGFlow chunks on size rather than on step boundaries, so a
chunk began in the middle of step 6 with its ``### Step 6`` heading and its
``**Step ID:** `NW-HTL-014#S06``` line cut away. The first step token the model
could see belonged to step *7*, so it paired the orphaned instruction with S07
and shifted every citation after it by one, running off the end of the
procedure at S10.

Raw retrieved text therefore cannot be the generation evidence. RAGFlow keeps
the job it is good at - deciding which procedure and which steps are relevant -
and the text the model reads is rebuilt here, from the structured source, so
every step arrives whole. A step either exists in this catalogue or it cannot be
cited at all.

What is loaded, and when
------------------------
:meth:`ProcedureCatalogue.load` reads ``data/synthetic/procedures/*.yaml`` once,
from :func:`~agentpath.api.deps.build_context`, at process start. Five files,
~10 KiB each. After that, assembling evidence for a request is dictionary
lookups: no YAML parsing, no filesystem access and no validation on the request
path. That is the same bargain :mod:`agentpath.screenshots` already makes for
image sidecars, and for the same reason.

The committed YAML is the source rather than a build artefact. ``make dataset``
writes rendered Markdown and a manifest into ``data/synthetic/build/``, which is
gitignored, so a catalogue read from there would make the API's behaviour depend
on whether someone had run a build - and would let a stale artefact disagree
with the corpus the tests assert on. The YAML is committed, is what every other
consumer treats as authoritative, and is what a reviewer reads.

Duplicated field names, on purpose
----------------------------------
The field names read below are also declared in
``agentpath_ingestion.schema``, and the validation here is a subset of that
model's. Importing it would be the DRY move and it is the wrong one:
``tests/test_visual_steps.py::test_the_api_does_not_import_the_ingestion_package``
asserts on the import graph that ``agentpath.api.app`` pulls in zero
``agentpath_ingestion`` modules, because that is what guarantees YAML loading and
image hashing cannot run during a request. :mod:`agentpath.screenshots` already
pays this price for sidecars and documents why.

This module beats that precedent in one respect. The screenshot catalogue is
kept honest only indirectly, by a rename on the ingestion side making a field
look missing. Here the two readers of the same files are compared directly:
``tests/test_procedure_catalogue.py`` loads the corpus through
``agentpath_ingestion.dataset.load_procedures`` *and* through this module and
asserts every step, id, version, control name and screenshot reference agrees.
A divergence is a test failure, not a silently degraded catalogue.

A strict subset is read. ``synonyms``, ``do``, ``do_not``, ``exceptions``,
``related_procedures``, ``owner_team``, ``severity`` and ``product_line`` are
not read: nothing in the answer path needs them, and a field that is never
loaded cannot be served.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from agentpath.paths import procedures_dir as default_procedures_dir

logger = structlog.get_logger(__name__)

#: Mirrors ``agentpath_ingestion.schema.PROCEDURE_ID_PATTERN``.
PROCEDURE_ID_PATTERN = re.compile(r"^NW-[A-Z]{3}-\d{3}$")

#: Mirrors ``agentpath_ingestion.schema.STEP_ID_PATTERN``.
STEP_ID_PATTERN = re.compile(r"^S\d{2,}$")

#: Mirrors ``agentpath_ingestion.schema.STEP_CITATION_SEPARATOR``.
STEP_CITATION_SEPARATOR = "#"

#: A citation token as it appears in rendered documents and in visual chunks.
#: Anchored nowhere, because it is used both to validate one token and to find
#: every token inside a retrieved chunk.
CITATION_TOKEN_PATTERN = re.compile(r"NW-[A-Z]{3}-\d{3}#S\d{2,}")

#: Refuse to hold an implausibly large corpus in memory. The five committed
#: procedures are 8-11 KiB each.
MAX_PROCEDURE_BYTES = 512 * 1024


def qualified_step_id(procedure_id: str, step_id: str) -> str:
    """The citation token for one step, e.g. ``NW-HTL-014#S01``.

    Mirrors ``agentpath_ingestion.schema.qualified_step_id``. ``S01`` alone is
    ambiguous - every procedure has one - so the token names the procedure too.
    """
    return f"{procedure_id}{STEP_CITATION_SEPARATOR}{step_id}"


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One procedure step, whole, as the generation evidence will carry it.

    "Whole" is the entire point. This record is built from the YAML and never
    from retrieved text, so no chunk boundary can remove its id, truncate its
    instruction, or separate it from the control it names.
    """

    procedure_id: str
    procedure_version: str
    step_id: str
    step_order: int
    instruction: str
    instruction_ar: str | None = None
    expected_result: str | None = None
    screen_name: str | None = None
    #: The authoritative name of the button or control the agent operates, taken
    #: from the step's ``ui_element``. Nothing in this system parses a control
    #: name out of instruction prose; if a label is wanted, it is read from here.
    control_name: str | None = None
    screenshot_id: str | None = None
    #: Decision branches that route into this step, rendered from
    #: ``decision_branches`` entries whose ``goto_step`` is this step's number.
    #: Structured data only - see :meth:`ProcedureRecord.gate_conditions` for the
    #: procedure-level half.
    conditions: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        """The token an answer cites for this step, e.g. ``NW-HTL-014#S07``."""
        return qualified_step_id(self.procedure_id, self.step_id)

    @property
    def source_section(self) -> str:
        """Where a human finds this text in the source document.

        The heading path of the rendered Markdown, which is what someone
        verifying an answer actually navigates: open
        ``data/synthetic/build/NW-HTL-014.md``, find ``## Steps``, then find
        ``### Step 7``. Both headings are written verbatim by
        ``agentpath_ingestion.render``, and
        ``tests/test_procedure_catalogue.py`` asserts they occur in the rendered
        document, so this cannot drift into naming a section that does not
        exist.

        A line number would be more precise and would rot on the next edit; a
        chunk id would name a RAGFlow object that no reviewer can open.
        """
        return f"{self.procedure_id}.md > ## Steps > ### Step {self.step_order}"

    @property
    def match_text(self) -> str:
        """Every word this step contributes, for matching truncated chunk text.

        Both language sides plus the expected result, because a chunk can be cut
        anywhere: the orphaned fragment left at the head of a chunk may be the
        English instruction, the Arabic one, or only the expected-result line.
        """
        return "\n".join(part for part in (self.instruction, self.instruction_ar, self.expected_result) if part)


@dataclass(frozen=True, slots=True)
class ProcedureRecord:
    """One procedure: its identity, its gate conditions, and its steps."""

    procedure_id: str
    version: str
    title: str
    title_ar: str
    effective_date: str
    scenario_tags: tuple[str, ...]
    applies_when: tuple[str, ...]
    does_not_apply_when: tuple[str, ...]
    required_information: tuple[str, ...]
    #: Decision branches with no ``goto_step``: they fork the procedure without
    #: naming a step to continue at, so they belong to the procedure rather than
    #: to any one step.
    unbound_conditions: tuple[str, ...]
    escalation: tuple[str, ...]
    steps: Mapping[str, StepRecord] = field(default_factory=dict)

    @property
    def gate_conditions(self) -> tuple[str, ...]:
        """The conditions that govern the procedure as a whole.

        Labelled with the structured field each line came from, so a reader - and
        a reviewer of a generated answer - can tell a scope rule from a fork, and
        can find the line in the YAML. Nothing here is composed or inferred.
        """
        lines = [f"applies_when: {item}" for item in self.applies_when]
        lines += [f"does_not_apply_when: {item}" for item in self.does_not_apply_when]
        lines += list(self.unbound_conditions)
        return tuple(lines)

    @property
    def ordered_steps(self) -> tuple[StepRecord, ...]:
        return tuple(sorted(self.steps.values(), key=lambda step: step.step_order))

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(step.citation for step in self.ordered_steps)

    def step(self, step_id: str) -> StepRecord | None:
        """Look up a step by its bare id (``S07``) or its citation token."""
        if step_id in self.steps:
            return self.steps[step_id]
        prefix = f"{self.procedure_id}{STEP_CITATION_SEPARATOR}"
        if step_id.startswith(prefix):
            return self.steps.get(step_id[len(prefix) :])
        return None


@dataclass(frozen=True, slots=True)
class ProcedureCatalogue:
    """Every procedure the answer path can build evidence from, keyed by id.

    Immutable after construction. ``problems`` records files that were rejected,
    so a procedure missing from the catalogue is reported rather than looking
    like a procedure that was never written.
    """

    procedures: Mapping[str, ProcedureRecord] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> ProcedureCatalogue:
        """A catalogue with no procedures.

        The default for :class:`~agentpath.api.deps.ServiceContext`, so a test or
        a harness that only exercises search gets a service where every citation
        is unknown rather than one that reads the filesystem.
        """
        return cls(procedures={})

    @classmethod
    def load(cls, directory: Path | None = None) -> ProcedureCatalogue:
        """Read and validate every procedure in *directory*.

        A file that cannot be trusted is skipped and recorded in
        :attr:`problems` rather than raised: one malformed procedure should
        degrade to "that procedure cannot be answered from" rather than to "the
        API will not start". Ingestion is the strict gate - it refuses to render
        or upload a corpus that does not validate - so a file that is bad here
        was never going to be retrievable either.
        """
        base = directory if directory is not None else default_procedures_dir()
        records: dict[str, ProcedureRecord] = {}
        problems: list[str] = []

        if not base.is_dir():
            # Named by the directory's own name only: the configured location can
            # be absolute, and this string reaches log lines.
            problems.append(f"procedure directory {base.name!r} does not exist")
            return cls(procedures={}, problems=tuple(problems))

        for path in sorted(base.glob("*.yaml")):
            record, failures = _load_one(path)
            if record is not None:
                if record.procedure_id in records:
                    problems.append(f"{path.name} declares procedure_id {record.procedure_id}, which is already loaded")
                    continue
                records[record.procedure_id] = record
            problems.extend(failures)

        logger.info(
            "procedures.loaded",
            count=len(records),
            steps=sum(len(record.steps) for record in records.values()),
            problems=len(problems),
        )
        for problem in problems:
            logger.warning("procedures.unusable", detail=problem)
        return cls(procedures=dict(sorted(records.items())), problems=tuple(problems))

    # -- lookup ------------------------------------------------------------ #

    def procedure(self, procedure_id: object) -> ProcedureRecord | None:
        """The procedure for *procedure_id*, or ``None``.

        Total, like :meth:`agentpath.screenshots.ScreenshotCatalogue.resolve`:
        the argument can come from retrieved metadata, so a malformed value is an
        expected input rather than an error.
        """
        if not isinstance(procedure_id, str) or not PROCEDURE_ID_PATTERN.fullmatch(procedure_id):
            return None
        return self.procedures.get(procedure_id)

    def step(self, citation: object) -> StepRecord | None:
        """The step a citation token names, or ``None`` if no such step exists.

        This is the belt to the schema's braces. Constraining the model's
        ``step_id`` field with a JSON-Schema ``enum`` makes a nonexistent
        citation unreachable during decoding, but a prompt change, a schema
        change or a future model that ignores the grammar would all bypass that,
        and this lookup would still catch it.
        """
        if not isinstance(citation, str):
            return None
        token = citation.strip()
        if not CITATION_TOKEN_PATTERN.fullmatch(token):
            return None
        procedure_id, _, step_id = token.partition(STEP_CITATION_SEPARATOR)
        record = self.procedures.get(procedure_id)
        return record.steps.get(step_id) if record is not None else None

    def steps_for(self, procedure_id: str) -> tuple[StepRecord, ...]:
        record = self.procedure(procedure_id)
        return record.ordered_steps if record is not None else ()

    # -- reporting --------------------------------------------------------- #

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.procedures)

    @property
    def step_count(self) -> int:
        return sum(len(record.steps) for record in self.procedures.values())

    def __len__(self) -> int:
        return len(self.procedures)

    def __iter__(self) -> Iterator[ProcedureRecord]:
        return iter(self.procedures.values())

    def __contains__(self, procedure_id: object) -> bool:
        return self.procedure(procedure_id) is not None


# --------------------------------------------------------------------------- #
# Loading one file
# --------------------------------------------------------------------------- #


def _load_one(path: Path) -> tuple[ProcedureRecord | None, list[str]]:
    """Build one record, or say why it cannot be built."""
    name = path.name
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, [f"procedure {name} cannot be read: {exc.strerror or exc.__class__.__name__}"]
    if size > MAX_PROCEDURE_BYTES:
        return None, [f"procedure {name} is {size} bytes, above the {MAX_PROCEDURE_BYTES} byte limit"]

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"procedure {name} is unreadable: {exc}"]
    if not isinstance(raw, dict):
        return None, [f"procedure {name} must contain a YAML mapping"]

    procedure_id = raw.get("procedure_id")
    if not isinstance(procedure_id, str) or not PROCEDURE_ID_PATTERN.fullmatch(procedure_id):
        return None, [f"procedure {name} declares an unusable procedure_id {procedure_id!r}"]
    if procedure_id != path.stem:
        # Otherwise a file could register itself under another procedure's id and
        # shadow the real one.
        return None, [f"procedure {name} declares procedure_id {procedure_id!r}, which is not its own name"]

    version = str(raw.get("version") or "")
    if not re.fullmatch(r"\d+\.\d+", version):
        return None, [f"procedure {name} declares an unusable version {version!r}"]

    steps, failures = _load_steps(raw.get("steps"), procedure_id=procedure_id, version=version, name=name)
    if failures:
        return None, failures
    if not steps:
        return None, [f"procedure {name} declares no usable steps"]

    branches = _branch_lines(raw.get("decision_branches"))
    bound: dict[int, list[str]] = {}
    unbound: list[str] = []
    for goto_step, line in branches:
        if goto_step is None:
            unbound.append(line)
        else:
            bound.setdefault(goto_step, []).append(line)

    record = ProcedureRecord(
        procedure_id=procedure_id,
        version=version,
        title=str(raw.get("title") or ""),
        title_ar=str(raw.get("title_ar") or ""),
        effective_date=str(raw.get("effective_date") or ""),
        scenario_tags=_strings(raw.get("scenario_tags")),
        applies_when=_strings(raw.get("applies_when")),
        does_not_apply_when=_strings(raw.get("does_not_apply_when")),
        required_information=_strings(raw.get("required_information")),
        unbound_conditions=tuple(unbound),
        escalation=_escalation_lines(raw.get("escalation")),
        steps={
            step.step_id: StepRecord(
                procedure_id=step.procedure_id,
                procedure_version=step.procedure_version,
                step_id=step.step_id,
                step_order=step.step_order,
                instruction=step.instruction,
                instruction_ar=step.instruction_ar,
                expected_result=step.expected_result,
                screen_name=step.screen_name,
                control_name=step.control_name,
                screenshot_id=step.screenshot_id,
                conditions=tuple(bound.get(step.step_order, ())),
            )
            for step in steps
        },
    )
    return record, []


def _load_steps(
    raw: object,
    *,
    procedure_id: str,
    version: str,
    name: str,
) -> tuple[list[StepRecord], list[str]]:
    """Read the step list, or report every problem in it at once."""
    if not isinstance(raw, list) or not raw:
        return [], [f"procedure {name} declares no steps"]

    records: list[StepRecord] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            problems.append(f"procedure {name} step {index} is not a mapping")
            continue
        number = entry.get("number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            problems.append(f"procedure {name} step {index} declares an unusable number {number!r}")
            continue
        # Mirrors the ingestion schema's default: the id is derived from the step
        # number unless the YAML pinned one, so renumbering cannot silently
        # repoint a citation that has already been given to an agent.
        step_id = str(entry.get("step_id") or "") or f"S{number:02d}"
        if not STEP_ID_PATTERN.fullmatch(step_id):
            problems.append(f"procedure {name} step {number} declares an unusable step_id {step_id!r}")
            continue
        if step_id in seen:
            problems.append(f"procedure {name} repeats step_id {step_id!r}, so a citation would be ambiguous")
            continue
        seen.add(step_id)
        instruction = str(entry.get("instruction") or "").strip()
        if not instruction:
            problems.append(f"procedure {name} step {number} has no instruction")
            continue
        records.append(
            StepRecord(
                procedure_id=procedure_id,
                procedure_version=version,
                step_id=step_id,
                step_order=number,
                instruction=instruction,
                instruction_ar=str(entry.get("instruction_ar") or "").strip() or None,
                expected_result=str(entry.get("expected_result") or "").strip() or None,
                screen_name=str(entry.get("screen") or "").strip() or None,
                control_name=str(entry.get("ui_element") or "").strip() or None,
                screenshot_id=str(entry.get("screenshot") or "").strip() or None,
            )
        )
    return records, problems


def _branch_lines(raw: object) -> tuple[tuple[int | None, str], ...]:
    """Render ``decision_branches`` as labelled lines, keeping their targets.

    The wording mirrors the rendered document's own ``**If** ... **then** ...``
    shape so a reviewer reading the evidence and a reviewer reading the source
    see the same sentence.
    """
    if not isinstance(raw, list):
        return ()
    lines: list[tuple[int | None, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        condition = str(entry.get("condition") or "").strip()
        then = str(entry.get("then") or "").strip()
        if not condition or not then:
            continue
        text = f"decision_branch: If {condition} then {then}"
        otherwise = str(entry.get("otherwise") or "").strip()
        if otherwise:
            text += f" Otherwise {otherwise}"
        goto = entry.get("goto_step")
        target = goto if isinstance(goto, int) and not isinstance(goto, bool) else None
        if target is not None:
            text += f" (continue at step {target})"
        lines.append((target, text))
    return tuple(lines)


def _escalation_lines(raw: object) -> tuple[str, ...]:
    """The escalation rule, one labelled line per structured field.

    Carried because an answer may legitimately say "escalate", and every part of
    that sentence - the trigger, the queue, whether approval is needed, who
    approves - is a fact an answer must not invent.
    """
    if not isinstance(raw, dict):
        return ()
    lines: list[str] = []
    for key in ("trigger", "queue", "contact_channel", "response_target"):
        value = str(raw.get(key) or "").strip()
        if value:
            lines.append(f"escalation_{key}: {value}")
    approval = raw.get("approval_required")
    lines.append(f"escalation_approval_required: {'yes' if approval else 'no'}")
    approver = str(raw.get("approver_role") or "").strip()
    if approver:
        lines.append(f"escalation_approver_role: {approver}")
    return tuple(lines)


def _strings(raw: object) -> tuple[str, ...]:
    """A YAML list of strings, cleaned. Anything else is an empty tuple."""
    if not isinstance(raw, list):
        return ()
    return tuple(text for text in (str(item).strip() for item in raw if _is_scalar(item)) if text)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float) and not isinstance(value, bool)


def all_citations(procedures: Sequence[ProcedureRecord]) -> tuple[str, ...]:
    """Every citation token *procedures* can offer, in procedure and step order."""
    return tuple(citation for record in procedures for citation in record.citations)

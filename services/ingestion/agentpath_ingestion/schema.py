"""Schema for the synthetic procedure knowledge base.

``data/synthetic/procedures/*.yaml`` is the single source of truth for the
knowledge base. Everything downstream (rendered Markdown, RAGFlow chunks,
evaluation expectations, screenshot templates) is derived from these files, so
the schema is strict on purpose: a typo in a procedure ID or a missing
escalation rule should fail the build, not surface as a bad answer to an agent.

Every procedure describes the *fictional* online travel agency **Northwind
Travel** and its *fictional* agent console **AgentDesk**. No real company
policy, wording, or interface is reproduced anywhere in this dataset.
"""

from __future__ import annotations

import datetime as _dt
import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Visible on every rendered document and screenshot. Non-negotiable.
FICTIONAL_BANNER = "FICTIONAL — SYNTHETIC DEMO DATA"

#: ``NW`` (Northwind) + three-letter domain + three digits, e.g. ``NW-HTL-014``.
PROCEDURE_ID_PATTERN = re.compile(r"^NW-[A-Z]{3}-\d{3}$")

ProcedureId = Annotated[str, Field(pattern=PROCEDURE_ID_PATTERN.pattern)]

#: ``S`` + at least two digits, e.g. ``S01``. Procedure-scoped, not global.
STEP_ID_PATTERN = re.compile(r"^S\d{2,}$")

#: Separator between a procedure id and a step id in a citation token.
STEP_CITATION_SEPARATOR = "#"


def format_step_id(number: int) -> str:
    """Derive the default step id for step *number*.

    Zero-padded to two digits so ids sort lexicographically in the same order
    they are executed: ``S02`` before ``S10``. Steps are capped at 40, so two
    digits always suffice.
    """
    return f"S{number:02d}"


def qualified_step_id(procedure_id: str, step_id: str) -> str:
    """The citation token for one step, e.g. ``NW-HTL-014#S01``.

    ``step_id`` is deliberately procedure-scoped, because ``S01`` on its own is
    ambiguous: every procedure has one. The answer layer validates a model's
    citation by looking for it verbatim in the retrieved evidence, so the token
    it cites has to identify the procedure too — otherwise a citation invented
    against the wrong procedure would still be found in some chunk and pass.

    This is the only place the qualified form is spelled, so the renderer, the
    visual-step records and the answer layer cannot drift apart.
    """
    return f"{procedure_id}{STEP_CITATION_SEPARATOR}{step_id}"


class ProductLine(StrEnum):
    """Top-level filter offered in the UI and pushed into RAGFlow metadata."""

    HOTELS = "hotels"
    FLIGHTS = "flights"
    CANCELLATIONS = "cancellations"
    REFUNDS = "refunds"
    ESCALATIONS = "escalations"


class Severity(StrEnum):
    """How much room the agent has to act alone."""

    ROUTINE = "routine"
    TIME_CRITICAL = "time_critical"
    APPROVAL_REQUIRED = "approval_required"


class LocalizedText(BaseModel):
    """A string an agent may need in either language.

    English is required because retrieval, evaluation and the reranker all use
    it as the pivot language; Arabic is required wherever the text is read out
    to a customer.
    """

    model_config = ConfigDict(extra="forbid")

    en: str = Field(min_length=1)
    ar: str = Field(min_length=1)


class Synonyms(BaseModel):
    """Alternative phrasings agents actually type.

    This is the lexical half of hybrid search. "Property unable to accommodate a
    confirmed guest" is never what an agent types; "hotel has no room",
    "walked guest", "overbooking" and "الفندق ما عنده غرف" are.
    """

    model_config = ConfigDict(extra="forbid")

    en: list[str] = Field(min_length=3)
    ar: list[str] = Field(min_length=2)

    @field_validator("en", "ar")
    @classmethod
    def _no_blank_entries(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("synonym entries must not be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("synonym entries must be unique")
        return cleaned


class Step(BaseModel):
    """One numbered action in the procedure.

    ``screen`` and ``ui_element`` name the AgentDesk surface the agent is looking
    at. In Stage 2 these fields drive screenshot generation and let a retrieved
    step be shown next to its original image; in this session they are already
    useful retrieval signal (agents search for button names).

    ``step_id`` is the stable handle an answer cites. It defaults to ``S{number}``
    so the five existing procedures need no editing, but it can be pinned
    explicitly in YAML. That matters later: renumbering a procedure would
    otherwise silently repoint every citation that was already given to an agent.
    """

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=40)
    step_id: str = Field(
        default="",
        description="Procedure-scoped stable id, e.g. 'S01'. Derived from number when omitted.",
    )
    requires: tuple[str, ...] = Field(
        default=(),
        description=(
            "Steps that must be done before this one, as procedure-scoped ids, e.g. ['S05']. "
            "Declared only where the procedure genuinely depends on an earlier step - a numbered "
            "list is not a dependency graph, and claiming a dependency that is not real would "
            "force irrelevant steps into an answer."
        ),
    )
    instruction: str = Field(min_length=10)
    instruction_ar: str | None = Field(default=None, min_length=5)
    screen: str | None = Field(default=None, description="AgentDesk screen name, e.g. 'Booking Detail'")
    ui_element: str | None = Field(default=None, description="Button or control the agent uses")
    expected_result: str | None = Field(default=None, min_length=5)
    screenshot: str | None = Field(
        default=None,
        description="Screenshot template id, rendered in Stage 2. Absent for text-only steps.",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_step_id_from_number(cls, data: Any) -> Any:
        """Fill ``step_id`` from ``number`` unless the YAML pinned one.

        Runs before field validation so an explicit value always wins, and so
        the derivation is a property of the model rather than of the loader —
        a ``Step`` built in Python gets the same id as one read from YAML.
        """
        if not isinstance(data, dict) or data.get("step_id"):
            return data
        number = data.get("number")
        if not isinstance(number, int) or isinstance(number, bool):
            return data
        return {**data, "step_id": format_step_id(number)}

    @field_validator("step_id")
    @classmethod
    def _step_id_is_well_formed(cls, value: str) -> str:
        if not STEP_ID_PATTERN.fullmatch(value):
            raise ValueError(f"step_id {value!r} must look like 'S01' (S followed by at least two digits)")
        return value


class DecisionBranch(BaseModel):
    """An explicit fork in the procedure.

    Kept structured rather than buried in prose so the answer layer can surface
    "which branch applies" as a clarification question instead of guessing.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=5)
    then: str = Field(min_length=5)
    otherwise: str | None = Field(default=None, min_length=5)
    goto_step: int | None = Field(default=None, ge=1)


class Escalation(BaseModel):
    """Who takes over, when, and with what authority."""

    model_config = ConfigDict(extra="forbid")

    trigger: str = Field(min_length=5)
    queue: str = Field(min_length=3, description="Fictional Northwind Travel queue name")
    contact_channel: str = Field(min_length=3)
    response_target: str = Field(min_length=3, description="Fictional internal response target")
    approval_required: bool = False
    approver_role: str | None = None

    @model_validator(mode="after")
    def _approver_present_when_required(self) -> Escalation:
        if self.approval_required and not self.approver_role:
            raise ValueError("approver_role is required when approval_required is true")
        return self


class Procedure(BaseModel):
    """A single knowledge-base article.

    Field order mirrors how an agent reads it under pressure: what is this, does
    it apply to me, what do I do, what must I not do, when do I hand it off,
    what do I tell the customer, and how current is it.
    """

    model_config = ConfigDict(extra="forbid")

    procedure_id: ProcedureId
    title: str = Field(min_length=10)
    title_ar: str = Field(min_length=5)
    product_line: ProductLine
    scenario_tags: list[str] = Field(min_length=1)
    severity: Severity = Severity.ROUTINE
    summary: LocalizedText
    synonyms: Synonyms
    applies_when: list[str] = Field(min_length=1)
    does_not_apply_when: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(min_length=3)
    decision_branches: list[DecisionBranch] = Field(default_factory=list)
    do: list[str] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    escalation: Escalation
    customer_wording: LocalizedText
    required_information: list[str] = Field(
        default_factory=list,
        description="Facts the agent must collect; drives clarification questions.",
    )
    related_procedures: list[str] = Field(default_factory=list)
    version: str = Field(pattern=r"^\d+\.\d+$")
    effective_date: _dt.date
    owner_team: str = Field(min_length=3)

    @field_validator("scenario_tags")
    @classmethod
    def _tags_are_snake_case(cls, values: list[str]) -> list[str]:
        for value in values:
            if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value):
                raise ValueError(f"scenario tag {value!r} must be lower_snake_case")
        return values

    @field_validator("related_procedures")
    @classmethod
    def _related_ids_are_well_formed(cls, values: list[str]) -> list[str]:
        for value in values:
            if not PROCEDURE_ID_PATTERN.match(value):
                raise ValueError(f"related procedure id {value!r} is malformed")
        return values

    @model_validator(mode="after")
    def _steps_are_sequential(self) -> Procedure:
        numbers = [step.number for step in self.steps]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"steps must be numbered 1..n without gaps, got {numbers}")
        return self

    @model_validator(mode="after")
    def _step_ids_are_unique(self) -> Procedure:
        """Two steps sharing an id would make a citation unresolvable."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for step in self.steps:
            if step.step_id in seen:
                duplicates.append(step.step_id)
            seen.add(step.step_id)
        if duplicates:
            raise ValueError(f"step_id must be unique within a procedure, repeated: {', '.join(duplicates)}")
        return self

    @model_validator(mode="after")
    def _dependencies_are_earlier_steps(self) -> Procedure:
        """A ``requires`` entry must name an earlier step of this procedure.

        Two failures are worth catching at load time rather than at answer time. A
        dependency on an unknown id would silently never be satisfied, so the
        prerequisite it was meant to protect would go missing anyway. A dependency
        on a *later* step is a cycle in something that is performed in order, and
        the answer layer resolves these by walking the graph - a cycle there would
        either loop or be quietly truncated.
        """
        order = {step.step_id: step.number for step in self.steps}
        for step in self.steps:
            for required in step.requires:
                if required not in order:
                    raise ValueError(f"step {step.step_id} requires unknown step {required}")
                if order[required] >= step.number:
                    raise ValueError(
                        f"step {step.step_id} requires {required}, which is not an earlier step; "
                        "dependencies must point backwards"
                    )
        return self

    @model_validator(mode="after")
    def _branch_targets_exist(self) -> Procedure:
        valid = {step.number for step in self.steps}
        for branch in self.decision_branches:
            if branch.goto_step is not None and branch.goto_step not in valid:
                raise ValueError(f"decision branch points at unknown step {branch.goto_step}")
        return self

    @model_validator(mode="after")
    def _no_self_reference(self) -> Procedure:
        if self.procedure_id in self.related_procedures:
            raise ValueError("a procedure must not list itself as related")
        return self

    # -- convenience ------------------------------------------------------- #

    @property
    def all_synonyms(self) -> list[str]:
        """Every alternative phrasing, English first (pivot language)."""
        return [*self.synonyms.en, *self.synonyms.ar]

    @property
    def screenshot_steps(self) -> list[Step]:
        return [step for step in self.steps if step.screenshot]

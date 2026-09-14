"""Non-sensitive case facts the agent already knows, and what they may change.

An agent asking "the hotel walked our guest, what do I do" is usually looking at
a booking while they type. The system currently ignores that: it asks whether the
guest has arrived even when the console beside the agent already says so. This
module is the contract for handing those facts over.

**Design only, and deliberately so.** Nothing here talks to a booking system, and
nothing here is populated from one. Every field is supplied by the caller and
every test uses fabricated values. What a real integration would have to satisfy
is written down at the bottom of this docstring rather than left implicit.

What the context is allowed to do
---------------------------------
Three things, all of them narrowing or informing:

1. **Restrict.** ``product_type`` becomes the same product-line restriction the
   dropdown produces. It is a filter because the agent stated it.
2. **Select a branch.** The facts are given to the model as clearly-labelled case
   data so it can follow the fork the procedure already defines - "if the guest is
   at the property" is a branch in the corpus, and the context says whether that
   is true.
3. **Stop asking what is already known.** A clarifying question about the guest's
   location is wasted when ``customer_arrived`` was supplied, so answered facts are
   removed from the candidate questions.

What it must never do
---------------------
Override the procedure, or become policy. The corpus is the only source of what to
do; the context only says which part of it applies. A value like
``payment_status="refunded"`` does not license telling the customer a refund was
issued - it selects the branch the procedure defines for that case, and the
procedure's own words still carry the instruction. This is enforced by
construction rather than by instruction: the context never reaches the evidence
blocks, and the model's citations are still constrained to steps from the
catalogue.

Why there are no identifiers in here
------------------------------------
No name, no booking reference, no phone number, no payment detail, no email. Not
because the model would misuse them, but because a field that does not exist
cannot be logged, cached, sent to a model or leaked in an error message. The
fields below are the *operational shape* of a case - is it a hotel, has the guest
arrived, did the supplier refuse - which is what changes the procedure. Identity
does not change the procedure.

The consequence is honest and worth stating: some required information can never
be filled from context. ``Booking reference`` is a required input of
``NW-HTL-014`` and this contract will never hold it.

It follows that we must not *ask* for it either, and :data:`NEVER_ASKED` is that
rule. Asking an agent for a booking reference is useless and undesirable at the
same time: useless because they are reading the booking while they type, and
undesirable because ``extra="forbid"`` means the only field they could put it in
does not exist - so an agent who answered would be feeding an identifier to a
system built to reject it. Those items are reported in
:attr:`ContextUse.not_collected` with the reason, so the requirement is visibly
declined rather than quietly missing, and they are removed from both the missing
information and the clarifying questions.

Where a requirement is genuinely operational, the answer is to add a
non-sensitive field instead of asking for a sensitive one. ``Number of guests and
rooms on the booking`` is required by ``NW-HTL-014`` - it decides which
alternative accommodation qualifies - so :attr:`CaseContext.party_size` and
:attr:`CaseContext.rooms_booked` carry it as counts. A count identifies nobody.

Requirements for any future real integration
--------------------------------------------
Not implemented, and not to be implemented without explicit approval:

* an official, documented API - never scraping, never a shared database
* read-only scope
* company approval before any connection to a real environment
* authentication and authorisation on every call
* audit logging of what was read, by whom, for which case
* data minimisation: request the fields in this model and no others
* retention: nothing persisted beyond the request that used it
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BookingStatus(StrEnum):
    """Where the booking stands in AgentDesk.

    A closed set rather than free text because the corpus branches on these words:
    ``NW-HTL-014`` applies only when the status is ``confirmed``, and a typo would
    silently fail to match.
    """

    CONFIRMED = "confirmed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    MODIFIED = "modified"
    NO_SHOW = "no_show"
    COMPLETED = "completed"


class SupplierStatus(StrEnum):
    """What the supplier has said, if anything."""

    NOT_CONTACTED = "not_contacted"
    UNREACHABLE = "unreachable"
    CONFIRMED_AVAILABLE = "confirmed_available"
    CANNOT_ACCOMMODATE = "cannot_accommodate"
    NO_RECORD_OF_BOOKING = "no_record_of_booking"
    DISPUTED = "disputed"


class PaymentStatus(StrEnum):
    """Where the money is. Never an amount - only the stage."""

    NOT_PAID = "not_paid"
    PREPAID = "prepaid"
    PAY_AT_PROPERTY = "pay_at_property"
    REFUND_REQUESTED = "refund_requested"
    REFUND_ISSUED = "refund_issued"
    CHARGEBACK = "chargeback"


class CaseContext(BaseModel):
    """Operational facts about one case. No identifiers, no amounts, no free text
    that could carry them.

    ``extra="forbid"`` is the point of the model, not a detail. It is what makes
    "we do not accept personal data" a property of the code rather than a promise
    in a document: a client that sends ``guest_name`` gets a 422, and the field
    never exists to be logged. Extending the contract is a change here, reviewed,
    not something a caller can do by sending more keys.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_type: str | None = Field(
        default=None,
        description="Which product line the case is about. Applied as a search restriction.",
        examples=["hotels", "flights"],
    )
    booking_status: BookingStatus | None = Field(default=None, description="Booking status as AgentDesk shows it.")
    service_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of travel or stay, ISO 8601. A date, not a timestamp: time of day is not needed.",
        examples=["2026-04-16"],
    )
    supplier_status: SupplierStatus | None = Field(
        default=None, description="What the supplier has said, if they have been reached."
    )
    payment_status: PaymentStatus | None = Field(default=None, description="Payment stage. Never an amount.")
    customer_arrived: bool | None = Field(
        default=None,
        description="Whether the customer is physically at the property or airport now.",
    )
    party_size: int | None = Field(
        default=None,
        ge=1,
        le=99,
        description=(
            "How many guests are on the booking. A count, not a manifest: it changes which "
            "alternative accommodation qualifies, and it identifies nobody."
        ),
        examples=[2],
    )
    rooms_booked: int | None = Field(
        default=None,
        ge=1,
        le=99,
        description="How many rooms are on the booking. A count, for the same reason as party_size.",
        examples=[1],
    )
    actions_already_taken: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description=(
            "Steps the agent has already completed, as citation tokens such as "
            "'NW-HTL-014#S03'. Anything that is not a citation token is ignored rather "
            "than passed on, so this cannot become a channel for free text."
        ),
    )
    error_code: str | None = Field(
        default=None,
        max_length=40,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="An error or rejection code the console displayed.",
        examples=["SUPPLIER_TIMEOUT"],
    )
    market: str | None = Field(
        default=None,
        max_length=8,
        pattern=r"^[A-Za-z]{2}(-[A-Za-z]{2})?$",
        description="Market or country code, e.g. 'JO' or 'en-GB'. Not a customer address.",
        examples=["JO"],
    )
    language: str | None = Field(
        default=None,
        pattern=r"^(ar|en)$",
        description="Preferred answer language, when the agent wants to override detection.",
    )

    @property
    def is_empty(self) -> bool:
        return not self.provided_fields

    @property
    def provided_fields(self) -> tuple[str, ...]:
        """Names of the fields the caller actually set, in declaration order."""
        return tuple(name for name in self.__class__.model_fields if _is_set(getattr(self, name)))

    @property
    def completed_steps(self) -> tuple[str, ...]:
        """``actions_already_taken`` filtered to well-formed citation tokens.

        Filtered rather than validated-and-rejected: an agent's tooling might send
        a free-text note here, and dropping it is better than failing the request
        or forwarding prose into the prompt.
        """
        return tuple(value for value in self.actions_already_taken if CITATION_TOKEN.fullmatch(value))


#: A qualified step id, e.g. ``NW-HTL-014#S07``. Mirrors the pattern in
#: :mod:`agentpath.procedures`; re-stated rather than imported to keep this module
#: free of dependencies, since it is validated at the API edge.
CITATION_TOKEN = re.compile(r"^NW-[A-Z]{3}-\d{3}#S\d{2,}$")


def _is_set(value: object) -> bool:
    return value is not None and value != ()


# --------------------------------------------------------------------------- #
# What the context answers, and what it does not
# --------------------------------------------------------------------------- #

#: Which required-information items each context field can answer.
#:
#: Keyword matching, and deliberately conservative: a field claims an item only
#: when the wording clearly refers to the same fact. Over-claiming is the
#: dangerous direction - it would suppress a question the agent needed to be
#: asked - so an item that is only loosely related stays unanswered and gets
#: asked.
#:
#: Items this contract refuses are handled separately, by :data:`NEVER_ASKED`:
#: they are neither answered here nor asked, because asking for something we
#: designed the model to reject is a question with no valid reply.
ANSWERS: Mapping[str, tuple[str, ...]] = {
    "customer_arrived": ("physically at the property", "guest is at the property", "at the property now"),
    "booking_status": ("booking status", "status in agentdesk"),
    "service_date": ("nights of the stay are affected", "affected nights", "date of travel"),
    "payment_status": ("refund destination", "refund stage"),
    "error_code": ("error code", "rejection code"),
    "party_size": ("number of guests",),
    "rooms_booked": ("rooms on the booking", "number of rooms"),
}
# `supplier_status` deliberately claims nothing. It used to claim "Name and role of
# the property contact who refused" and "Which system the property searched", and
# both were over-claims: `cannot_accommodate` records *that* the supplier refused,
# not *who* refused, and `no_record_of_booking` records that they found nothing,
# not which system they looked in. Suppressing those questions meant an agent was
# never prompted for evidence the procedure requires. The field is still given to
# the model as branch data; it simply no longer answers a requirement.
#
# `payment_status` no longer claims the bare word "payment" either: it made
# "Original payment method and last four digits" look answered by a status enum.

#: Required information this system will not ask an agent for, and why.
#:
#: The reason is not that the information is unimportant - the procedure genuinely
#: requires it - but that the question is useless and undesirable at the same time.
#: Useless because the agent is reading the booking while they type, so they are
#: not missing it. Undesirable because :class:`CaseContext` is built to reject
#: exactly these values: an agent who answered would be typing an identifier into
#: a system designed never to hold one, and the field they typed it into does not
#: exist. Asking anyway trains agents to supply data the contract then refuses.
#:
#: Matched on substrings of the requirement or question text so both the corpus's
#: wording and a model's own phrasing are caught. Reported in
#: :attr:`ContextUse.not_collected` rather than dropped silently, so the policy
#: stays visible to a reviewer.
NEVER_ASKED: Mapping[str, tuple[str, ...]] = {
    "an identifier the agent is already looking at, and that this contract does not accept": (
        "booking reference",
        "booking ref",
        "confirmation number",
    ),
    "a customer name, which this contract does not accept": (
        "guest name",
        "customer name",
        "name spelling",
    ),
    "payment card data, which this contract does not accept": (
        "last four digits",
        "card number",
    ),
}


def refused_reason(text: str) -> str | None:
    """Why *text* names something we will not ask for, or ``None``.

    Takes a requirement or a whole question, because both are checked: the corpus
    turns ``Booking reference`` into "What is the booking reference?", and a model
    can ask for it in wording of its own.
    """
    lowered = text.lower()
    for reason, phrases in NEVER_ASKED.items():
        if any(phrase in lowered for phrase in phrases):
            return reason
    return None


@dataclass(frozen=True, slots=True)
class ContextUse:
    """How the supplied context was actually used, for the response to show.

    Exists so the effect is visible rather than inferred. An agent who supplied
    ``customer_arrived`` and got no question about it should be able to see *why*,
    and a reviewer should be able to tell that the context narrowed the search
    rather than changed the advice.
    """

    provided: tuple[str, ...] = ()
    #: field name -> what it did, in words an agent can read.
    applied: Mapping[str, str] = field(default_factory=dict)
    #: Required information the context could not answer, so still worth asking.
    unanswered: tuple[str, ...] = ()
    #: Required information this system will not ask for, mapped to why. Reported
    #: rather than dropped: the procedure does require these, and a reviewer should
    #: see a deliberate policy instead of a requirement that quietly disappeared.
    not_collected: Mapping[str, str] = field(default_factory=dict)
    #: Steps the agent said they had already done, so the answer can acknowledge them.
    completed_steps: tuple[str, ...] = ()
    #: Product-line restriction derived from the context, if any.
    product_line: str | None = None

    @property
    def used_anything(self) -> bool:
        return bool(self.applied)


def answers_requirement(context: CaseContext, requirement: str) -> tuple[str, ...]:
    """The context fields that answer *requirement*, or ``()``.

    One place owns this mapping so the prompt, the clarifying question and the
    response's ``context_used`` cannot disagree about whether a fact is known.

    A requirement is answered only when **every** field it names has been
    supplied. "Number of guests and rooms on the booking" names two, and knowing
    the guest count while the room count is missing does not answer it - it would
    suppress the question while half the answer was still unknown. Returning the
    whole set rather than the first match is what makes that checkable.
    """
    lowered = requirement.lower()
    claimed = tuple(name for name, phrases in ANSWERS.items() if any(phrase in lowered for phrase in phrases))
    if not claimed:
        return ()
    if not all(_is_set(getattr(context, name, None)) for name in claimed):
        return ()
    return claimed


def describe_use(
    context: CaseContext | None,
    *,
    requirements: Sequence[str] = (),
    product_line_applied: str | None = None,
) -> ContextUse:
    """Summarise what the context did for this request.

    ``product_line_applied`` is passed in rather than recomputed: the request may
    also carry an explicit ``product_line``, and the caller is the only place that
    knows which one won.
    """
    # The refusal policy does not depend on what the caller supplied: an item we
    # will not accept is not missing information for the agent to go and fetch,
    # whether or not any context came with the request.
    not_collected = {item: reason for item in requirements if (reason := refused_reason(item)) is not None}
    askable = tuple(item for item in requirements if item not in not_collected)

    if context is None or context.is_empty:
        return ContextUse(unanswered=askable, not_collected=not_collected)

    applied: dict[str, str] = {}
    if product_line_applied:
        applied["product_type"] = f"restricted the search to the {product_line_applied} product line"

    unanswered: list[str] = []
    for requirement in askable:
        names = answers_requirement(context, requirement)
        if not names:
            unanswered.append(requirement)
            continue
        for name in names:
            applied.setdefault(name, f"already known, so it is not asked again: {requirement.lower()}")

    for name in (
        "booking_status",
        "supplier_status",
        "customer_arrived",
        "payment_status",
        "error_code",
        "party_size",
        "rooms_booked",
    ):
        if _is_set(getattr(context, name)) and name not in applied:
            applied[name] = "given to the model as case data, so it can follow the matching branch"

    if context.completed_steps:
        applied["actions_already_taken"] = "steps the agent has already completed, acknowledged in the answer"
    if context.language:
        applied["language"] = f"answer language forced to {context.language}"

    return ContextUse(
        provided=context.provided_fields,
        applied=applied,
        unanswered=tuple(unanswered),
        not_collected=not_collected,
        completed_steps=context.completed_steps,
        product_line=product_line_applied,
    )


def render_for_prompt(context: CaseContext | None) -> str:
    """The case facts as a labelled block, or an empty string.

    Rendered from named fields rather than dumped as JSON so the model reads facts
    rather than a data structure, and so nothing unexpected can appear in it: every
    line below is a field this module declares.

    The block is labelled as agent-supplied case data. It is not evidence and not
    policy: it says which situation applies, and the procedure still says what to
    do about it.
    """
    if context is None or context.is_empty:
        return ""

    lines: list[str] = []
    if context.product_type:
        lines.append(f"product type: {context.product_type}")
    if context.booking_status:
        lines.append(f"booking status: {context.booking_status.value}")
    if context.service_date:
        lines.append(f"service date: {context.service_date}")
    if context.supplier_status:
        lines.append(f"supplier status: {context.supplier_status.value.replace('_', ' ')}")
    if context.payment_status:
        lines.append(f"payment status: {context.payment_status.value.replace('_', ' ')}")
    if context.customer_arrived is not None:
        lines.append("customer is at the property now: " + ("yes" if context.customer_arrived else "no"))
    if context.party_size is not None:
        lines.append(f"guests on the booking: {context.party_size}")
    if context.rooms_booked is not None:
        lines.append(f"rooms on the booking: {context.rooms_booked}")
    if context.completed_steps:
        lines.append("steps the agent has already completed: " + ", ".join(context.completed_steps))
    if context.error_code:
        lines.append(f"error code shown in the console: {context.error_code}")
    if context.market:
        lines.append(f"market: {context.market}")
    return "\n".join(lines)


__all__ = [
    "ANSWERS",
    "CITATION_TOKEN",
    "NEVER_ASKED",
    "BookingStatus",
    "CaseContext",
    "ContextUse",
    "PaymentStatus",
    "SupplierStatus",
    "answers_requirement",
    "describe_use",
    "refused_reason",
    "render_for_prompt",
]

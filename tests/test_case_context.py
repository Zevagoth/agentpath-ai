"""The case-context contract: what it accepts, and what it refuses to accept.

Every value here is fabricated. Nothing in this suite touches a booking system,
and the contract is designed so that it could not carry customer data even if
something tried to put it there.

The tests are weighted towards refusal on purpose. The value of this model is not
the fields it has - those are easy - it is the fields it does not have and cannot
be given. A field that does not exist cannot be logged, cached, put in a prompt or
printed in a stack trace, so ``extra="forbid"`` is the actual privacy control and
is tested as one.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentpath.case_context import (
    BookingStatus,
    CaseContext,
    PaymentStatus,
    SupplierStatus,
    answers_requirement,
    describe_use,
    refused_reason,
    render_for_prompt,
)
from agentpath.procedures import ProcedureCatalogue

# A fabricated hotel case: guest at the desk, supplier has refused.
WALKED_GUEST = CaseContext(
    product_type="hotels",
    booking_status=BookingStatus.CONFIRMED,
    service_date="2026-04-16",
    supplier_status=SupplierStatus.CANNOT_ACCOMMODATE,
    payment_status=PaymentStatus.PREPAID,
    customer_arrived=True,
    actions_already_taken=("NW-HTL-014#S01", "NW-HTL-014#S02"),
    market="JO",
    language="en",
)


@pytest.fixture(scope="module")
def catalogue() -> ProcedureCatalogue:
    return ProcedureCatalogue.load()


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rejected",
    [
        {"guest_name": "A. Sample"},
        {"customer_name": "A. Sample"},
        {"booking_reference": "NW-8F42-6C19"},
        {"phone": "+000 555 0142"},
        {"phone_number": "+000 555 0142"},
        {"email": "guest@example.invalid"},
        {"card_last_four": "4242"},
        {"payment_amount": 120.0},
        {"refund_amount": 120.0},
        {"address": "1 Harbour Road"},
        {"passport_number": "X1234567"},
        {"notes": "guest was upset"},
    ],
)
def test_personal_and_financial_fields_are_refused(rejected: dict) -> None:
    """The privacy control is the schema, not a policy document.

    A caller that sends one of these gets a validation error, so the value never
    reaches a log line, a prompt or an exception repr. Free-text ``notes`` is in
    this list for the same reason: it is where personal data ends up when every
    structured field refuses it.
    """
    with pytest.raises(ValidationError):
        CaseContext(**rejected)


def test_the_status_vocabularies_are_closed() -> None:
    """The corpus branches on these words, so a typo must fail loudly.

    ``NW-HTL-014`` applies only when the booking is confirmed. A free-text status
    of ``"Confirmed "`` or ``"CONFIRMD"`` would silently fail to match the branch
    and the agent would never know why.
    """
    with pytest.raises(ValidationError):
        CaseContext(booking_status="CONFIRMD")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CaseContext(supplier_status="refused")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        CaseContext(payment_status="paid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"service_date": "16/04/2026"},
        {"service_date": "2026-04-16T14:30:00Z"},
        {"error_code": "supplier timeout; DROP TABLE bookings"},
        {"error_code": "x" * 41},
        {"market": "Jordan"},
        {"language": "fr"},
    ],
)
def test_loosely_shaped_values_are_refused(bad: dict) -> None:
    """Each pattern exists to stop a specific kind of smuggling.

    A timestamp instead of a date carries more precision than the procedure needs;
    an unconstrained ``error_code`` is a free-text field wearing a code's name.
    """
    with pytest.raises(ValidationError):
        CaseContext(**bad)


def test_the_context_is_frozen() -> None:
    """Nothing downstream may edit the case facts it was handed."""
    with pytest.raises(ValidationError):
        WALKED_GUEST.customer_arrived = False  # type: ignore[misc]


def test_free_text_in_actions_already_taken_is_dropped_not_forwarded() -> None:
    """An agent's tooling might send a note here; it must not reach the prompt.

    Filtered rather than rejected: failing the whole request over a stray note
    would be worse than ignoring it, and forwarding it would make this the free-text
    channel the rest of the model refuses to be.
    """
    context = CaseContext(actions_already_taken=("NW-HTL-014#S01", "called the guest, she is upset", "NW-HTL-014#S03"))
    assert context.completed_steps == ("NW-HTL-014#S01", "NW-HTL-014#S03")
    assert "upset" not in render_for_prompt(context)


# --------------------------------------------------------------------------- #
# What it accepts
# --------------------------------------------------------------------------- #


def test_an_empty_context_is_valid_and_inert() -> None:
    """The contract is optional, so absent must be a first-class state."""
    empty = CaseContext()
    assert empty.is_empty
    assert empty.provided_fields == ()
    assert render_for_prompt(empty) == ""
    assert render_for_prompt(None) == ""


def test_provided_fields_reports_only_what_was_set() -> None:
    context = CaseContext(product_type="hotels", customer_arrived=False)
    assert context.provided_fields == ("product_type", "customer_arrived")
    # False is a value, not an absence - the guest not being there changes the path.
    assert "customer_arrived" in context.provided_fields


def test_the_rendered_block_reads_as_facts_not_as_a_data_structure() -> None:
    rendered = render_for_prompt(WALKED_GUEST)

    assert "product type: hotels" in rendered
    assert "booking status: confirmed" in rendered
    assert "supplier status: cannot accommodate" in rendered
    assert "customer is at the property now: yes" in rendered
    assert "NW-HTL-014#S01" in rendered
    # No JSON, no field names the agent would not recognise.
    assert "{" not in rendered
    assert "actions_already_taken" not in rendered


def test_a_guest_who_is_not_present_is_rendered_explicitly() -> None:
    """ "No" has to be stated. Omitting it would read as "unknown" to the model."""
    rendered = render_for_prompt(CaseContext(customer_arrived=False))
    assert "customer is at the property now: no" in rendered


# --------------------------------------------------------------------------- #
# What it changes
# --------------------------------------------------------------------------- #


def test_a_known_fact_stops_the_question_being_asked_again(catalogue: ProcedureCatalogue) -> None:
    """The point of the contract: do not ask what the console already shows."""
    record = catalogue.procedure("NW-HTL-014")
    assert record is not None
    presence = next(item for item in record.required_information if "physically at the property" in item.lower())

    assert answers_requirement(WALKED_GUEST, presence) == ("customer_arrived",)
    assert answers_requirement(CaseContext(), presence) == ()

    use = describe_use(WALKED_GUEST, requirements=record.required_information)
    assert presence not in use.unanswered
    assert "customer_arrived" in use.applied


# --------------------------------------------------------------------------- #
# What it refuses to ask for
# --------------------------------------------------------------------------- #


def test_information_the_contract_refuses_is_never_asked_for(
    catalogue: ProcedureCatalogue,
) -> None:
    """A question the contract would reject the answer to is not a question.

    ``Booking reference`` is required by ``NW-HTL-014`` and rejected by
    :class:`CaseContext`, so asking an agent for it is a dead end: the field they
    would type it into does not exist. It is declined with a reason rather than
    listed as missing information.
    """
    record = catalogue.procedure("NW-HTL-014")
    assert record is not None
    reference = next(item for item in record.required_information if "booking reference" in item.lower())

    use = describe_use(WALKED_GUEST, requirements=record.required_information)

    assert reference not in use.unanswered
    assert not any("booking reference" in item.lower() for item in use.unanswered)
    # Declined, with the reason, rather than silently gone.
    assert reference in use.not_collected
    assert "does not accept" in use.not_collected[reference]


@pytest.mark.parametrize(
    "requirement",
    [
        "Booking reference",
        "Booking reference and supplier confirmation number",
        "Both booking references",
        "Exact guest name spelling as it appears on the booking",
        "Original payment method and last four digits, as shown on the booking",
        # A model's own phrasing, not the corpus's.
        "What is the guest's booking ref?",
        "Please confirm the card number on file.",
    ],
)
def test_every_refused_wording_is_recognised(requirement: str) -> None:
    assert refused_reason(requirement) is not None


@pytest.mark.parametrize(
    "requirement",
    [
        "Whether the guest is physically at the property right now",
        "Name and role of the property contact who refused",
        "Which nights of the stay are affected",
        "Number of guests and rooms on the booking",
        "Reason the customer is cancelling, in their own words",
        "Which booking the customer wants to keep",
        "Which system the property searched (property management system or channel inbox)",
    ],
)
def test_operational_requirements_stay_askable(requirement: str) -> None:
    """The filter removes identifiers, not everything inconvenient.

    "Name and role of the property contact who refused" names a person, but a
    supplier-side business contact the procedure requires as evidence of the
    refusal - and the agent records it in AgentDesk, not here. Suppressing it would
    remove a real requirement.
    """
    assert refused_reason(requirement) is None


def test_every_refused_item_is_declined_across_the_whole_corpus(
    catalogue: ProcedureCatalogue,
) -> None:
    """No procedure may leak a refused requirement into missing information."""
    for record in catalogue:
        use = describe_use(None, requirements=record.required_information)
        for item in use.unanswered:
            assert refused_reason(item) is None, f"{record.procedure_id} still asks for {item!r}"


def test_a_refused_item_is_declined_even_with_no_context_at_all(
    catalogue: ProcedureCatalogue,
) -> None:
    """The policy is ours, not a consequence of what the caller sent."""
    record = catalogue.procedure("NW-REF-018")
    assert record is not None

    use = describe_use(None, requirements=record.required_information)

    assert any("last four digits" in item.lower() for item in use.not_collected)
    assert not any("last four digits" in item.lower() for item in use.unanswered)


# --------------------------------------------------------------------------- #
# A non-sensitive field instead of a sensitive question
# --------------------------------------------------------------------------- #


def test_guest_and_room_counts_answer_the_requirement_that_needs_them(
    catalogue: ProcedureCatalogue,
) -> None:
    """The operational requirement is met by counts rather than by identity."""
    record = catalogue.procedure("NW-HTL-014")
    assert record is not None
    occupancy = next(item for item in record.required_information if "number of guests" in item.lower())

    context = CaseContext(party_size=3, rooms_booked=2)
    assert set(answers_requirement(context, occupancy)) == {"party_size", "rooms_booked"}

    use = describe_use(context, requirements=record.required_information)
    assert occupancy not in use.unanswered


def test_a_requirement_naming_two_facts_needs_both_before_it_is_answered(
    catalogue: ProcedureCatalogue,
) -> None:
    """Half an answer must not suppress the question.

    "Number of guests and rooms on the booking" names two counts. Knowing the guest
    count while the room count is missing does not answer it, and reporting it as
    answered would stop the agent being asked for the half we do not have.
    """
    record = catalogue.procedure("NW-HTL-014")
    assert record is not None
    occupancy = next(item for item in record.required_information if "number of guests" in item.lower())

    assert answers_requirement(CaseContext(party_size=3), occupancy) == ()
    assert answers_requirement(CaseContext(rooms_booked=2), occupancy) == ()
    assert occupancy in describe_use(CaseContext(party_size=3), requirements=(occupancy,)).unanswered


def test_the_counts_reach_the_model_as_readable_facts() -> None:
    rendered = render_for_prompt(CaseContext(party_size=4, rooms_booked=2))
    assert "guests on the booking: 4" in rendered
    assert "rooms on the booking: 2" in rendered


@pytest.mark.parametrize("bad", [{"party_size": 0}, {"party_size": 100}, {"rooms_booked": 0}, {"rooms_booked": -1}])
def test_the_counts_are_bounded(bad: dict[str, int]) -> None:
    """A count outside a plausible booking is a data error, not a large party."""
    with pytest.raises(ValidationError):
        CaseContext(**bad)


def test_supplier_status_no_longer_claims_to_know_who_refused() -> None:
    """Knowing that the supplier refused is not knowing which person refused.

    ``supplier_status`` used to answer "Name and role of the property contact who
    refused", which suppressed a question the agent needed: the procedure requires
    the named contact as evidence, and an enum value cannot supply a name. It is
    still handed to the model as branch data.
    """
    context = CaseContext(supplier_status=SupplierStatus.CANNOT_ACCOMMODATE)
    assert answers_requirement(context, "Name and role of the property contact who refused") == ()

    use = describe_use(context, requirements=("Name and role of the property contact who refused",))
    assert use.unanswered == ("Name and role of the property contact who refused",)
    assert "supplier_status" in use.applied
    assert "case data" in use.applied["supplier_status"]


def test_the_response_can_show_exactly_what_the_context_did() -> None:
    use = describe_use(WALKED_GUEST, requirements=(), product_line_applied="hotels")

    assert use.provided == WALKED_GUEST.provided_fields
    assert use.product_line == "hotels"
    assert "restricted the search" in use.applied["product_type"]
    assert use.completed_steps == ("NW-HTL-014#S01", "NW-HTL-014#S02")
    assert use.used_anything
    # Readable by an agent, not a dump of internals.
    assert "hotels" in json.dumps(dict(use.applied))


def test_an_absent_context_reports_every_requirement_as_missing(
    catalogue: ProcedureCatalogue,
) -> None:
    record = catalogue.procedure("NW-HTL-014")
    assert record is not None

    use = describe_use(None, requirements=record.required_information)

    # Every requirement except the ones this contract refuses to hold, which are
    # declined with a reason instead.
    expected = tuple(item for item in record.required_information if refused_reason(item) is None)
    assert use.unanswered == expected
    assert set(use.not_collected) == set(record.required_information) - set(expected)
    assert not use.used_anything
    assert use.provided == ()


def test_context_never_carries_an_instruction(catalogue: ProcedureCatalogue) -> None:
    """The contract selects which part of the procedure applies. It is not policy.

    Asserted structurally: every value the context can hold is either a closed
    enum, a pattern-constrained code, a date, a boolean or a citation token. None
    of those can express "approve a refund", so a context cannot become an
    instruction no matter what is put in it.
    """
    rendered = render_for_prompt(
        CaseContext(payment_status=PaymentStatus.REFUND_ISSUED, supplier_status=SupplierStatus.DISPUTED)
    )
    # It states the situation and nothing about what to do.
    assert "payment status: refund issued" in rendered
    assert "approve" not in rendered.lower()
    assert "must" not in rendered.lower()
    assert "should" not in rendered.lower()

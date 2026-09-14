"""Tests for the structured procedure source the answer path reads.

Two guarantees are load-bearing here and neither is obvious from reading the
module:

1. **The API's reader and the ingestion schema agree.**
   :mod:`agentpath.procedures` deliberately re-states the field names that
   ``agentpath_ingestion.schema`` declares, because the API must not import the
   ingestion package (``tests/test_visual_steps.py`` asserts that on the import
   graph). Duplication without a cross-check is how two readers of one file
   drift, so this module loads the corpus *both* ways and compares every step.

2. **``source_section`` names a heading that exists.** It is the only handle a
   human has for finding the text behind a citation, so it is checked against the
   rendered Markdown rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentpath.procedures import (
    CITATION_TOKEN_PATTERN,
    ProcedureCatalogue,
    all_citations,
    qualified_step_id,
)
from agentpath_ingestion.dataset import LoadedProcedure, load_procedures
from agentpath_ingestion.render import render_markdown
from agentpath_ingestion.schema import qualified_step_id as ingestion_qualified_step_id

REPO_ROOT = Path(__file__).resolve().parents[1]
SLICE_PROCEDURE_ID = "NW-HTL-014"


@pytest.fixture(scope="module")
def catalogue() -> ProcedureCatalogue:
    return ProcedureCatalogue.load()


@pytest.fixture(scope="module")
def dataset() -> list[LoadedProcedure]:
    return load_procedures()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_the_committed_corpus_loads_without_a_single_problem(catalogue: ProcedureCatalogue) -> None:
    assert catalogue.problems == ()
    assert len(catalogue) == 5
    assert catalogue.ids == ("NW-CXL-032", "NW-CXL-045", "NW-HTL-014", "NW-HTL-021", "NW-REF-018")
    assert catalogue.step_count == 39


def test_every_step_arrives_whole(catalogue: ProcedureCatalogue) -> None:
    """The point of the catalogue: nothing is truncated, everything has an id."""
    for record in catalogue:
        assert record.ordered_steps
        for step in record.ordered_steps:
            assert step.instruction.endswith(".")
            assert CITATION_TOKEN_PATTERN.fullmatch(step.citation)
            assert step.procedure_version == record.version


def test_the_api_reader_and_the_ingestion_schema_agree_on_every_step(
    catalogue: ProcedureCatalogue,
    dataset: list[LoadedProcedure],
) -> None:
    """The cross-check that makes the duplicated field names safe.

    ``agentpath.procedures`` re-states field names rather than importing the
    ingestion schema, so that the API cannot pull YAML parsing onto the request
    path. This is what keeps the two honest: a rename, a default change or a
    different id derivation on either side fails here.
    """
    assert catalogue.ids == tuple(item.procedure_id for item in dataset)

    for item in dataset:
        record = catalogue.procedure(item.procedure_id)
        assert record is not None, item.procedure_id
        procedure = item.procedure

        assert record.version == procedure.version
        assert record.title == procedure.title
        assert record.title_ar == procedure.title_ar
        assert record.effective_date == procedure.effective_date.isoformat()
        assert record.scenario_tags == tuple(procedure.scenario_tags)
        assert record.applies_when == tuple(procedure.applies_when)
        assert record.does_not_apply_when == tuple(procedure.does_not_apply_when)
        assert record.required_information == tuple(procedure.required_information)
        assert len(record.steps) == len(procedure.steps)

        for step, expected in zip(record.ordered_steps, procedure.steps, strict=True):
            assert step.step_id == expected.step_id
            assert step.step_order == expected.number
            assert step.instruction == expected.instruction.strip()
            assert step.instruction_ar == ((expected.instruction_ar or "").strip() or None)
            assert step.expected_result == ((expected.expected_result or "").strip() or None)
            assert step.screen_name == expected.screen
            assert step.control_name == expected.ui_element
            assert step.screenshot_id == expected.screenshot
            # The two implementations of the citation token must agree exactly:
            # the answer layer proves a citation by string equality.
            assert step.citation == ingestion_qualified_step_id(item.procedure_id, expected.step_id)
            assert step.citation == qualified_step_id(item.procedure_id, expected.step_id)


def test_control_name_comes_from_the_ui_element_field(catalogue: ProcedureCatalogue) -> None:
    """The authoritative source for a button name, and the only one.

    Most controls in the corpus are not quoted in the instruction prose at all,
    so a system that parsed labels out of text would simply not have them.
    """
    record = catalogue.procedure(SLICE_PROCEDURE_ID)
    assert record is not None
    step = record.step("S07")
    assert step is not None
    assert step.control_name == "Rebook and Link"

    not_in_prose = [
        step for step in record.ordered_steps if step.control_name and step.control_name not in step.instruction
    ]
    assert not_in_prose, "every control is quoted in prose here, so the field is not being exercised"


def test_arabic_instructions_carry_the_english_control_label(catalogue: ProcedureCatalogue) -> None:
    """The corpus fix, seen from the consumer that depends on it."""
    step = catalogue.step("NW-HTL-014#S07")
    assert step is not None
    assert step.instruction_ar is not None
    assert "Rebook and Link" in step.instruction_ar
    assert "(إعادة الحجز والربط)" in step.instruction_ar


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_a_citation_resolves_to_exactly_one_step(catalogue: ProcedureCatalogue) -> None:
    step = catalogue.step("NW-HTL-014#S04")
    assert step is not None
    assert step.step_order == 4
    assert step.control_name == "Log supplier contact"
    assert step.screenshot_id == "agentdesk-supplier-log"


@pytest.mark.parametrize(
    "bad",
    [
        "NW-HTL-014#S99",  # well-formed, does not exist
        "NW-XXX-999#S01",  # unknown procedure
        "NW-HTL-014",  # procedure only
        "S04",  # step only: ambiguous by design
        "NW-HTL-014#S04 ",  # trailing space is tolerated by strip
        "nw-htl-014#s04",  # wrong case
        "",
        None,
        42,
        ["NW-HTL-014#S04"],
    ],
)
def test_a_citation_that_is_not_a_known_step_resolves_to_nothing(catalogue: ProcedureCatalogue, bad: object) -> None:
    resolved = catalogue.step(bad)
    # The one tolerated case is surrounding whitespace, which is stripped.
    if isinstance(bad, str) and bad.strip() == "NW-HTL-014#S04":
        assert resolved is not None
        return
    assert resolved is None


@pytest.mark.parametrize("bad", ["", "NW-HTL", "nw-htl-014", None, 7, ["NW-HTL-014"]])
def test_a_malformed_procedure_id_resolves_to_nothing(catalogue: ProcedureCatalogue, bad: object) -> None:
    assert catalogue.procedure(bad) is None


def test_an_empty_catalogue_serves_nothing() -> None:
    empty = ProcedureCatalogue.empty()
    assert len(empty) == 0
    assert empty.step("NW-HTL-014#S01") is None
    assert empty.steps_for("NW-HTL-014") == ()
    assert "NW-HTL-014" not in empty


def test_all_citations_lists_every_step_in_order(catalogue: ProcedureCatalogue) -> None:
    citations = all_citations(list(catalogue))
    assert len(citations) == catalogue.step_count
    assert len(set(citations)) == len(citations)
    assert citations[:2] == ("NW-CXL-032#S01", "NW-CXL-032#S02")


# --------------------------------------------------------------------------- #
# source_section
# --------------------------------------------------------------------------- #


def test_source_section_names_a_heading_that_exists_in_the_rendered_document(
    catalogue: ProcedureCatalogue,
    dataset: list[LoadedProcedure],
) -> None:
    """A citation an agent cannot follow is not a citation."""
    for item in dataset:
        markdown = render_markdown(item.procedure)
        record = catalogue.procedure(item.procedure_id)
        assert record is not None
        for step in record.ordered_steps:
            document, steps_heading, step_heading = (part.strip() for part in step.source_section.split(">"))
            assert document == f"{item.procedure_id}.md"
            assert steps_heading in markdown
            assert f"{step_heading} " in markdown or markdown.endswith(step_heading)


def test_source_section_is_stable_and_human_readable(catalogue: ProcedureCatalogue) -> None:
    step = catalogue.step("NW-HTL-014#S07")
    assert step is not None
    assert step.source_section == "NW-HTL-014.md > ## Steps > ### Step 7"


# --------------------------------------------------------------------------- #
# Conditions come from structured fields, never from prose
# --------------------------------------------------------------------------- #


def test_step_conditions_are_the_decision_branches_that_target_the_step(
    catalogue: ProcedureCatalogue,
    dataset: list[LoadedProcedure],
) -> None:
    procedure = next(i.procedure for i in dataset if i.procedure_id == SLICE_PROCEDURE_ID)
    branch = next(b for b in procedure.decision_branches if b.goto_step == 10)

    step_ten = catalogue.step("NW-HTL-014#S10")
    assert step_ten is not None
    assert len(step_ten.conditions) == 1
    assert step_ten.conditions[0].startswith("decision_branch: If ")
    assert branch.condition in step_ten.conditions[0]
    assert branch.then in step_ten.conditions[0]
    assert (branch.otherwise or "") in step_ten.conditions[0]
    assert "(continue at step 10)" in step_ten.conditions[0]

    # A step no branch targets carries no step-level condition.
    step_two = catalogue.step("NW-HTL-014#S02")
    assert step_two is not None
    assert step_two.conditions == ()


def test_gate_conditions_are_labelled_with_the_field_they_came_from(
    catalogue: ProcedureCatalogue,
    dataset: list[LoadedProcedure],
) -> None:
    procedure = next(i.procedure for i in dataset if i.procedure_id == SLICE_PROCEDURE_ID)
    record = catalogue.procedure(SLICE_PROCEDURE_ID)
    assert record is not None
    gate = record.gate_conditions

    for item in procedure.applies_when:
        assert f"applies_when: {item}" in gate
    for item in procedure.does_not_apply_when:
        assert f"does_not_apply_when: {item}" in gate
    # Branches with no goto_step belong to the procedure, not to a step.
    unbound = [b for b in procedure.decision_branches if b.goto_step is None]
    assert len(record.unbound_conditions) == len(unbound)
    assert all(line.startswith("decision_branch: If ") for line in record.unbound_conditions)
    assert all(line in gate for line in record.unbound_conditions)


def test_every_condition_line_is_traceable_to_a_yaml_field(
    catalogue: ProcedureCatalogue,
    dataset: list[LoadedProcedure],
) -> None:
    """Nothing in ``conditions`` may be composed prose.

    Each line is ``<field>: <verbatim value>`` (or the rendered branch sentence),
    so a reviewer can find the source line. This asserts the label vocabulary is
    closed.
    """
    allowed = ("applies_when: ", "does_not_apply_when: ", "decision_branch: If ")
    for record in catalogue:
        for line in record.gate_conditions:
            assert line.startswith(allowed), line
        for step in record.ordered_steps:
            for line in step.conditions:
                assert line.startswith(allowed), line

    raw = yaml.safe_load((REPO_ROOT / "data/synthetic/procedures/NW-HTL-014.yaml").read_text(encoding="utf-8"))
    record = catalogue.procedure(SLICE_PROCEDURE_ID)
    assert record is not None
    for line in record.gate_conditions:
        if line.startswith("applies_when: "):
            assert line.removeprefix("applies_when: ") in raw["applies_when"]


def test_escalation_is_carried_field_by_field(catalogue: ProcedureCatalogue) -> None:
    """An answer may say "escalate"; every part of that sentence is corpus data."""
    record = catalogue.procedure(SLICE_PROCEDURE_ID)
    assert record is not None
    joined = "\n".join(record.escalation)
    assert "escalation_queue: Northwind Travel Accommodation Support (Tier 2)" in joined
    assert "escalation_approval_required: yes" in joined
    assert "escalation_approver_role: Accommodation Support Supervisor" in joined


# --------------------------------------------------------------------------- #
# A bad file degrades, it does not crash the service
# --------------------------------------------------------------------------- #


def _write(directory: Path, name: str, payload: object) -> None:
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload, allow_unicode=True)
    (directory / name).write_text(text, encoding="utf-8")


def _valid_payload(procedure_id: str = "NW-TST-001") -> dict:
    return {
        "procedure_id": procedure_id,
        "title": "Test procedure title",
        "title_ar": "إجراء اختباري",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "applies_when": ["Always in tests"],
        "steps": [
            {"number": 1, "instruction": "Do the first thing carefully.", "ui_element": "Test chip"},
            {"number": 2, "instruction": "Do the second thing carefully."},
        ],
        "escalation": {"trigger": "Something went wrong", "queue": "Northwind Travel Test Queue"},
    }


def test_a_missing_directory_is_reported_not_raised(tmp_path: Path) -> None:
    loaded = ProcedureCatalogue.load(tmp_path / "nope")
    assert len(loaded) == 0
    assert any("does not exist" in problem for problem in loaded.problems)
    # No absolute path in the message: it reaches log lines.
    assert str(tmp_path) not in " ".join(loaded.problems)


@pytest.mark.parametrize(
    ("name", "payload", "expected"),
    [
        ("NW-TST-001.yaml", "{not: valid: yaml", "unreadable"),
        ("NW-TST-001.yaml", ["not", "a", "mapping"], "YAML mapping"),
        ("NW-TST-001.yaml", {"procedure_id": "bad-id"}, "unusable procedure_id"),
        ("NW-TST-002.yaml", _valid_payload("NW-TST-001"), "not its own name"),
        ("NW-TST-001.yaml", _valid_payload() | {"version": "1"}, "unusable version"),
        ("NW-TST-001.yaml", _valid_payload() | {"steps": []}, "declares no steps"),
    ],
)
def test_an_unusable_file_is_skipped_and_recorded(tmp_path: Path, name: str, payload: object, expected: str) -> None:
    """One bad procedure must degrade to "cannot answer from that one", not to a crash."""
    _write(tmp_path, name, payload)
    loaded = ProcedureCatalogue.load(tmp_path)
    assert len(loaded) == 0
    assert any(expected in problem for problem in loaded.problems), loaded.problems


def test_a_duplicate_step_id_is_rejected_because_a_citation_would_be_ambiguous(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["steps"][1]["step_id"] = "S01"
    _write(tmp_path, "NW-TST-001.yaml", payload)
    loaded = ProcedureCatalogue.load(tmp_path)
    assert len(loaded) == 0
    assert any("ambiguous" in problem for problem in loaded.problems)


def test_a_good_file_loads_alongside_a_bad_one(tmp_path: Path) -> None:
    _write(tmp_path, "NW-TST-001.yaml", _valid_payload())
    _write(tmp_path, "NW-TST-002.yaml", "{broken")
    loaded = ProcedureCatalogue.load(tmp_path)
    assert loaded.ids == ("NW-TST-001",)
    assert len(loaded.problems) == 1


def test_step_ids_default_from_the_step_number(tmp_path: Path) -> None:
    """Same derivation as the ingestion schema, so citations cannot disagree."""
    _write(tmp_path, "NW-TST-001.yaml", _valid_payload())
    loaded = ProcedureCatalogue.load(tmp_path)
    record = loaded.procedure("NW-TST-001")
    assert record is not None
    assert record.citations == ("NW-TST-001#S01", "NW-TST-001#S02")


def test_a_pinned_step_id_wins_over_the_derived_one(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["steps"][1]["step_id"] = "S07"
    _write(tmp_path, "NW-TST-001.yaml", payload)
    loaded = ProcedureCatalogue.load(tmp_path)
    record = loaded.procedure("NW-TST-001")
    assert record is not None
    assert record.citations == ("NW-TST-001#S01", "NW-TST-001#S07")


def test_an_oversized_file_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "NW-TST-001.yaml", "# padding\n" + ("x" * (512 * 1024 + 1)))
    loaded = ProcedureCatalogue.load(tmp_path)
    assert any("above the" in problem for problem in loaded.problems)


def test_no_absolute_path_appears_in_any_problem(tmp_path: Path) -> None:
    _write(tmp_path, "NW-TST-001.yaml", "{broken")
    loaded = ProcedureCatalogue.load(tmp_path)
    assert loaded.problems
    assert "/home/" not in " ".join(loaded.problems)
    assert str(tmp_path) not in " ".join(loaded.problems)

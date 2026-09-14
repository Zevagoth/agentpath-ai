"""Tests for the synthetic procedure dataset, its schema, and its renderer.

Three things matter here and are tested separately:

1. The dataset on disk is valid and internally consistent.
2. The dataset is unmistakably fictional (banner present, no real brands).
3. The renderer preserves everything retrieval depends on: synonyms in both
   languages, screen and button names, versions, escalation rules.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agentpath_ingestion.cli import main as cli_main
from agentpath_ingestion.dataset import (
    DatasetError,
    LoadedProcedure,
    find_forbidden_brand_tokens,
    load_procedure_file,
    load_procedures,
    procedures_dir,
)
from agentpath_ingestion.render import build_metadata, render, render_markdown, write_build
from agentpath_ingestion.schema import (
    FICTIONAL_BANNER,
    STEP_ID_PATTERN,
    Procedure,
    ProductLine,
    qualified_step_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SESSION_ONE_IDS = {"NW-HTL-014", "NW-HTL-021", "NW-CXL-032", "NW-CXL-045", "NW-REF-018"}


@pytest.fixture(scope="module")
def dataset() -> list[LoadedProcedure]:
    return load_procedures()


# --------------------------------------------------------------------------- #
# Dataset integrity
# --------------------------------------------------------------------------- #


def test_dataset_loads_and_contains_the_session_one_procedures(dataset: list[LoadedProcedure]) -> None:
    ids = {item.procedure_id for item in dataset}
    assert EXPECTED_SESSION_ONE_IDS <= ids
    assert len(dataset) >= 5


def test_every_procedure_covers_more_than_one_product_line(dataset: list[LoadedProcedure]) -> None:
    lines = {item.procedure.product_line for item in dataset}
    assert {ProductLine.HOTELS, ProductLine.CANCELLATIONS, ProductLine.REFUNDS} <= lines


def test_ids_are_unique_and_match_filenames(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        assert item.path.name == f"{item.procedure_id}.yaml"
    assert len({item.procedure_id for item in dataset}) == len(dataset)


def test_related_procedure_references_resolve(dataset: list[LoadedProcedure]) -> None:
    known = {item.procedure_id for item in dataset}
    for item in dataset:
        for related in item.procedure.related_procedures:
            assert related in known


def test_every_procedure_has_bilingual_search_surface(dataset: list[LoadedProcedure]) -> None:
    """Arabic queries must be able to reach an English procedure."""
    for item in dataset:
        procedure = item.procedure
        assert len(procedure.synonyms.en) >= 3, procedure.procedure_id
        assert len(procedure.synonyms.ar) >= 2, procedure.procedure_id
        assert procedure.title_ar
        assert procedure.summary.ar
        assert procedure.customer_wording.ar


def test_every_procedure_has_escalation_and_customer_wording(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        procedure = item.procedure
        assert procedure.escalation.queue.startswith("Northwind Travel")
        assert procedure.escalation.trigger
        assert procedure.customer_wording.en


def test_at_least_some_steps_reference_a_screenshot(dataset: list[LoadedProcedure]) -> None:
    """Stage 2 needs screenshot-bearing steps; the references exist from the start."""
    with_screenshots = [item for item in dataset if item.procedure.screenshot_steps]
    assert len(with_screenshots) == len(dataset), "each procedure should have at least one screenshot step"


def test_approval_required_procedures_name_an_approver(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        escalation = item.procedure.escalation
        if escalation.approval_required:
            assert escalation.approver_role


def test_no_procedure_promises_a_specific_money_amount(dataset: list[LoadedProcedure]) -> None:
    """Guard against inventing compensation limits or fees.

    The procedures may talk about limits that AgentDesk displays, but must never
    hard-code a currency amount that an answer could then quote as policy.
    """
    import re

    money = re.compile(r"(?:USD|EUR|SAR|AED|JOD|\$|€)\s?\d|\d+\s?(?:USD|EUR|SAR|AED|JOD|dollars|euros)", re.I)
    for item in dataset:
        raw = item.path.read_text(encoding="utf-8")
        assert not money.search(raw), f"{item.procedure_id} contains a currency amount"


# --------------------------------------------------------------------------- #
# Synthetic-data guarantees
# --------------------------------------------------------------------------- #


def test_source_files_carry_the_fictional_banner(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        assert FICTIONAL_BANNER in item.path.read_text(encoding="utf-8")


def test_no_real_brand_appears_anywhere_in_the_data_directory() -> None:
    data_dir = REPO_ROOT / "data"
    offenders: dict[str, list[str]] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".md", ".json", ".html", ".css"}:
            continue
        found = find_forbidden_brand_tokens(path.read_text(encoding="utf-8", errors="ignore"))
        if found:
            offenders[str(path.relative_to(REPO_ROOT))] = found
    assert not offenders, f"real brand names found in synthetic data: {offenders}"


def test_brand_guard_detects_a_planted_token() -> None:
    assert find_forbidden_brand_tokens("we integrate with Booking.com today") == ["booking.com"]
    assert find_forbidden_brand_tokens("Northwind Travel and AgentDesk") == []


def test_fictional_agency_naming_is_used(dataset: list[LoadedProcedure]) -> None:
    joined = "\n".join(item.path.read_text(encoding="utf-8") for item in dataset)
    assert "Northwind Travel" in joined
    assert "AgentDesk" in joined


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def _minimal_procedure_payload() -> dict:
    return {
        "procedure_id": "NW-TST-001",
        "title": "Test procedure title",
        "title_ar": "إجراء اختباري",
        "product_line": "hotels",
        "scenario_tags": ["test_tag"],
        "summary": {"en": "English summary", "ar": "ملخص عربي"},
        "synonyms": {"en": ["one", "two", "three"], "ar": ["واحد", "اثنان"]},
        "applies_when": ["Always in tests"],
        "steps": [
            {"number": 1, "instruction": "Do the first thing carefully"},
            {"number": 2, "instruction": "Do the second thing carefully"},
            {"number": 3, "instruction": "Do the third thing carefully"},
        ],
        "escalation": {
            "trigger": "Something went wrong",
            "queue": "Northwind Travel Test Queue",
            "contact_channel": "AgentDesk Escalate button",
            "response_target": "Target shown on the card",
        },
        "customer_wording": {"en": "English wording", "ar": "صياغة عربية"},
        "version": "1.0",
        "effective_date": "2026-01-01",
        "owner_team": "Northwind Travel Test Team",
    }


def test_minimal_payload_validates() -> None:
    procedure = Procedure.model_validate(_minimal_procedure_payload())
    assert procedure.procedure_id == "NW-TST-001"
    assert procedure.effective_date == _dt.date(2026, 1, 1)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ({"procedure_id": "HTL-014"}, "procedure_id"),
        ({"procedure_id": "NW-HOTEL-014"}, "procedure_id"),
        ({"version": "2"}, "version"),
        ({"scenario_tags": ["Not Snake Case"]}, "lower_snake_case"),
        ({"synonyms": {"en": ["only", "two"], "ar": ["واحد", "اثنان"]}}, "synonyms"),
        ({"related_procedures": ["NOT-AN-ID"]}, "malformed"),
        ({"product_line": "trains"}, "product_line"),
    ],
)
def test_schema_rejects_malformed_fields(mutation: dict, expected_error: str) -> None:
    payload = _minimal_procedure_payload() | mutation
    with pytest.raises(ValidationError) as exc:
        Procedure.model_validate(payload)
    assert expected_error in str(exc.value)


def test_schema_rejects_step_numbering_gaps() -> None:
    payload = _minimal_procedure_payload()
    payload["steps"][2]["number"] = 5
    with pytest.raises(ValidationError, match="without gaps"):
        Procedure.model_validate(payload)


def test_schema_rejects_branch_pointing_at_missing_step() -> None:
    payload = _minimal_procedure_payload()
    payload["decision_branches"] = [{"condition": "Some condition", "then": "Some action", "goto_step": 9}]
    with pytest.raises(ValidationError, match="unknown step"):
        Procedure.model_validate(payload)


def test_schema_rejects_self_reference() -> None:
    payload = _minimal_procedure_payload()
    payload["related_procedures"] = ["NW-TST-001"]
    with pytest.raises(ValidationError, match="itself"):
        Procedure.model_validate(payload)


def test_schema_requires_an_approver_when_approval_is_required() -> None:
    payload = _minimal_procedure_payload()
    payload["escalation"]["approval_required"] = True
    with pytest.raises(ValidationError, match="approver_role"):
        Procedure.model_validate(payload)


def test_schema_forbids_unknown_fields() -> None:
    payload = _minimal_procedure_payload() | {"unexpected_field": "value"}
    with pytest.raises(ValidationError, match="unexpected_field"):
        Procedure.model_validate(payload)


# --------------------------------------------------------------------------- #
# Step identity
# --------------------------------------------------------------------------- #


def test_step_id_defaults_to_the_zero_padded_step_number() -> None:
    """Derived, not hand-written, so the five existing procedures need no editing."""
    procedure = Procedure.model_validate(_minimal_procedure_payload())
    assert [step.step_id for step in procedure.steps] == ["S01", "S02", "S03"]


def test_step_ids_sort_in_execution_order(dataset: list[LoadedProcedure]) -> None:
    """Zero padding is the point: 'S02' must sort before 'S10', not after."""
    procedure = next(i.procedure for i in dataset if i.procedure_id == "NW-HTL-014")
    ids = [step.step_id for step in procedure.steps]
    assert len(ids) == 10
    assert sorted(ids) == ids
    assert ids[-1] == "S10"


def test_step_id_can_be_pinned_explicitly_in_yaml() -> None:
    """Renumbering a procedure must not be able to repoint an existing citation."""
    payload = _minimal_procedure_payload()
    payload["steps"][1]["step_id"] = "S07"
    procedure = Procedure.model_validate(payload)
    assert [step.step_id for step in procedure.steps] == ["S01", "S07", "S03"]
    assert procedure.steps[1].number == 2


def test_step_id_survives_a_round_trip_through_yaml(tmp_path: Path) -> None:
    payload = _minimal_procedure_payload()
    payload["steps"][0]["step_id"] = "S99"
    (tmp_path / "NW-TST-001.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    (loaded,) = load_procedures(tmp_path)
    assert loaded.procedure.steps[0].step_id == "S99"


def test_step_id_derivation_is_stable_across_reloads() -> None:
    first = Procedure.model_validate(_minimal_procedure_payload())
    second = Procedure.model_validate(_minimal_procedure_payload())
    assert [s.step_id for s in first.steps] == [s.step_id for s in second.steps]


@pytest.mark.parametrize("bad_id", ["1", "S1", "step-1", "s01", "S01a"])
def test_schema_rejects_a_malformed_explicit_step_id(bad_id: str) -> None:
    payload = _minimal_procedure_payload()
    payload["steps"][0]["step_id"] = bad_id
    with pytest.raises(ValidationError, match="step_id"):
        Procedure.model_validate(payload)


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_step_id_means_use_the_derived_one(empty: str | None) -> None:
    """``step_id:`` written with no value is a request for the default, not an error."""
    payload = _minimal_procedure_payload()
    payload["steps"][0]["step_id"] = empty
    procedure = Procedure.model_validate(payload)
    assert procedure.steps[0].step_id == "S01"


def test_schema_rejects_duplicate_step_ids() -> None:
    """A repeated id would make a citation point at two different instructions."""
    payload = _minimal_procedure_payload()
    payload["steps"][1]["step_id"] = "S01"
    with pytest.raises(ValidationError, match="unique"):
        Procedure.model_validate(payload)


def test_a_step_may_declare_the_steps_it_depends_on() -> None:
    """``requires`` makes step order a checkable property of the corpus."""
    payload = _minimal_procedure_payload()
    payload["steps"][1]["requires"] = ["S01"]
    procedure = Procedure.model_validate(payload)
    assert procedure.steps[1].requires == ("S01",)
    assert procedure.steps[0].requires == ()


def test_schema_rejects_a_dependency_on_a_step_that_does_not_exist() -> None:
    """An unresolvable dependency would silently never be satisfied."""
    payload = _minimal_procedure_payload()
    payload["steps"][1]["requires"] = ["S09"]
    with pytest.raises(ValidationError, match="unknown step"):
        Procedure.model_validate(payload)


def test_schema_rejects_a_dependency_on_a_later_step() -> None:
    """A forward dependency is a cycle in something performed in order."""
    payload = _minimal_procedure_payload()
    payload["steps"][0]["requires"] = ["S02"]
    with pytest.raises(ValidationError, match="earlier step"):
        Procedure.model_validate(payload)


def test_schema_rejects_a_step_that_depends_on_itself() -> None:
    payload = _minimal_procedure_payload()
    payload["steps"][1]["requires"] = ["S02"]
    with pytest.raises(ValidationError, match="earlier step"):
        Procedure.model_validate(payload)


def test_the_relocation_procedure_declares_the_dependency_that_was_being_skipped() -> None:
    """Guards the corpus half of the step-order fix.

    A generated answer repeatedly cited step 6, "present the shortlisted option",
    without step 5, which produces the shortlist. The dependency is declared in
    the procedure so the answer layer can enforce it; if this declaration is ever
    removed, that answer becomes representable again.
    """
    procedure = load_procedure_file(procedures_dir() / "NW-HTL-014.yaml")
    by_id = {step.step_id: step for step in procedure.steps}
    assert by_id["S06"].requires == ("S05",)
    # Sparse on purpose: an early step an agent can do independently declares none.
    assert by_id["S01"].requires == ()
    assert by_id["S05"].requires == ()


def test_qualified_step_id_names_the_procedure() -> None:
    """'S01' alone is ambiguous: every procedure has one."""
    assert qualified_step_id("NW-HTL-014", "S01") == "NW-HTL-014#S01"


def test_every_step_in_the_dataset_has_a_stable_id(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        ids = [step.step_id for step in item.procedure.steps]
        assert len(set(ids)) == len(ids), item.procedure_id
        assert all(STEP_ID_PATTERN.fullmatch(step_id) for step_id in ids), item.procedure_id


def test_loader_reports_the_offending_file(tmp_path: Path) -> None:
    bad = tmp_path / "NW-BAD-001.yaml"
    bad.write_text("procedure_id: NW-BAD-001\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"NW-BAD-001\.yaml"):
        load_procedure_file(bad)


def test_loader_rejects_filename_id_mismatch(tmp_path: Path) -> None:
    payload = _minimal_procedure_payload()
    (tmp_path / "wrong-name.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DatasetError, match=r"file name should be NW-TST-001\.yaml"):
        load_procedures(tmp_path)


def test_loader_rejects_dangling_related_procedure(tmp_path: Path) -> None:
    payload = _minimal_procedure_payload() | {"related_procedures": ["NW-XXX-999"]}
    (tmp_path / "NW-TST-001.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DatasetError, match="does not exist in the dataset"):
        load_procedures(tmp_path)


def test_loader_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no procedure files"):
        load_procedures(tmp_path)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_rendered_document_preserves_retrieval_signal(dataset: list[LoadedProcedure]) -> None:
    item = next(i for i in dataset if i.procedure_id == "NW-HTL-014")
    markdown = render_markdown(item.procedure)

    # Banner and identity
    assert FICTIONAL_BANNER in markdown
    assert "Northwind Travel" in markdown
    assert "NW-HTL-014" in markdown

    # Lexical surface: the phrasings agents actually type, both languages
    assert "hotel has no room" in markdown
    assert "overbooking" in markdown
    assert "الفندق ما عنده غرف" in markdown

    # Screen and control names stay searchable
    assert "Relocation Finder" in markdown
    assert "Rebook and Link" in markdown

    # Structure an answer layer will need later
    for heading in (
        "## Summary",
        "## Also known as (search terms)",
        "## When this procedure applies",
        "## Steps",
        "## Decision branches",
        "## Do not",
        "## Escalation",
        "## Suggested customer wording",
    ):
        assert heading in markdown, heading

    # Version metadata must be visible for conflict resolution
    assert "2.1" in markdown
    assert "2026-03-01" in markdown


def test_rendered_steps_are_numbered_and_carry_expected_results(dataset: list[LoadedProcedure]) -> None:
    item = next(i for i in dataset if i.procedure_id == "NW-REF-018")
    markdown = render_markdown(item.procedure)
    for step in item.procedure.steps:
        assert f"### Step {step.number}" in markdown
        assert step.instruction.strip()[:40] in markdown
    assert "**Expected result:**" in markdown


def test_every_step_id_appears_verbatim_in_the_rendered_document(
    dataset: list[LoadedProcedure],
) -> None:
    """The citation check downstream is a substring search over the evidence text.

    If the token an answer cites is not literally present in the rendered
    document, a valid citation is indistinguishable from an invented one.
    """
    for item in dataset:
        markdown = render_markdown(item.procedure)
        for step in item.procedure.steps:
            token = qualified_step_id(item.procedure_id, step.step_id)
            assert f"**Step ID:** `{token}`" in markdown, token


def test_the_step_id_line_sits_directly_under_its_heading(dataset: list[LoadedProcedure]) -> None:
    """The id has to travel with the heading it names, whatever the chunk boundary."""
    item = next(i for i in dataset if i.procedure_id == "NW-HTL-014")
    lines = render_markdown(item.procedure).splitlines()
    headings = [index for index, line in enumerate(lines) if line.startswith("### Step ")]
    assert len(headings) == len(item.procedure.steps)
    for index, step in zip(headings, item.procedure.steps, strict=True):
        token = qualified_step_id(item.procedure_id, step.step_id)
        assert lines[index + 2] == f"**Step ID:** `{token}`"


def test_rendered_step_ids_are_unique_within_a_document(dataset: list[LoadedProcedure]) -> None:
    """An ambiguous token would let a citation resolve to the wrong instruction."""
    for item in dataset:
        markdown = render_markdown(item.procedure)
        for step in item.procedure.steps:
            token = qualified_step_id(item.procedure_id, step.step_id)
            assert markdown.count(token) == 1, token


def test_rendered_screenshot_steps_reference_their_template(dataset: list[LoadedProcedure]) -> None:
    for item in dataset:
        markdown = render_markdown(item.procedure)
        for step in item.procedure.screenshot_steps:
            assert step.screenshot in markdown


def test_customer_wording_is_labelled_as_generated(dataset: list[LoadedProcedure]) -> None:
    """Retrieved policy and generated phrasing must not look alike."""
    markdown = render_markdown(dataset[0].procedure)
    assert "carries no policy weight" in markdown


def test_render_rejects_a_procedure_mentioning_a_real_brand() -> None:
    payload = _minimal_procedure_payload()
    payload["summary"]["en"] = "Escalate to Booking.com support"
    procedure = Procedure.model_validate(payload)
    loaded = LoadedProcedure(path=Path("NW-TST-001.yaml"), procedure=procedure)
    with pytest.raises(ValueError, match="real brands"):
        render(loaded)


def test_metadata_is_stringified_for_ragflow_filters(dataset: list[LoadedProcedure]) -> None:
    metadata = build_metadata(dataset[0].procedure)
    assert set(metadata) == {
        "procedure_id",
        "title",
        "title_ar",
        "product_line",
        "scenario_tags",
        "severity",
        "version",
        "effective_date",
        "owner_team",
        "doc_kind",
        "language",
        "synthetic",
    }
    assert all(isinstance(value, str) for value in metadata.values())
    assert metadata["synthetic"] == "true"


def test_write_build_produces_documents_and_a_manifest(tmp_path: Path, dataset: list[LoadedProcedure]) -> None:
    rendered = [render(item) for item in dataset]
    manifest_path = write_build(rendered, tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["document_count"] == len(dataset)
    assert manifest["banner"] == FICTIONAL_BANNER
    for entry in manifest["documents"]:
        document = tmp_path / entry["filename"]
        assert document.is_file()
        assert entry["sha256"]
        assert entry["metadata"]["procedure_id"] == entry["filename"].removesuffix(".md")


def test_rendering_is_deterministic(dataset: list[LoadedProcedure]) -> None:
    first = render(dataset[0])
    second = render(dataset[0])
    assert first.markdown == second.markdown
    assert first.sha256 == second.sha256


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_validate_succeeds_on_the_real_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["validate"]) == 0
    output = capsys.readouterr().out
    assert "validated" in output
    assert "NW-HTL-014" in output


def test_cli_build_writes_to_a_target_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["build", "--target", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "rendered" in output
    assert (tmp_path / "NW-HTL-014.md").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_cli_validate_fails_cleanly_on_a_broken_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "NW-BAD-001.yaml").write_text("procedure_id: NW-BAD-001\n", encoding="utf-8")
    assert cli_main(["validate", "--source", str(tmp_path)]) == 1
    assert "dataset invalid" in capsys.readouterr().err


def test_procedures_directory_is_where_the_loader_expects_it() -> None:
    assert procedures_dir() == REPO_ROOT / "data" / "synthetic" / "procedures"
    assert procedures_dir().is_dir()


# --------------------------------------------------------------------------- #
# English control labels survive translation
# --------------------------------------------------------------------------- #
#
# T7 measured English control-label preservation at 0/4 and the cause was the
# corpus, not the model: NW-HTL-014's own ``instruction_ar`` wrote
# ``"إعادة الحجز والربط"`` where the step declares ``ui_element: Rebook and
# Link``, so an Arabic answer that was maximally faithful to the evidence *lost*
# the label. Faithfulness and label preservation contradicted each other, and no
# amount of prompting can satisfy both.
#
# The corpus rule that removes the contradiction: an Arabic instruction keeps the
# English label verbatim and puts the Arabic gloss after it in brackets -
# ``اضغط Rebook and Link (إعادة الحجز والربط)``. The agent can then find the
# control on screen and still read the sentence.
#
# Two checks, because one direction is mechanically decidable and the other needs
# the corpus to declare its own translations:
#
# 1. If the *English* instruction quotes the step's declared control label
#    verbatim, the Arabic instruction must quote it too. Capitalisation is the
#    corpus's own signal: "read the cancellation terms" uses a common noun,
#    "through Request Waiver Approval" names a button.
# 2. No Arabic gloss may appear anywhere in the corpus without the English label
#    it glosses. The gloss pairs are harvested from the YAML itself, so reverting
#    any one of these edits fails the test without anything having to be listed
#    twice.

#: ``Rebook and Link (إعادة الحجز والربط)`` - an English UI string followed by its
#: Arabic gloss in brackets. This is the shape the corpus commits to.
GLOSSED_LABEL = re.compile(r"([A-Z][A-Za-z]*(?: [A-Za-z]+)*)\s*\(([\u0600-\u06FF][^)]*)\)")

#: A gloss shorter than this is a single common word whose appearance elsewhere
#: says nothing. The shortest real gloss in the corpus is 5 characters.
MIN_GLOSS_LENGTH = 5


def _steps_with_arabic(dataset: list[LoadedProcedure]):
    for item in dataset:
        for step in item.procedure.steps:
            if step.instruction_ar:
                yield item.procedure_id, step


def find_translated_control_labels(procedures: list[LoadedProcedure]) -> list[str]:
    """Steps whose Arabic instruction dropped a control label the English keeps.

    A function rather than an inline loop so the guard itself can be tested
    against a planted violation - a check that cannot fail is not a check.
    """
    violations = []
    for procedure_id, step in _steps_with_arabic(procedures):
        label = step.ui_element
        if not label or label not in step.instruction:
            continue
        if label not in (step.instruction_ar or ""):
            violations.append(f"{procedure_id}#{step.step_id}: instruction_ar drops the control label {label!r}")
    return violations


def harvest_glosses(procedures: list[LoadedProcedure]) -> dict[str, str]:
    """Every ``Label (gloss)`` pair the corpus's Arabic instructions declare."""
    glosses: dict[str, str] = {}
    for _procedure_id, step in _steps_with_arabic(procedures):
        for label, gloss in GLOSSED_LABEL.findall(step.instruction_ar or ""):
            cleaned = gloss.strip()
            if len(cleaned) >= MIN_GLOSS_LENGTH:
                glosses[cleaned] = label
    return glosses


def find_glosses_standing_in_for_a_label(procedures: list[LoadedProcedure]) -> list[str]:
    """Arabic instructions using a known gloss where the English label is absent."""
    glosses = harvest_glosses(procedures)
    violations = []
    for procedure_id, step in _steps_with_arabic(procedures):
        arabic = step.instruction_ar or ""
        for gloss, label in glosses.items():
            if gloss in arabic and label not in arabic:
                violations.append(
                    f"{procedure_id}#{step.step_id}: uses the Arabic gloss {gloss!r} without the English {label!r}"
                )
    return violations


def test_an_arabic_instruction_never_translates_away_its_own_control_label(
    dataset: list[LoadedProcedure],
) -> None:
    assert find_translated_control_labels(dataset) == []


def test_the_control_label_guard_catches_a_planted_translation() -> None:
    """The negative control for the check above."""
    payload = _minimal_procedure_payload()
    payload["steps"][0] |= {
        "instruction": "Submit the request through Request Waiver Approval before anything else",
        "instruction_ar": 'أرسل الطلب عبر "طلب موافقة على الاستثناء" قبل أي شيء آخر',
        "screen": "Waiver Grounds",
        "ui_element": "Request Waiver Approval",
    }
    planted = [LoadedProcedure(path=Path("NW-TST-001.yaml"), procedure=Procedure.model_validate(payload))]

    (violation,) = find_translated_control_labels(planted)
    assert "Request Waiver Approval" in violation


def test_the_corpus_declares_the_glosses_it_uses(dataset: list[LoadedProcedure]) -> None:
    """The label/gloss pairs are in the data, so nothing has to be listed twice."""
    glosses = harvest_glosses(dataset)
    assert glosses, "no English label carries an Arabic gloss; the convention is not being used"
    # The two labels T7 gated on, and the one whose absence caused the 0/4.
    assert glosses.get("إعادة الحجز والربط") == "Rebook and Link"
    assert "Log supplier contact" in glosses.values()


def test_no_arabic_gloss_stands_in_for_the_english_label_it_glosses(
    dataset: list[LoadedProcedure],
) -> None:
    """Reverting any one of these edits reintroduces the T7 failure. This catches it."""
    assert find_glosses_standing_in_for_a_label(dataset) == []


def test_the_gloss_guard_catches_a_reverted_edit(dataset: list[LoadedProcedure]) -> None:
    """The negative control: put a declared gloss back on its own and it is flagged."""
    payload = _minimal_procedure_payload()
    payload["steps"][0] |= {
        "instruction": "Issue the alternative with Rebook and Link and keep the case attached",
        "instruction_ar": "أصدر الحجز البديل باستخدام إعادة الحجز والربط",
        "screen": "Relocation Finder",
        "ui_element": "Rebook and Link",
    }
    reverted = [*dataset, LoadedProcedure(path=Path("NW-TST-001.yaml"), procedure=Procedure.model_validate(payload))]

    violations = find_glosses_standing_in_for_a_label(reverted)
    assert any("إعادة الحجز والربط" in violation for violation in violations)


def test_control_names_are_read_from_the_field_not_parsed_from_prose(
    dataset: list[LoadedProcedure],
) -> None:
    """``ui_element`` stays the authoritative source for a control name.

    The label appearing in the instruction text is a convenience for lexical
    retrieval and for an agent reading the sentence. It is never the source: the
    answer layer reads ``ui_element`` (as ``StepRecord.control_name``), which is
    why a step can name a control the prose never mentions - and most of them do.
    """
    with_control = [step for _id, step in _steps_with_arabic(dataset) if step.ui_element]
    assert with_control
    unnamed = [step for step in with_control if step.ui_element not in step.instruction]
    assert unnamed, "every control happens to be quoted in prose, so this guarantee is untested"


def test_the_rendered_document_carries_the_english_label_inside_the_arabic_line(
    dataset: list[LoadedProcedure],
) -> None:
    """The fix has to survive rendering, because that is what gets indexed."""
    item = next(i for i in dataset if i.procedure_id == "NW-HTL-014")
    markdown = render_markdown(item.procedure)
    arabic_line = next(line for line in markdown.splitlines() if "إعادة الحجز والربط" in line)
    assert arabic_line.startswith("**عربي:**")
    assert "Rebook and Link" in arabic_line

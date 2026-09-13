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
from agentpath_ingestion.schema import FICTIONAL_BANNER, Procedure, ProductLine

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


def test_loader_reports_the_offending_file(tmp_path: Path) -> None:
    bad = tmp_path / "NW-BAD-001.yaml"
    bad.write_text("procedure_id: NW-BAD-001\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="NW-BAD-001.yaml"):
        load_procedure_file(bad)


def test_loader_rejects_filename_id_mismatch(tmp_path: Path) -> None:
    payload = _minimal_procedure_payload()
    (tmp_path / "wrong-name.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(DatasetError, match="file name should be NW-TST-001.yaml"):
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


def test_cli_validate_fails_cleanly_on_a_broken_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "NW-BAD-001.yaml").write_text("procedure_id: NW-BAD-001\n", encoding="utf-8")
    assert cli_main(["validate", "--source", str(tmp_path)]) == 1
    assert "dataset invalid" in capsys.readouterr().err


def test_procedures_directory_is_where_the_loader_expects_it() -> None:
    assert procedures_dir() == REPO_ROOT / "data" / "synthetic" / "procedures"
    assert procedures_dir().is_dir()

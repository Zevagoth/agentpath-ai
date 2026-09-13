"""Tests for the visual-step data model.

A visual step joins two artefacts that are written by different tools at
different times: the procedure YAML and the screenshot sidecar JSON. The value
of the model is almost entirely in what it refuses to build, so most of what is
tested here is failure:

* a step that claims a screenshot nobody generated,
* a sidecar that describes a different step than the one pointing at it,
* a PNG that has been regenerated since the sidecar was written.

Any of those would put a picture of the wrong screen next to an instruction, and
an agent would trust it. The happy path is checked once; the ways it can go
wrong are checked individually.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from agentpath_ingestion.dataset import (
    LoadedProcedure,
    find_forbidden_brand_tokens,
    load_procedures,
    screenshots_dir,
)
from agentpath_ingestion.schema import Procedure
from agentpath_ingestion.visual_steps import (
    SCREENSHOT_PATH_PREFIX,
    HighlightBox,
    VisualStep,
    VisualStepError,
    find_screenshot_gaps,
    load_visual_steps,
)
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The vertical slice T2 actually generated screenshots for.
SLICE_PROCEDURE_ID = "NW-HTL-014"
GENERATED_SCREENSHOT_IDS = {
    "agentdesk-booking-detail-status",
    "agentdesk-supplier-log",
    "agentdesk-relocation-finder",
}


@pytest.fixture(scope="module")
def dataset() -> list[LoadedProcedure]:
    return load_procedures()


@pytest.fixture(scope="module")
def slice_procedure(dataset: list[LoadedProcedure]) -> LoadedProcedure:
    return next(item for item in dataset if item.procedure_id == SLICE_PROCEDURE_ID)


# --------------------------------------------------------------------------- #
# A synthetic repository, so failure modes can be provoked without touching
# the committed screenshots.
# --------------------------------------------------------------------------- #

SCREENSHOT_ID = "agentdesk-test-screen"


def _procedure_payload() -> dict:
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
            {
                "number": 1,
                "instruction": "Open the test screen and read the chip",
                "screen": "Test Screen",
                "ui_element": "Test chip",
                "expected_result": "The chip reads Confirmed",
                "screenshot": SCREENSHOT_ID,
            },
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


def _sidecar_payload(image_bytes: bytes) -> dict:
    return {
        "screenshot_id": SCREENSHOT_ID,
        "screen_name": "Test Screen",
        "control_name": "Test chip",
        "procedure_id": "NW-TST-001",
        "procedure_version": "1.0",
        "step_order": 1,
        "image_path": f"{SCREENSHOT_PATH_PREFIX}{SCREENSHOT_ID}.png",
        "width": 1440,
        "height": 900,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "language": "en",
        "dom_text": "FICTIONAL — SYNTHETIC DEMO DATA\nTest Screen\nTest chip\nConfirmed",
        "ocr_text": None,
        "image_description": {
            "en": "AgentDesk Test Screen with the Test chip highlighted.",
            "ar": "شاشة اختبار في AgentDesk مع إبراز شريحة الاختبار.",
        },
        "highlight": {
            "control_name": "Test chip",
            "x": 243,
            "y": 231.58,
            "width": 109.25,
            "height": 26,
        },
    }


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repository root carrying one procedure, one PNG and one sidecar."""
    directory = tmp_path / "data" / "synthetic" / "screenshots"
    directory.mkdir(parents=True)

    image_bytes = b"\x89PNG\r\n\x1a\n synthetic test image"
    (directory / f"{SCREENSHOT_ID}.png").write_bytes(image_bytes)
    (directory / f"{SCREENSHOT_ID}.json").write_text(
        json.dumps(_sidecar_payload(image_bytes), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tmp_path


def _fake_procedures() -> list[LoadedProcedure]:
    procedure = Procedure.model_validate(_procedure_payload())
    return [LoadedProcedure(path=Path("NW-TST-001.yaml"), procedure=procedure)]


def _load(fake_repo: Path) -> list[VisualStep]:
    return load_visual_steps(
        _fake_procedures(),
        directory=fake_repo / "data" / "synthetic" / "screenshots",
        root=fake_repo,
    )


def _edit_sidecar(fake_repo: Path, **changes: object) -> None:
    path = fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.json"
    payload = json.loads(path.read_text(encoding="utf-8")) | changes
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_a_visual_step_joins_the_procedure_step_to_its_sidecar(fake_repo: Path) -> None:
    (visual,) = _load(fake_repo)

    # Identity, taken from the procedure rather than the sidecar.
    assert visual.procedure_id == "NW-TST-001"
    assert visual.procedure_version == "1.0"
    assert visual.step_id == "S01"
    assert visual.step_order == 1
    assert visual.citation == "NW-TST-001#S01"

    # What the agent is looking at.
    assert visual.screen_name == "Test Screen"
    assert visual.control_name == "Test chip"
    assert visual.instruction == "Open the test screen and read the chip"
    assert visual.expected_result == "The chip reads Confirmed"

    # The image and its description.
    assert visual.image_path == f"{SCREENSHOT_PATH_PREFIX}{SCREENSHOT_ID}.png"
    assert visual.image_width == 1440
    assert visual.image_height == 900
    assert visual.image_description.en.startswith("AgentDesk Test Screen")
    assert visual.image_description.ar
    assert visual.language == "en"

    # DOM text is the screenshot's text; OCR is never populated.
    assert "Test chip" in visual.dom_text
    assert visual.ocr_text is None

    # The highlight box locates the control in the image.
    assert visual.highlight.control_name == "Test chip"
    assert (visual.highlight.x, visual.highlight.width) == (243.0, 109.25)


def test_the_stage_2a_slice_loads_from_the_committed_screenshots(
    slice_procedure: LoadedProcedure,
) -> None:
    """The three real screenshots T2 generated must join cleanly to their steps."""
    visuals = load_visual_steps([slice_procedure])

    assert {item.screenshot_id for item in visuals} == GENERATED_SCREENSHOT_IDS
    assert [item.step_order for item in visuals] == [1, 4, 5]
    assert [item.citation for item in visuals] == [
        "NW-HTL-014#S01",
        "NW-HTL-014#S04",
        "NW-HTL-014#S05",
    ]
    for item in visuals:
        assert (REPO_ROOT / item.image_path).is_file()
        assert item.ocr_text is None
        assert item.dom_text.strip()


def test_the_citation_appears_verbatim_in_the_rendered_document(
    slice_procedure: LoadedProcedure,
) -> None:
    """A visual step's citation has to be findable in the evidence text.

    This is the property the answer layer leans on to catch invented citations,
    so the two producers of the token must not drift apart.
    """
    from agentpath_ingestion.render import render_markdown

    markdown = render_markdown(slice_procedure.procedure)
    for item in load_visual_steps([slice_procedure]):
        assert item.citation in markdown, item.citation


# --------------------------------------------------------------------------- #
# No absolute paths, ever
# --------------------------------------------------------------------------- #


def test_loaded_records_never_carry_an_absolute_path(slice_procedure: LoadedProcedure) -> None:
    for item in load_visual_steps([slice_procedure]):
        assert not Path(item.image_path).is_absolute()
        assert item.image_path.startswith(SCREENSHOT_PATH_PREFIX)
        assert "/home/" not in json.dumps(item.model_dump(), ensure_ascii=False)


@pytest.mark.parametrize(
    ("bad_path", "expected_error"),
    [
        ("/home/someone/repo/data/synthetic/screenshots/x.png", "absolute"),
        ("/data/synthetic/screenshots/x.png", "absolute"),
        ("data/synthetic/screenshots/../../../etc/passwd", "traverse"),
        ("data\\synthetic\\screenshots\\x.png", "POSIX"),
        ("data/synthetic/other/x.png", SCREENSHOT_PATH_PREFIX),
        ("x.png", SCREENSHOT_PATH_PREFIX),
    ],
)
def test_the_model_rejects_paths_that_are_not_repository_relative(bad_path: str, expected_error: str) -> None:
    with pytest.raises(ValidationError, match=expected_error):
        VisualStep(
            procedure_id="NW-TST-001",
            procedure_version="1.0",
            step_id="S01",
            step_order=1,
            screen_name="Test Screen",
            control_name="Test chip",
            instruction="Open the test screen and read the chip",
            screenshot_id=SCREENSHOT_ID,
            image_path=bad_path,
            image_width=1440,
            image_height=900,
            image_sha256="0" * 64,
            image_description={"en": "English", "ar": "عربي"},
            dom_text="Test chip",
            language="en",
            highlight=HighlightBox(control_name="Test chip", x=1, y=1, width=1, height=1),
        )


def test_error_messages_do_not_leak_the_filesystem_layout(fake_repo: Path) -> None:
    (fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.json").unlink()
    with pytest.raises(VisualStepError) as exc:
        _load(fake_repo)
    assert str(fake_repo) not in str(exc.value)
    assert SCREENSHOT_PATH_PREFIX in str(exc.value)


# --------------------------------------------------------------------------- #
# OCR is carried, never populated
# --------------------------------------------------------------------------- #


def test_populating_ocr_text_is_a_validation_error(fake_repo: Path) -> None:
    """Nothing may depend on OCR, so the type makes filling the field impossible."""
    _edit_sidecar(fake_repo, ocr_text="text lifted from the image by an OCR engine")
    with pytest.raises(VisualStepError, match="ocr_text"):
        _load(fake_repo)


# --------------------------------------------------------------------------- #
# Failure: no sidecar
# --------------------------------------------------------------------------- #


def test_a_step_declaring_a_screenshot_that_was_never_generated_fails(fake_repo: Path) -> None:
    (fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.json").unlink()
    with pytest.raises(VisualStepError, match="no sidecar JSON"):
        _load(fake_repo)


def test_a_sidecar_pointing_at_a_missing_image_fails(fake_repo: Path) -> None:
    (fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.png").unlink()
    with pytest.raises(VisualStepError, match="is not on disk"):
        _load(fake_repo)


def test_an_unreadable_sidecar_fails(fake_repo: Path) -> None:
    path = fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(VisualStepError, match="unreadable"):
        _load(fake_repo)


# --------------------------------------------------------------------------- #
# Failure: the sidecar describes a different step
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("procedure_id", "NW-HTL-014"),
        ("step_order", 2),
        ("screen_name", "Some Other Screen"),
        ("control_name", "Some Other Control"),
        ("procedure_version", "9.9"),
        ("screenshot_id", "agentdesk-something-else"),
    ],
)
def test_a_sidecar_that_disagrees_with_the_procedure_fails(fake_repo: Path, field: str, wrong_value: object) -> None:
    _edit_sidecar(fake_repo, **{field: wrong_value})
    with pytest.raises(VisualStepError, match=f"sidecar {field}=.*disagrees"):
        _load(fake_repo)


def test_every_disagreement_is_reported_in_one_pass(fake_repo: Path) -> None:
    """One run should tell you everything that is wrong, not just the first thing."""
    _edit_sidecar(fake_repo, screen_name="Wrong Screen", control_name="Wrong Control")
    with pytest.raises(VisualStepError) as exc:
        _load(fake_repo)
    message = str(exc.value)
    assert "screen_name" in message
    assert "control_name" in message


def test_a_screenshot_captured_for_another_procedure_is_not_silently_reused(
    fake_repo: Path,
) -> None:
    """Two procedures may reference the same screen, but the capture belongs to one step.

    The generated image carries the procedure id and step number in its own
    watermark, so reusing it elsewhere would show an agent a caption naming a
    procedure they are not following.
    """
    _edit_sidecar(fake_repo, procedure_id="NW-HTL-021")
    with pytest.raises(VisualStepError, match="procedure_id"):
        _load(fake_repo)


# --------------------------------------------------------------------------- #
# Failure: the image changed after the sidecar was written
# --------------------------------------------------------------------------- #


def test_a_regenerated_image_that_no_longer_matches_its_hash_fails(fake_repo: Path) -> None:
    image = fake_repo / "data" / "synthetic" / "screenshots" / f"{SCREENSHOT_ID}.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n a different image entirely")
    with pytest.raises(VisualStepError, match="stale"):
        _load(fake_repo)


def test_a_sidecar_without_a_recorded_hash_fails(fake_repo: Path) -> None:
    _edit_sidecar(fake_repo, sha256="")
    with pytest.raises(VisualStepError, match="does not record a sha256"):
        _load(fake_repo)


def test_a_screenshot_step_without_a_screen_or_control_fails(fake_repo: Path) -> None:
    payload = _procedure_payload()
    payload["steps"][0].pop("ui_element")
    procedure = Procedure.model_validate(payload)
    with pytest.raises(VisualStepError, match="ui_element"):
        load_visual_steps(
            [LoadedProcedure(path=Path("NW-TST-001.yaml"), procedure=procedure)],
            directory=fake_repo / "data" / "synthetic" / "screenshots",
            root=fake_repo,
        )


# --------------------------------------------------------------------------- #
# Gaps are visible rather than silent
# --------------------------------------------------------------------------- #


def test_the_stage_2a_slice_has_no_gaps(slice_procedure: LoadedProcedure) -> None:
    assert find_screenshot_gaps([slice_procedure]) == []


def test_screens_that_have_not_been_generated_are_reported_not_skipped(
    dataset: list[LoadedProcedure],
) -> None:
    """The corpus declares more screenshots than T2 generated.

    Loading the whole corpus strictly would fail, and skipping quietly is what
    this module exists to prevent, so the shortfall is reported explicitly.
    """
    gaps = find_screenshot_gaps(dataset)

    assert gaps, "the corpus still declares screenshots that have not been generated"
    assert all(gap.problems for gap in gaps)
    assert SLICE_PROCEDURE_ID not in {gap.procedure_id for gap in gaps}

    ungenerated = {gap.screenshot_id for gap in gaps if any("no sidecar" in p for p in gap.problems)}
    assert ungenerated
    assert not (ungenerated & GENERATED_SCREENSHOT_IDS)


def test_the_api_does_not_import_the_ingestion_package() -> None:
    """Screenshot enrichment is an ingestion-time concern and must stay there.

    Nothing on this path may run while a user search or answer request is in
    flight, and the cheapest way to guarantee that is for the request path not
    to be able to reach the code at all. Asserted on the import graph rather
    than by reading the imports, so it cannot rot.
    """
    import importlib
    import subprocess
    import sys

    assert importlib.util.find_spec("agentpath.api.app") is not None

    # A subprocess, because this test session has already imported ingestion.
    probe = (
        "import sys; import agentpath.api.app; "
        "print(','.join(sorted(m for m in sys.modules if m.startswith('agentpath_ingestion'))))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "", f"the API pulled in ingestion modules: {result.stdout.strip()}"


def test_screenshots_directory_is_where_the_loader_expects_it() -> None:
    assert screenshots_dir() == REPO_ROOT / "data" / "synthetic" / "screenshots"
    assert screenshots_dir().is_dir()


# --------------------------------------------------------------------------- #
# Synthetic-data guarantee for the screenshot artefacts
# --------------------------------------------------------------------------- #


def test_no_real_brand_appears_in_the_generated_screens_or_sidecars() -> None:
    """The repo-wide data scan would pass vacuously if these files vanished.

    This names the screenshot artefacts explicitly, so the guarantee is tied to
    the files actually existing rather than to the directory happening to be
    non-empty.
    """
    screens = sorted((REPO_ROOT / "data" / "synthetic" / "screens").glob("*.html"))
    stylesheets = sorted((REPO_ROOT / "data" / "synthetic" / "screens").glob("*.css"))
    sidecars = sorted(screenshots_dir().glob("*.json"))

    assert len(screens) >= 3, "the three AgentDesk screens should be committed"
    assert stylesheets, "the screen stylesheet should be committed"
    assert len(sidecars) >= 3, "each generated screenshot should have a sidecar"

    offenders: dict[str, list[str]] = {}
    for path in (*screens, *stylesheets, *sidecars):
        found = find_forbidden_brand_tokens(path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = found
    assert not offenders, f"real brand names found in screenshot artefacts: {offenders}"


def test_sidecar_dom_text_and_descriptions_are_fictional(
    slice_procedure: LoadedProcedure,
) -> None:
    """The screenshot text an answer may quote must be synthetic too."""
    for item in load_visual_steps([slice_procedure]):
        assert find_forbidden_brand_tokens(item.dom_text) == []
        assert find_forbidden_brand_tokens(item.image_description.en) == []
        assert find_forbidden_brand_tokens(item.image_description.ar) == []
        assert "Northwind Travel" in item.dom_text or "AgentDesk" in item.dom_text

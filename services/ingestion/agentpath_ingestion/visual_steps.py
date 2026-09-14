"""Visual steps: a procedure step joined to the screenshot that illustrates it.

Two artefacts describe the same step and neither is complete on its own. The
procedure YAML owns the instruction, the version and the step id. The sidecar
JSON written by ``make screenshots`` owns the image, its dimensions, its hash,
the DOM text that was on screen and the bilingual description. A ``VisualStep``
is the join, and it is built strictly: if the two disagree about which step is
depicted, or if the PNG on disk is not the PNG the sidecar recorded, loading
fails rather than handing the answer layer a screenshot of the wrong thing.

Three deliberate constraints:

* **Repository-relative paths only.** ``image_path`` is validated as relative and
  confined to the screenshots directory. An absolute path such as ``/home/...``
  is rejected at the model boundary, so it cannot reach a chunk, an API response
  or a log line. Resolving to a real file happens in the loader, and the
  resolved path is never stored on the record.
* **No OCR.** ``ocr_text`` is annotated ``None``: the field exists so the shape
  matches systems that do carry OCR, and the type makes populating it a
  validation error. ``dom_text`` is the authoritative screenshot text, taken
  from the page the generator rendered.
* **Ingestion time only.** Nothing here is imported by the API package, so no
  YAML parsing, no JSON parsing and no image hashing can happen while a user
  search or answer request is in flight. The dependency runs one way:
  ``services/ingestion`` depends on ``apps/api``, never the reverse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentpath_ingestion.dataset import LoadedProcedure, repo_root, screenshots_dir
from agentpath_ingestion.schema import (
    STEP_ID_PATTERN,
    LocalizedText,
    Procedure,
    ProcedureId,
    Step,
    qualified_step_id,
)

#: Every screenshot lives here, and ``image_path`` may point nowhere else.
SCREENSHOT_PATH_PREFIX = "data/synthetic/screenshots/"

#: Screenshot ids are used as filenames, so the character set is restricted.
SCREENSHOT_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class VisualStepError(RuntimeError):
    """A screenshot and the step it claims to depict do not agree."""


class HighlightBox(BaseModel):
    """Where the highlighted control sits in the image, in CSS pixels.

    Top-left origin, ``deviceScaleFactor`` 1, as measured by the generator from
    the live element rather than guessed, so the UI can draw the same box over
    the image at any display size.
    """

    model_config = ConfigDict(extra="forbid")

    control_name: str = Field(min_length=1)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class VisualStep(BaseModel):
    """One screenshot-bearing procedure step, ready to attach to a chunk."""

    model_config = ConfigDict(extra="forbid")

    procedure_id: ProcedureId
    procedure_version: str = Field(pattern=r"^\d+\.\d+$")
    step_id: str = Field(pattern=STEP_ID_PATTERN.pattern)
    step_order: int = Field(ge=1, le=40)
    screen_name: str = Field(min_length=1)
    control_name: str = Field(min_length=1)
    instruction: str = Field(min_length=10)
    instruction_ar: str | None = Field(
        default=None,
        min_length=5,
        description="The step's Arabic instruction from the procedure YAML, when it has one.",
    )
    expected_result: str | None = Field(default=None, min_length=5)
    screenshot_id: str = Field(pattern=SCREENSHOT_ID_PATTERN)
    image_path: str = Field(description="Repository-relative POSIX path to the PNG.")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_description: LocalizedText
    dom_text: str = Field(min_length=1, description="Authoritative screenshot text, from the rendered page.")
    ocr_text: None = Field(
        default=None,
        description="Present for schema compatibility only. Populating it is a validation error.",
    )
    language: str = Field(pattern=r"^(en|ar)$")
    highlight: HighlightBox

    @field_validator("image_path")
    @classmethod
    def _image_path_stays_inside_the_repository(cls, value: str) -> str:
        """Reject anything that is not a tame repository-relative path.

        This is the choke point for the "no absolute paths downstream" rule, so
        it checks the shape rather than trusting the producer: an absolute path,
        a Windows path, or any ``..`` segment fails here instead of leaking a
        filesystem layout into a chunk or an HTTP response.
        """
        if "\\" in value:
            raise ValueError(f"image_path must use POSIX separators, got {value!r}")
        candidate = PurePosixPath(value)
        if candidate.is_absolute():
            raise ValueError(f"image_path must be repository-relative, got absolute path {value!r}")
        if ".." in candidate.parts:
            raise ValueError(f"image_path must not traverse upwards, got {value!r}")
        if not value.startswith(SCREENSHOT_PATH_PREFIX):
            raise ValueError(f"image_path must sit under {SCREENSHOT_PATH_PREFIX}, got {value!r}")
        return value

    @property
    def citation(self) -> str:
        """The token an answer cites for this step, e.g. ``NW-HTL-014#S01``."""
        return qualified_step_id(self.procedure_id, self.step_id)


@dataclass(frozen=True)
class ScreenshotGap:
    """A step that declares a screenshot which is not usable yet, and why."""

    procedure_id: str
    step_id: str
    step_order: int
    screenshot_id: str
    problems: tuple[str, ...]


def sidecar_path(screenshot_id: str, directory: Path) -> Path:
    """Where the sidecar for *screenshot_id* is expected to live."""
    return directory / f"{screenshot_id}.json"


def collect_visual_steps(
    procedures: Sequence[LoadedProcedure],
    *,
    directory: Path | None = None,
    root: Path | None = None,
) -> tuple[list[VisualStep], list[ScreenshotGap]]:
    """Build what can be built, and report what cannot, in a single pass.

    This is the primitive the other two loaders are written in terms of, and the
    one ingestion uses. Ingestion needs both halves at once: the vertical slice
    has usable screenshots for exactly one procedure, so it must attach those and
    still say out loud what is missing elsewhere. Getting the two halves from two
    separate calls would hash and read every PNG twice and could disagree if the
    directory changed in between.
    """
    directory = directory if directory is not None else screenshots_dir()
    base = root if root is not None else repo_root()

    steps: list[VisualStep] = []
    gaps: list[ScreenshotGap] = []
    for loaded in procedures:
        procedure = loaded.procedure
        for step in procedure.screenshot_steps:
            visual, problems = _build(procedure, step, directory, base)
            if visual is not None:
                steps.append(visual)
            if problems:
                gaps.append(
                    ScreenshotGap(
                        procedure_id=procedure.procedure_id,
                        step_id=step.step_id,
                        step_order=step.number,
                        screenshot_id=str(step.screenshot),
                        problems=tuple(problems),
                    )
                )
    steps.sort(key=lambda item: (item.procedure_id, item.step_order))
    return steps, gaps


def load_visual_steps(
    procedures: Sequence[LoadedProcedure],
    *,
    directory: Path | None = None,
    root: Path | None = None,
) -> list[VisualStep]:
    """Join every screenshot-bearing step in *procedures* to its sidecar.

    Strict by design. Every problem across every step is collected and raised
    together, so one run tells you everything that is wrong instead of making
    you fix them one at a time. Use :func:`find_screenshot_gaps` when you need
    to know what is missing without failing, or :func:`collect_visual_steps`
    when you need both.
    """
    steps, gaps = collect_visual_steps(procedures, directory=directory, root=root)
    if gaps:
        failures = [
            f"{gap.procedure_id} {gap.step_id} ({gap.screenshot_id}): {problem}"
            for gap in gaps
            for problem in gap.problems
        ]
        raise VisualStepError("unusable screenshot metadata:\n  - " + "\n  - ".join(failures))
    return steps


def find_screenshot_gaps(
    procedures: Sequence[LoadedProcedure],
    *,
    directory: Path | None = None,
    root: Path | None = None,
) -> list[ScreenshotGap]:
    """Report declared screenshots that are not usable, without raising.

    The corpus declares more screenshots than have been generated so far, and a
    silent skip is exactly the failure mode this module exists to prevent. This
    gives a caller the list explicitly so the shortfall is visible and can be
    asserted on.
    """
    return collect_visual_steps(procedures, directory=directory, root=root)[1]


def _build(procedure: Procedure, step: Step, directory: Path, base: Path) -> tuple[VisualStep | None, list[str]]:
    """Build one record, or return the reasons it cannot be built."""
    screenshot_id = str(step.screenshot)

    if not step.screen or not step.ui_element:
        return None, [
            "a screenshot step must name both 'screen' and 'ui_element'; "
            "there is nothing to caption or highlight otherwise"
        ]

    path = sidecar_path(screenshot_id, directory)
    if not path.is_file():
        return None, [f"no sidecar JSON at {_display(path, base)}; run: make screenshots"]

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"sidecar {_display(path, base)} is unreadable: {exc}"]
    if not isinstance(raw, dict):
        return None, [f"sidecar {_display(path, base)} must contain a JSON object"]

    problems = _metadata_disagreements(procedure, step, raw, screenshot_id)
    problems += _image_disagreements(raw, base, path)
    if problems:
        return None, problems

    try:
        visual = VisualStep(
            # The procedure YAML is authoritative for everything it owns. The
            # sidecar's copies were only compared above; they are not the source.
            procedure_id=procedure.procedure_id,
            procedure_version=procedure.version,
            step_id=step.step_id,
            step_order=step.number,
            screen_name=step.screen,
            control_name=step.ui_element,
            instruction=step.instruction.strip(),
            # Carried on the record rather than looked up again later: the chunk
            # this feeds has to be retrievable by an Arabic query, and re-joining
            # a step to its YAML downstream would be a second chance to pair the
            # wrong texts together.
            instruction_ar=(step.instruction_ar or "").strip() or None,
            expected_result=step.expected_result,
            screenshot_id=screenshot_id,
            image_path=str(raw.get("image_path") or ""),
            image_width=int(raw["width"]),
            image_height=int(raw["height"]),
            image_sha256=str(raw["sha256"]),
            image_description=LocalizedText.model_validate(raw.get("image_description") or {}),
            dom_text=str(raw.get("dom_text") or ""),
            ocr_text=raw.get("ocr_text"),
            language=str(raw.get("language") or ""),
            highlight=_highlight(raw),
        )
    except Exception as exc:
        # Broad on purpose: a malformed sidecar is a data problem to report
        # alongside the others, not a traceback out of the middle of a build.
        return None, [f"sidecar {_display(path, base)} does not describe a usable visual step: {exc}"]
    return visual, []


def _metadata_disagreements(procedure: Procedure, step: Step, raw: dict[str, Any], screenshot_id: str) -> list[str]:
    """Compare the sidecar's idea of the step against the YAML's."""
    checks = (
        ("screenshot_id", raw.get("screenshot_id"), screenshot_id),
        ("procedure_id", raw.get("procedure_id"), procedure.procedure_id),
        ("procedure_version", raw.get("procedure_version"), procedure.version),
        ("step_order", raw.get("step_order"), step.number),
        ("screen_name", raw.get("screen_name"), step.screen),
        ("control_name", raw.get("control_name"), step.ui_element),
    )
    return [
        f"sidecar {field}={found!r} disagrees with the procedure ({expected!r})"
        for field, found, expected in checks
        if found != expected
    ]


def _image_disagreements(raw: dict[str, Any], base: Path, path: Path) -> list[str]:
    """Prove the PNG on disk is the PNG the sidecar was written for."""
    relative = str(raw.get("image_path") or "")
    if not relative:
        return [f"sidecar {_display(path, base)} does not record an image_path"]

    recorded = str(raw.get("sha256") or "")
    if not recorded:
        return [f"sidecar {_display(path, base)} does not record a sha256 for {relative}"]

    image = base / relative
    if not image.is_file():
        return [f"screenshot {relative} referenced by {_display(path, base)} is not on disk"]

    actual = hashlib.sha256(image.read_bytes()).hexdigest()
    if actual != recorded:
        return [
            f"screenshot {relative} has sha256 {actual} but the sidecar records "
            f"{recorded}; the image is stale, re-run: make screenshots"
        ]
    return []


def _highlight(raw: dict[str, Any]) -> HighlightBox:
    highlight = raw.get("highlight") or {}
    if not isinstance(highlight, dict):
        raise ValueError("highlight must be a JSON object")
    return HighlightBox(
        control_name=str(highlight.get("control_name") or ""),
        x=float(highlight["x"]),
        y=float(highlight["y"]),
        width=float(highlight["width"]),
        height=float(highlight["height"]),
    )


def _display(path: Path, base: Path) -> str:
    """Render *path* for a message without leaking an absolute path."""
    try:
        return str(path.resolve().relative_to(base.resolve()).as_posix())
    except ValueError:
        return path.name

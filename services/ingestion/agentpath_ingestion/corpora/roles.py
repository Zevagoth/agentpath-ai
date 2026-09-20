"""What kind of figure is this, and what should the system do with it?

Two roles need different handling, and conflating them is what made the Stage 2B1
image numbers look worse than the product problem actually is.

**A UI screenshot** - an interface with an arrow pointing at the control to press - does
not need to be *chosen*. Once the right procedure is found, every one of its screenshots
is relevant, in order, and picking one with a ranking model can only hide a correct
image. These are handled by showing all of them; see
``agentpath.corpus_retrieval.gallery``.

**A workflow diagram** - a decision flowchart with questions, branches and arrows - has
to be *understood*, once, at ingestion time, so an agent can be told which path their
case falls down instead of being handed a picture to trace. See
``agentpath_ingestion.corpora.workflow``.

How an image gets its role
--------------------------
Three methods, and the record says which one was used, because they are not equally
trustworthy.

``corpus_metadata`` - the corpus said so. Best, and unavailable for MRAMG-Manual, which
ships no type information at all (and no captions either).

``vision_model`` - a local vision model looked at the pixels. Used only on images a cheap
prefilter flagged as *possibly* a workflow diagram, because that is the only decision the
model's answer changes: UI screenshots and everything else are displayed identically, so
paying 2,599 model calls to separate two roles that behave the same would be spending an
hour to change nothing.

``prefilter`` - no model looked. The image is recorded as :attr:`ImageRole.OTHER` with a
warning saying so. This is the honest default: it is not a claim that the image is not a
workflow diagram, it is a statement that nothing checked.

The prefilter's recall is the acknowledged weakness, and it is stated rather than
estimated away: a workflow diagram that is small *and* has no flow vocabulary near it
would be missed. On MRAMG-Manual that limit was probed by classifying 60 images the
prefilter rejected, drawn at random with a recorded seed - see ``docs/IMAGE_ROLES.md``.
"""

from __future__ import annotations

import base64
import json
import re
import struct
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import httpx
import structlog

from agentpath_ingestion.corpora.base import CorpusBundle, CorpusImage, write_jsonl

logger = structlog.get_logger(__name__)


class ImageRole(StrEnum):
    """What the system should do with a figure."""

    WORKFLOW_DIAGRAM = "workflow_diagram"
    """A decision flowchart: questions, branches, arrows that route the reader.

    Extracted into a searchable schema at ingestion. A linear sequence of steps with
    arrows is *not* this - it has no decision to route - and neither is a table, however
    much it is about troubleshooting.
    """

    UI_SCREENSHOT = "ui_screenshot"
    """An interface, usually with an arrow or circle marking what to press.

    Never selected by a ranking model. Shown in full, in document order.
    """

    OTHER = "other"
    """Everything else, and the default when nothing has looked.

    Product photos, exploded parts diagrams, wiring schematics, tables of text, single
    step illustrations. Displayed exactly like a UI screenshot, which is why leaving an
    image here costs nothing behaviourally.
    """


class RoleMethod(StrEnum):
    """How the role was decided. Recorded because they differ in trustworthiness."""

    CORPUS_METADATA = "corpus_metadata"
    VISION_MODEL = "vision_model"
    PREFILTER = "prefilter"


#: Confidence recorded for an image no model examined. Deliberately low: the record is
#: "nothing checked", not "this is not a diagram".
UNEXAMINED_CONFIDENCE = 0.0

#: Minimum area for the prefilter to consider an image a possible diagram. A decision
#: flowchart is a page-scale figure; a 468x282 control-panel callout - the corpus median -
#: cannot hold one. Set well below the smallest plausible diagram rather than tuned.
MIN_DIAGRAM_AREA = 250_000

#: Words that appear near a flowchart and not near a product photo. Deliberately
#: over-inclusive: this decides who gets *looked at*, so a false positive costs one model
#: call and a false negative costs a missed diagram.
FLOW_WORDS = (
    "flowchart",
    "flow chart",
    "diagram",
    "chart",
    "decision",
    "if yes",
    "if no",
    "go to step",
    "proceed to",
    "troubleshoot",
)

_BRANCH = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

#: Markers carrying frame dimensions. SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC).
_SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})

ROLES_NAME = "roles.jsonl"


@dataclass(frozen=True, slots=True)
class ImageRoleRecord:
    """One image's role, and everything needed to doubt it."""

    image_id: str
    document_id: str
    role: ImageRole
    method: RoleMethod
    confidence: float
    reason: str
    warnings: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    #: What the model reported seeing. ``None`` when no model looked.
    has_arrows: bool | None = None
    has_branches: bool | None = None
    #: Text the model could read, truncated. Feeds the searchable index for diagrams.
    visible_text: str = ""
    model: str | None = None
    seconds: float | None = None

    @property
    def examined(self) -> bool:
        return self.method is not RoleMethod.PREFILTER

    @property
    def is_workflow(self) -> bool:
        return self.role is ImageRole.WORKFLOW_DIAGRAM

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "document_id": self.document_id,
            "role": self.role.value,
            "method": self.method.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "warnings": list(self.warnings),
            "width": self.width,
            "height": self.height,
            "has_arrows": self.has_arrows,
            "has_branches": self.has_branches,
            "visible_text": self.visible_text,
            "model": self.model,
            "seconds": round(self.seconds, 2) if self.seconds is not None else None,
        }


@dataclass
class RoleReport:
    """What a classification run produced."""

    corpus_id: str
    records: list[ImageRoleRecord] = field(default_factory=list)
    examined: int = 0
    candidates: int = 0
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for record in self.records:
            totals[record.role.value] = totals.get(record.role.value, 0) + 1
        return dict(sorted(totals.items()))

    @property
    def workflows(self) -> list[ImageRoleRecord]:
        return [record for record in self.records if record.is_workflow]


# --------------------------------------------------------------------------- #
# Geometry, without an image library
# --------------------------------------------------------------------------- #


def jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """``(width, height)`` from a JPEG's SOF marker.

    Pillow is not installed, and 2,599 subprocess calls to ImageMagick would cost more
    than reading a few bytes per file. Returns ``None`` for anything that is not a JPEG
    this can parse, and the caller treats that as "geometry unknown" rather than as an
    error - 86 of the corpus's images are not parseable here and are simply not
    prefiltered on size.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"\xff\xd8"):
        return None
    index, end = 2, len(data)
    while index < end - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if marker in _SOF_MARKERS:
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        index += 2 + length
    return None


def is_diagram_candidate(image: CorpusImage, window: str, dimensions: tuple[int, int] | None) -> tuple[bool, str]:
    """Whether *image* is worth showing a vision model, and why.

    Two signals, either of which is enough, because they fail in different directions:
    a diagram with no nearby vocabulary is still page-sized, and a diagram whose page was
    split small is still described as a chart.
    """
    reasons: list[str] = []
    lowered = window.lower()

    found = [word for word in FLOW_WORDS if word in lowered]
    branches = len({match.lower() for match in _BRANCH.findall(lowered)})
    has_question = "?" in window

    if branches >= 2 and has_question:
        reasons.append("surrounding text asks a question and offers yes/no")
    if found:
        reasons.append(f"surrounding text mentions {', '.join(found[:3])}")

    area = (dimensions[0] * dimensions[1]) if dimensions else None
    large = area is not None and area >= MIN_DIAGRAM_AREA
    if large:
        reasons.append(f"{area / 1e6:.2f} MP")

    # Vocabulary alone is not enough - "troubleshoot" sits beside tables constantly - and
    # size alone is not either. One of each, or the strong branch signal on its own.
    strong = branches >= 2 and has_question
    candidate = strong or (bool(found) and large)
    return candidate, "; ".join(reasons) if reasons else "no diagram signal"


# --------------------------------------------------------------------------- #
# The vision pass
# --------------------------------------------------------------------------- #

CLASSIFY_PROMPT = """You are looking at one figure taken from a product manual.

Classify the figure into exactly one of these roles:

- "workflow_diagram": a decision flowchart. Boxes or nodes joined by arrows, routing the
  reader down different branches depending on an answer or a condition.
- "ui_screenshot": a screenshot or drawing of a software or device interface, often with
  arrows or circles pointing at a control the reader should press.
- "other": anything else. Product photos, exploded parts diagrams, wiring schematics,
  tables of text, control-panel illustrations, single step illustrations.

Rules:
- A table of text is NOT a workflow_diagram, even when it is about troubleshooting.
- A linear sequence of numbered steps is NOT a workflow_diagram. A workflow_diagram must
  branch on a condition.
- Use exactly one of the three role values above. Do not invent another.

Reply with JSON only and nothing else:
{"role": "...", "confidence": 0.0 to 1.0, "reason": "one short sentence",
 "has_arrows": true or false, "has_branches": true or false,
 "visible_text": "up to 400 characters of text you can actually read, or empty"}

If you cannot tell, use "other" with a low confidence and say so in the reason."""


class VisionClassifier:
    """Asks a local vision model what a figure is.

    Nothing about the expected answer reaches the prompt: no ground truth, no question
    text, no surrounding document text. The model sees the pixels and the instruction.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> tuple[bool, str]:
        """Whether the model exists and reports a vision capability.

        Checked rather than assumed: a text-only model answers this prompt fluently and
        entirely from imagination, which is the one failure mode that would silently
        produce a corpus full of confident nonsense.
        """
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(f"{self._base_url}/api/show", json={"model": self._model})
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return False, f"cannot reach {self._base_url}: {exc}"
        capabilities = body.get("capabilities") or []
        if "vision" not in capabilities:
            return False, f"model {self._model!r} does not report a vision capability: {capabilities}"
        return True, f"{self._model} ({', '.join(capabilities)})"

    def classify(self, path: Path) -> tuple[dict[str, object], float, list[str]]:
        """Returns ``(parsed, seconds, warnings)``. Never raises on a bad answer."""
        warnings: list[str] = []
        payload = {
            "model": self._model,
            "prompt": CLASSIFY_PROMPT,
            "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
            "stream": False,
            "think": False,
            # Zero temperature: this is a labelling task and a re-run should agree with
            # itself, so the role file is reproducible.
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 500},
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(f"{self._base_url}/api/generate", json=payload)
                response.raise_for_status()
                raw = str((response.json() or {}).get("response") or "")
        except (httpx.HTTPError, ValueError) as exc:
            return {}, time.perf_counter() - started, [f"vision call failed: {exc}"]
        seconds = time.perf_counter() - started

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}, seconds, ["model did not return JSON"]
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {}, seconds, [f"model returned unparseable JSON: {exc}"]
        if not isinstance(parsed, dict):
            return {}, seconds, ["model returned JSON that is not an object"]
        return parsed, seconds, warnings


def _role_from_answer(parsed: dict[str, object]) -> tuple[ImageRole, list[str]]:
    """Map the model's ``role`` onto the enum, recording any deviation.

    Measured: the model invents a fourth value. Over 78 images it answered "table" nine
    times despite the prompt offering three options. Such an answer is mapped to
    :attr:`ImageRole.OTHER` - which is where a table belongs - and the deviation is kept
    as a warning rather than silently normalised.
    """
    warnings: list[str] = []
    raw = str(parsed.get("role") or "").strip().lower()
    try:
        return ImageRole(raw), warnings
    except ValueError:
        warnings.append(f"model returned role {raw!r}, which is not one of the three offered; recorded as other")
        return ImageRole.OTHER, warnings


def classify_bundle(
    bundle: CorpusBundle,
    *,
    images_root: Path,
    classifier: VisionClassifier | None = None,
    only: Sequence[str] | None = None,
) -> RoleReport:
    """Assign a role to every image in *bundle*.

    The vision model runs only on prefilter candidates. Everything else is recorded as
    :attr:`ImageRole.OTHER` with a warning that nothing examined it, which is honest and
    costs nothing: the display path treats ``other`` and ``ui_screenshot`` identically.

    ``only`` restricts the run to named image ids, for re-classifying a handful without
    redoing the corpus.
    """
    report = RoleReport(corpus_id=bundle.corpus_id)
    started = time.perf_counter()

    if classifier is not None:
        ok, detail = classifier.available()
        if not ok:
            report.problems.append(f"vision model unavailable, every image left unexamined: {detail}")
            classifier = None
        else:
            logger.info("corpus.roles.vision_ready", corpus=bundle.corpus_id, detail=detail)

    wanted = set(only) if only else None
    seen: set[str] = set()

    for document in bundle.documents:
        for image in document.images:
            if image.image_id in seen:
                continue
            seen.add(image.image_id)
            if wanted is not None and image.image_id not in wanted:
                continue
            report.records.append(
                _classify_one(
                    bundle,
                    document.text,
                    image,
                    images_root=images_root,
                    classifier=classifier,
                    report=report,
                )
            )

    report.seconds = round(time.perf_counter() - started, 1)
    logger.info(
        "corpus.roles.done",
        corpus=bundle.corpus_id,
        images=len(report.records),
        candidates=report.candidates,
        examined=report.examined,
        seconds=report.seconds,
        **report.counts,
    )
    return report


def _classify_one(
    bundle: CorpusBundle,
    document_text: str,
    image: CorpusImage,
    *,
    images_root: Path,
    classifier: VisionClassifier | None,
    report: RoleReport,
) -> ImageRoleRecord:
    from agentpath_ingestion.corpora.base import context_window

    path = _resolve(image, images_root)
    dimensions = jpeg_dimensions(path) if path is not None else None
    window = context_window(document_text, image.char_offset, before=600, after=300)
    candidate, why = is_diagram_candidate(image, window, dimensions)

    warnings: list[str] = []
    if path is None:
        warnings.append("image bytes are not on disk; classified on text signal only")
    if dimensions is None and path is not None:
        warnings.append("dimensions unreadable; size prefilter did not apply")

    if not candidate:
        return ImageRoleRecord(
            image_id=image.image_id,
            document_id=image.document_id,
            role=ImageRole.OTHER,
            method=RoleMethod.PREFILTER,
            confidence=UNEXAMINED_CONFIDENCE,
            reason=f"no vision model examined this image: {why}",
            warnings=(*warnings, "not examined by a model; 'other' here means unchecked, not checked-and-negative"),
            width=dimensions[0] if dimensions else None,
            height=dimensions[1] if dimensions else None,
        )

    report.candidates += 1
    if classifier is None or path is None:
        return ImageRoleRecord(
            image_id=image.image_id,
            document_id=image.document_id,
            role=ImageRole.OTHER,
            method=RoleMethod.PREFILTER,
            confidence=UNEXAMINED_CONFIDENCE,
            reason=f"flagged as a possible diagram but no model was available: {why}",
            warnings=(*warnings, "candidate left unexamined"),
            width=dimensions[0] if dimensions else None,
            height=dimensions[1] if dimensions else None,
        )

    parsed, seconds, call_warnings = classifier.classify(path)
    warnings.extend(call_warnings)
    if not parsed:
        return ImageRoleRecord(
            image_id=image.image_id,
            document_id=image.document_id,
            role=ImageRole.OTHER,
            method=RoleMethod.PREFILTER,
            confidence=UNEXAMINED_CONFIDENCE,
            reason=f"the model gave no usable answer: {why}",
            warnings=tuple(warnings),
            width=dimensions[0] if dimensions else None,
            height=dimensions[1] if dimensions else None,
            model=classifier.model,
            seconds=round(seconds, 2),
        )

    role, role_warnings = _role_from_answer(parsed)
    warnings.extend(role_warnings)
    report.examined += 1

    confidence = parsed.get("confidence")
    has_branches = parsed.get("has_branches")
    has_arrows = parsed.get("has_arrows")

    # A workflow diagram that the model itself says does not branch is not one. Measured:
    # every image this model called a workflow_diagram on MRAMG-Manual reported
    # has_branches false, and all three turned out to be a lookup matrix or a linear
    # assembly sequence. Trusting the label over the model's own evidence would have put
    # three non-workflows into the extraction pipeline.
    if role is ImageRole.WORKFLOW_DIAGRAM and has_branches is False:
        warnings.append("model labelled this a workflow_diagram but reported no branches; recorded as other")
        role = ImageRole.OTHER

    return ImageRoleRecord(
        image_id=image.image_id,
        document_id=image.document_id,
        role=role,
        method=RoleMethod.VISION_MODEL,
        confidence=float(confidence) if isinstance(confidence, int | float) else 0.0,
        reason=str(parsed.get("reason") or "")[:300],
        warnings=tuple(warnings),
        width=dimensions[0] if dimensions else None,
        height=dimensions[1] if dimensions else None,
        has_arrows=bool(has_arrows) if isinstance(has_arrows, bool) else None,
        has_branches=bool(has_branches) if isinstance(has_branches, bool) else None,
        visible_text=str(parsed.get("visible_text") or "")[:400],
        model=classifier.model,
        seconds=round(seconds, 2),
    )


def _resolve(image: CorpusImage, images_root: Path) -> Path | None:
    from agentpath.paths import repo_root

    if image.local_path:
        candidate = repo_root() / image.local_path
        if candidate.is_file():
            return candidate
    for suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        candidate = images_root / f"{image.image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def write_roles(report: RoleReport, target: Path) -> Path:
    """Write ``roles.jsonl`` beside the rest of the bundle."""
    path = target / ROLES_NAME
    write_jsonl(path, [record.to_dict() for record in report.records])
    return path


def read_roles(path: Path) -> Iterable[dict[str, object]]:
    from agentpath_ingestion.corpora.base import iter_jsonl

    for row in iter_jsonl(path):
        if isinstance(row, dict):
            yield row


__all__ = [
    "CLASSIFY_PROMPT",
    "FLOW_WORDS",
    "MIN_DIAGRAM_AREA",
    "ROLES_NAME",
    "UNEXAMINED_CONFIDENCE",
    "ImageRole",
    "ImageRoleRecord",
    "RoleMethod",
    "RoleReport",
    "VisionClassifier",
    "classify_bundle",
    "is_diagram_candidate",
    "jpeg_dimensions",
    "read_roles",
    "write_roles",
]

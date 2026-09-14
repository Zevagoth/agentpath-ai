"""The set of screenshots this service will serve, resolved once at startup.

RAGFlow stores the image bytes for an image-bearing chunk and exposes them at
``GET /api/v1/documents/images/{img_id}``, which requires a RAGFlow API key. A
browser cannot send that key and we will not hand a browser ours, so the
screenshots are served from this service instead. They are committed in the
repository next to the sidecar JSON that ``make screenshots`` wrote, which makes
them a fixed, known set rather than user-supplied content.

Three properties the design turns on.

**A closed set, built before any request.** :meth:`ScreenshotCatalogue.load`
lists the screenshots directory, validates each sidecar, reads and verifies each
PNG, and holds the bytes. After that the request path is a dictionary lookup on
a validated key. It never touches the filesystem, never parses JSON and never
hashes an image, so no part of the ingestion-time enrichment can end up on a
user's latency budget - and, more importantly, no request input can reach a path
operation. There is nothing to traverse *to*.

**No path is ever composed from request input, or from sidecar content.** The
PNG for an id is ``directory / f"{id}.png"`` where ``id`` already matched
:data:`SCREENSHOT_ID_PATTERN` and ``directory`` came from configuration. The
sidecar's own ``image_path`` is only cross-checked against that, never joined.

**No absolute path leaves this module.** :class:`Screenshot` carries the
repository-relative path for logging and nothing else. The absolute path exists
only as a local variable during :meth:`ScreenshotCatalogue.load`.

Duplicated field names, on purpose
----------------------------------
The sidecar field names read below (``screenshot_id``, ``sha256``, ``width``,
``height``, ``dom_text``'s siblings and so on) are also declared in
``agentpath_ingestion.visual_steps.VisualStep``, and the id pattern is also
declared there as ``SCREENSHOT_ID_PATTERN``. Importing that module would be the
DRY move and it is the wrong one: ``tests/test_visual_steps.py``'s
``test_the_api_does_not_import_the_ingestion_package`` asserts on the import
graph that ``agentpath.api.app`` pulls in zero ``agentpath_ingestion`` modules,
because that is what guarantees YAML loading, JSON parsing and image hashing
cannot run during a request. Re-stating a handful of field names is a cheaper
price than weakening that guarantee, so it is deliberate rather than an
oversight. The two are kept honest by the sidecars themselves: a rename on the
ingestion side makes :meth:`ScreenshotCatalogue.load` report the field as
missing, and ``tests/test_screenshots.py`` loads the committed sidecars and
asserts the catalogue is complete.

This module also reads a strict subset of the sidecar: the API has no use for
``dom_text``, ``source_html_path``, ``perceptual_hash`` or the generator's notes,
so they are not read and cannot be served.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import structlog

from agentpath.paths import screenshots_dir

logger = structlog.get_logger(__name__)

#: The only media type served. Asserted against the file's magic bytes at load,
#: so the declared ``Content-Type`` is a fact about the bytes rather than a guess
#: from a file extension.
SCREENSHOT_MEDIA_TYPE = "image/png"

#: PNG signature, per the format specification.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Route prefix for the serving endpoint. Defined here, next to the catalogue, so
#: the URL the API advertises in a response and the URL the route is registered
#: at cannot drift apart.
SCREENSHOT_URL_PREFIX = "/api/screenshots"

#: Mirrors ``agentpath_ingestion.visual_steps.SCREENSHOT_ID_PATTERN``. Lowercase
#: alphanumerics in hyphen-separated groups: no dot, no slash, no backslash, no
#: percent sign, no NUL, no whitespace, no uppercase. Traversal is unrepresentable
#: in this alphabet rather than filtered out of it.
SCREENSHOT_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

#: Length bound applied before the pattern. The pattern is linear so a long input
#: is not a ReDoS risk, but a megabyte of path separators has no business being
#: matched at all, and the longest real id is 30 characters.
MAX_SCREENSHOT_ID_LENGTH = 64

#: Refuse to hold an implausibly large capture in memory. The three committed
#: 1440x900 PNGs are 206-221 KiB each; this is roughly forty times the largest.
MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024


def is_valid_screenshot_id(value: object) -> bool:
    """Whether *value* is shaped like a screenshot id.

    Deliberately total: any object, any length, any encoding returns a bool
    rather than raising, because the caller is a request handler and a malformed
    id is an expected input, not an error condition.
    """
    if not isinstance(value, str) or not value:
        return False
    if len(value) > MAX_SCREENSHOT_ID_LENGTH:
        return False
    # ``fullmatch``, not ``match``: an anchored ``$`` would also accept a trailing
    # newline, which is exactly the kind of smuggled byte this check is for.
    return SCREENSHOT_ID_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class ScreenshotHighlight:
    """Where the highlighted control sits in the image, in CSS pixels.

    Top-left origin, ``deviceScaleFactor`` 1, measured by the generator from the
    live element. The UI needs it to draw the same box over the image at whatever
    size it renders.
    """

    control_name: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One servable screenshot: its metadata, and its bytes.

    ``image`` is excluded from ``repr`` so a debug log or an exception repr cannot
    dump 200 KiB of PNG into a log file. The wire format is a separate model in
    ``agentpath.api.schemas``; nothing here is serialised directly, which is what
    keeps ``image`` and ``relative_path`` out of responses.
    """

    screenshot_id: str
    procedure_id: str
    procedure_version: str
    step_order: int
    screen_name: str
    control_name: str
    width: int
    height: int
    sha256: str
    description_en: str
    description_ar: str
    watermark: str
    language: str
    #: Repository-relative, for log lines. Never an absolute path, and never
    #: returned to a client.
    relative_path: str
    image: bytes = field(repr=False, default=b"")
    highlight: ScreenshotHighlight | None = None

    @property
    def byte_count(self) -> int:
        return len(self.image)

    @property
    def url_path(self) -> str:
        """The path a browser fetches. Server-issued, so no client builds it."""
        return f"{SCREENSHOT_URL_PREFIX}/{self.screenshot_id}"

    @property
    def etag(self) -> str:
        """A strong validator, quoted as HTTP requires.

        The content is immutable for a given sha256, but the *URL* is not: it is
        keyed by the stable id, and regenerating the capture replaces the bytes
        behind it. So the sha256 is spent on an ``ETag`` - which stays correct
        across a regeneration - rather than on a long ``immutable`` lifetime,
        which would not.
        """
        return f'"sha256-{self.sha256}"'


@dataclass(frozen=True, slots=True)
class ScreenshotCatalogue:
    """Every screenshot this service can serve, keyed by id.

    Immutable after construction. ``problems`` records sidecars that were
    rejected, so a missing screenshot is reported rather than looking like a
    screenshot that was never declared.
    """

    screenshots: Mapping[str, Screenshot] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> ScreenshotCatalogue:
        """A catalogue that serves nothing.

        The default for :class:`~agentpath.api.deps.ServiceContext`, so a test or
        a harness that does not care about screenshots gets a service where every
        screenshot id is a 404 rather than a service that reads the filesystem.
        """
        return cls(screenshots={})

    @classmethod
    def load(cls, directory: Path | None = None) -> ScreenshotCatalogue:
        """Read and verify every screenshot in *directory*.

        Called from :func:`~agentpath.api.deps.build_context` at startup. Cost on
        the committed slice: three sidecars, three PNGs, ~650 KiB, one sha256
        each. Nothing here runs again for the lifetime of the process.

        A sidecar that cannot be trusted is skipped and recorded in
        :attr:`problems`, not raised: one stale capture should degrade to "that
        screenshot is unavailable" rather than to "the API will not start".
        """
        base = directory if directory is not None else screenshots_dir()
        records: dict[str, Screenshot] = {}
        problems: list[str] = []

        if not base.is_dir():
            # Named by the directory's own name only: the configured location can
            # be an absolute path, and this string reaches log lines.
            problems.append(f"screenshot directory {base.name!r} does not exist; run: make screenshots")
            return cls(screenshots={}, problems=tuple(problems))

        for sidecar in sorted(base.glob("*.json")):
            record, failures = _load_one(sidecar, base)
            if record is not None:
                records[record.screenshot_id] = record
            problems.extend(failures)

        logger.info(
            "screenshots.loaded",
            count=len(records),
            total_bytes=sum(record.byte_count for record in records.values()),
            problems=len(problems),
        )
        for problem in problems:
            logger.warning("screenshots.unusable", detail=problem)
        return cls(screenshots=dict(sorted(records.items())), problems=tuple(problems))

    # -- lookup ------------------------------------------------------------ #

    def resolve(self, screenshot_id: object) -> Screenshot | None:
        """The screenshot for *screenshot_id*, or ``None``.

        The whole security model of the serving route is this method: shape check
        first, then a lookup in a set built at startup. Anything that is not a
        key returns ``None``, and there is no branch in which the argument
        reaches a filesystem call.
        """
        if not is_valid_screenshot_id(screenshot_id):
            return None
        return self.screenshots.get(screenshot_id)  # type: ignore[arg-type]

    def for_chunk_text(self, text: str) -> Screenshot | None:
        """The screenshot a retrieved chunk is about, if it names one.

        Chunk text is the mapping, because it is the only handle that survives
        everything RAGFlow does to a document. Both kinds of chunk carry the id
        verbatim: the rendered Markdown writes ``**Screenshot reference:**
        `agentdesk-...`​`` under the step heading, and
        ``build_visual_chunk_content`` writes ``Screenshot agentdesk-... (1440x900)
        shows:``. So a text chunk for the step and the image-bearing chunk for the
        step both resolve to the same screenshot, which is what the UI wants.

        Matching is a substring test against the catalogue's own keys, not a
        pattern extracted from the text. That direction matters: the text is
        retrieved content and therefore untrusted, so it is only ever allowed to
        *select* from a set we built, never to describe something to go and find.

        When a chunk spans two screenshot-bearing steps, the earliest-mentioned
        wins, because that is the step the passage leads with. Ties break on the
        id so the result is stable.
        """
        if not text:
            return None
        best: tuple[int, str] | None = None
        for screenshot_id in self.screenshots:
            position = text.find(screenshot_id)
            if position < 0:
                continue
            if best is None or (position, screenshot_id) < best:
                best = (position, screenshot_id)
        return self.screenshots[best[1]] if best is not None else None

    # -- reporting --------------------------------------------------------- #

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.screenshots)

    @property
    def total_bytes(self) -> int:
        return sum(record.byte_count for record in self.screenshots.values())

    def __len__(self) -> int:
        return len(self.screenshots)

    def __iter__(self) -> Iterator[Screenshot]:
        return iter(self.screenshots.values())

    def __contains__(self, screenshot_id: object) -> bool:
        return self.resolve(screenshot_id) is not None


# --------------------------------------------------------------------------- #
# Loading one sidecar
# --------------------------------------------------------------------------- #


def _load_one(sidecar: Path, base: Path) -> tuple[Screenshot | None, list[str]]:
    """Build one record from a sidecar, or say why it cannot be built.

    Ordering is the point: the id is validated and reconciled with the filename
    *before* the PNG path is formed, and the PNG path is formed from that id
    rather than from anything the sidecar says.
    """
    name = sidecar.name
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"sidecar {name} is unreadable: {exc}"]
    if not isinstance(raw, dict):
        return None, [f"sidecar {name} must contain a JSON object"]

    screenshot_id = raw.get("screenshot_id")
    if not is_valid_screenshot_id(screenshot_id):
        return None, [f"sidecar {name} declares an unusable screenshot_id {screenshot_id!r}"]
    assert isinstance(screenshot_id, str)  # narrowed by is_valid_screenshot_id
    if screenshot_id != sidecar.stem:
        # Otherwise a sidecar could register itself under someone else's id and
        # shadow a real screenshot.
        return None, [f"sidecar {name} declares screenshot_id {screenshot_id!r}, which is not its own name"]

    recorded_path = str(raw.get("image_path") or "")
    expected_name = f"{screenshot_id}.png"
    if PurePosixPath(recorded_path).name != expected_name:
        return None, [f"sidecar {name} records image_path {recorded_path!r}, which is not {expected_name}"]

    # Composed from the validated id and the configured directory. The sidecar's
    # own path string was compared above and is not used to build this.
    image_path = base / expected_name
    if not image_path.is_file():
        return None, [f"screenshot {expected_name} is missing from the screenshots directory"]

    size = image_path.stat().st_size
    if size > MAX_SCREENSHOT_BYTES:
        return None, [f"screenshot {expected_name} is {size} bytes, above the {MAX_SCREENSHOT_BYTES} byte limit"]

    try:
        image = image_path.read_bytes()
    except OSError as exc:
        return None, [f"screenshot {expected_name} could not be read: {exc.strerror or exc.__class__.__name__}"]

    if not image.startswith(PNG_MAGIC):
        return None, [f"screenshot {expected_name} is not a PNG, so it cannot be served as {SCREENSHOT_MEDIA_TYPE}"]

    digest = hashlib.sha256(image).hexdigest()
    recorded_digest = str(raw.get("sha256") or "")
    if digest != recorded_digest:
        return None, [
            f"screenshot {expected_name} has sha256 {digest} but sidecar {name} records "
            f"{recorded_digest or 'nothing'}; the image is stale, re-run: make screenshots"
        ]

    try:
        record = Screenshot(
            screenshot_id=screenshot_id,
            procedure_id=str(raw["procedure_id"]),
            procedure_version=str(raw["procedure_version"]),
            step_order=int(raw["step_order"]),
            screen_name=str(raw["screen_name"]),
            control_name=str(raw["control_name"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            sha256=digest,
            description_en=str((raw.get("image_description") or {}).get("en") or ""),
            description_ar=str((raw.get("image_description") or {}).get("ar") or ""),
            watermark=str(raw.get("watermark") or ""),
            language=str(raw.get("language") or ""),
            relative_path=recorded_path,
            image=image,
            highlight=_highlight(raw.get("highlight")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"sidecar {name} does not describe a servable screenshot: {exc}"]

    if record.width <= 0 or record.height <= 0:
        return None, [f"sidecar {name} records non-positive dimensions"]
    return record, []


def _highlight(raw: object) -> ScreenshotHighlight | None:
    """Read the highlight box, or ``None`` when it is absent or unusable.

    A missing box is not a reason to withhold the screenshot: the image is still
    correct and useful, the UI just cannot draw the overlay.
    """
    if not isinstance(raw, dict):
        return None
    try:
        return ScreenshotHighlight(
            control_name=str(raw.get("control_name") or ""),
            x=float(raw["x"]),
            y=float(raw["y"]),
            width=float(raw["width"]),
            height=float(raw["height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None

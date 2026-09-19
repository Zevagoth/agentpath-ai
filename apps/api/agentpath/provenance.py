"""Where a claim came from, in one shape that works for every corpus.

Stage 2B1 asks for every result and every step to be verifiable by hand. That is
only possible if an answer carries enough to *find the original* - not a
procedure id on its own, but the document, the position inside it, and a link that
opens there.

Two corpora have to fit the same contract, and they can supply different things:

* **synthetic-procedures** has a procedure id, a version, an effective date, a
  step id and a heading path. Its "original" is a rendered Markdown document in
  this repository, so its link is local.
* **mramg-manual** has a document id, a document title, character offsets and, for
  most images, an upstream URL. It has no version and no effective date.

So every field except :attr:`Provenance.corpus_id` is optional, and a consumer is
expected to render what is present rather than assume a shape. The alternative -
one contract per corpus - would push that branching into the UI and into every
test, and would make "is this answer traceable" a question with two answers.

Three rules this module enforces, because they are the difference between a
provenance link and a liability:

1. **Only ``http`` and ``https`` reach the browser as a link.** ``javascript:``,
   ``data:``, ``file:`` and anything unparseable are refused, not sanitised. See
   :func:`safe_external_url`.
2. **Request input never becomes a filesystem path.** A local source is named by
   an opaque id that must already exist in a registry built at startup, exactly as
   :mod:`agentpath.screenshots` does for captures. Path traversal has nothing to
   traverse when there is no path operation on the request path at all.
3. **A missing link is stated, never invented.** :attr:`Provenance.link` returns
   ``None`` and :attr:`Provenance.link_absent_reason` says why, so the console can
   say "no original source recorded" instead of rendering a dead anchor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit

#: The only schemes that may be handed to a browser as a link.
SAFE_URL_SCHEMES = frozenset({"http", "https"})

#: Schemes that are refused loudly rather than quietly dropped, because seeing one
#: means something upstream is either broken or hostile.
DANGEROUS_URL_SCHEMES = frozenset({"javascript", "data", "vbscript", "file", "blob", "about"})

#: A source id: opaque, flat, and incapable of being a path. No dot, no slash, no
#: backslash, no colon, so it cannot be a relative path, an absolute path, a UNC
#: path or a URL no matter what a caller sends.
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

#: Bounded so a hostile id cannot bloat a log line or a response.
MAX_SOURCE_ID_LENGTH = 128


class SourceKind(StrEnum):
    """What kind of original a piece of evidence has."""

    EXTERNAL_URL = "external_url"
    """An upstream document on the public web. Opened directly."""

    LOCAL_DOCUMENT = "local_document"
    """A document held in this repository and served by the API."""

    NONE = "none"
    """No original is recorded. Stated to the agent, never faked."""


def safe_external_url(raw: object) -> str | None:
    """*raw* if it is a plain ``http``/``https`` URL, else ``None``.

    Deliberately strict and deliberately silent about *why* in the return value -
    the caller gets a link or nothing. Rejections:

    * any scheme outside :data:`SAFE_URL_SCHEMES`, which covers ``javascript:``
      and ``data:`` and also ``file:``, since a link into the serving machine's
      filesystem is not provenance
    * a missing host, which is what ``http:///etc/passwd`` and ``https:`` alone
      look like after parsing
    * embedded newlines or control characters, which are how a single attribute
      value turns into two
    * credentials in the authority, which have no place in a citation

    No attempt is made to repair a rejected value. A URL that needs repairing is
    not one we should be vouching for.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if not candidate or len(candidate) > 2048:
        return None
    # Control characters first: a newline inside an href can terminate the
    # attribute and start another one.
    if any(character in candidate for character in "\r\n\t\x00") or any(
        ord(character) < 0x20 for character in candidate
    ):
        return None

    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme in DANGEROUS_URL_SCHEMES or scheme not in SAFE_URL_SCHEMES:
        return None
    if not parts.hostname:
        return None
    # `user:pass@host` in a provenance link is either a mistake or a leak.
    if parts.username or parts.password:
        return None
    return candidate


def safe_source_id(raw: object) -> str | None:
    """*raw* if it is a usable opaque source id, else ``None``.

    This is the only thing a request may send to identify a local document, and it
    is checked for *shape* here and for *existence* by the registry. Both halves
    matter: the shape check means no request value can ever look like a path, and
    the existence check means an unknown id is a 404 rather than a filesystem
    probe.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if len(candidate) > MAX_SOURCE_ID_LENGTH:
        return None
    return candidate if SOURCE_ID_PATTERN.fullmatch(candidate) else None


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """Where inside a document the evidence sits.

    Every field is optional and they are not alternatives - a corpus may know a
    heading path and character offsets and nothing else, or a page and no heading.
    :attr:`is_precise` is what a caller should branch on before promising to open
    a document "at the right place", because a locator with nothing set means the
    honest offer is the top of the document.
    """

    #: Human-readable path to the place, e.g. ``## Steps > ### Step 5``.
    section: str | None = None
    #: A fragment usable in a URL or by the viewer to scroll, e.g. ``step-5``.
    anchor: str | None = None
    #: 1-based page, for paged originals.
    page: int | None = None
    #: Character span in the document's own text. The most precise thing a
    #: plain-text corpus can offer, and what the local viewer highlights.
    char_start: int | None = None
    char_end: int | None = None

    @property
    def is_precise(self) -> bool:
        """Whether this points somewhere narrower than "the document"."""
        return any(value is not None for value in (self.anchor, self.page, self.char_start, self.section))

    @property
    def summary(self) -> str:
        """One line naming the position, for display beside a citation."""
        parts: list[str] = []
        if self.section:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.char_start is not None:
            end = f"-{self.char_end}" if self.char_end is not None else ""
            parts.append(f"chars {self.char_start}{end}")
        return " · ".join(parts) if parts else "position not recorded"

    def fragment(self) -> str:
        """The URL fragment that opens at this position, without the ``#``.

        Character offsets are emitted as a fragment the local viewer understands,
        because an arbitrary text document has no anchors to link to. An explicit
        ``anchor`` wins when the corpus supplies one.
        """
        if self.anchor:
            return self.anchor
        if self.char_start is not None:
            return f"char={self.char_start}"
        if self.page is not None:
            return f"page={self.page}"
        return ""


@dataclass(frozen=True, slots=True)
class ImageProvenance:
    """One image, and everything needed to verify where it came from.

    ``position_index`` and ``char_offset`` are what make an image checkable in a
    corpus with no captions: they say *this is the Nth image in this document, and
    it belongs at this character*, which a reader can confirm by opening the
    document at that offset.
    """

    image_id: str
    #: Where the API serves the bytes from, e.g. ``/api/corpus-images/Manual04_0``.
    served_url: str | None = None
    #: The upstream URL the dataset recorded, when it recorded one. Validated.
    source_url: str | None = None
    #: Document this image belongs to.
    document_id: str | None = None
    #: The image's ordinal in the document, 0-based. For MRAMG this is which
    #: ``<PIC>`` marker it fills.
    position_index: int | None = None
    #: Character offset of the marker in the document's own text.
    char_offset: int | None = None
    #: Caption, if the corpus has one. Empty for MRAMG-Manual - see
    #: docs/RETRIEVAL_MODES.md.
    caption: str | None = None
    #: The step this image illustrates, for corpora with steps.
    step_id: str | None = None

    @property
    def has_original(self) -> bool:
        return self.source_url is not None

    @property
    def position_summary(self) -> str:
        if self.position_index is None:
            return "position in document not recorded"
        ordinal = self.position_index + 1
        where = f"image {ordinal} in the document"
        if self.char_offset is not None:
            where += f", at character {self.char_offset}"
        return where


@dataclass(frozen=True, slots=True)
class Provenance:
    """The full trail for one piece of evidence.

    Built by the corpus adapter at ingestion time and carried through retrieval to
    the response, so nothing downstream has to reconstruct it - and so a field
    that is absent is absent because the *corpus* lacks it, not because a layer
    dropped it.
    """

    corpus_id: str
    #: The retrieval dataset this lives in. Also the isolation boundary: two
    #: corpora never share one.
    dataset_id: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    #: Present for corpora organised as versioned procedures.
    procedure_id: str | None = None
    procedure_version: str | None = None
    effective_date: str | None = None
    #: The retrieval unit.
    chunk_id: str | None = None
    step_id: str | None = None
    #: An upstream URL for the document, validated. ``None`` when there is none.
    source_url: str | None = None
    #: Opaque id of a local document in the source registry. Never a path.
    source_id: str | None = None
    locator: SourceLocator = field(default_factory=SourceLocator)
    images: tuple[ImageProvenance, ...] = ()

    # -- how to open it ----------------------------------------------------- #

    @property
    def kind(self) -> SourceKind:
        if self.source_url:
            return SourceKind.EXTERNAL_URL
        if self.source_id:
            return SourceKind.LOCAL_DOCUMENT
        return SourceKind.NONE

    def link(self, *, local_prefix: str = "/api/sources") -> str | None:
        """A URL that opens the original at this position, or ``None``.

        The fragment is appended for both kinds, so "open at the section" is one
        code path rather than a local special case. For an external URL that means
        the upstream page's own anchor when the corpus recorded one, and nothing
        when it did not - a fabricated fragment would be a guess about someone
        else's document.
        """
        fragment = self.locator.fragment()
        if self.source_url:
            if fragment and "#" not in self.source_url and self.locator.anchor:
                return f"{self.source_url}#{fragment}"
            return self.source_url
        if self.source_id:
            base = f"{local_prefix}/{self.source_id}"
            return f"{base}#{fragment}" if fragment else base
        return None

    @property
    def link_absent_reason(self) -> str | None:
        """Why there is no link, in words for an agent, or ``None`` if there is one."""
        if self.kind is not SourceKind.NONE:
            return None
        return "No original source is recorded for this corpus entry."

    @property
    def opens_at_position(self) -> bool:
        """Whether the link lands on the evidence rather than the document's top."""
        if self.kind is SourceKind.NONE:
            return False
        if self.kind is SourceKind.EXTERNAL_URL:
            # Only claim a deep link when the corpus gave an anchor to use.
            return bool(self.locator.anchor)
        return bool(self.locator.fragment())

    @property
    def citation(self) -> str:
        """A short human-readable citation, assembled from whatever is present."""
        if self.procedure_id:
            head = self.procedure_id
            if self.procedure_version:
                head += f" v{self.procedure_version}"
            if self.step_id:
                head += f" · {self.step_id}"
            return head
        parts = [self.corpus_id]
        if self.document_title:
            parts.append(self.document_title)
        elif self.document_id:
            parts.append(f"doc {self.document_id}")
        if self.chunk_id:
            parts.append(self.chunk_id)
        return " · ".join(parts)


def build_provenance(
    *,
    corpus_id: str,
    dataset_id: str | None = None,
    document_id: str | None = None,
    document_title: str | None = None,
    procedure_id: str | None = None,
    procedure_version: str | None = None,
    effective_date: str | None = None,
    chunk_id: str | None = None,
    step_id: str | None = None,
    source_url: object = None,
    source_id: object = None,
    locator: SourceLocator | None = None,
    images: tuple[ImageProvenance, ...] = (),
) -> Provenance:
    """Construct a :class:`Provenance` with the unsafe fields validated.

    The single construction point, so a URL can only enter the system through
    :func:`safe_external_url` and a local id only through :func:`safe_source_id`.
    A value that fails validation becomes ``None`` rather than raising: provenance
    arrives from a third-party dataset, so a malformed URL is expected input, and
    the honest outcome is "this entry has no usable link" rather than a failed
    ingestion.
    """
    return Provenance(
        corpus_id=corpus_id,
        dataset_id=dataset_id,
        document_id=document_id,
        document_title=document_title,
        procedure_id=procedure_id,
        procedure_version=procedure_version,
        effective_date=effective_date,
        chunk_id=chunk_id,
        step_id=step_id,
        source_url=safe_external_url(source_url),
        source_id=safe_source_id(source_id),
        locator=locator if locator is not None else SourceLocator(),
        images=images,
    )


def build_image_provenance(
    *,
    image_id: str,
    served_url: str | None = None,
    source_url: object = None,
    document_id: str | None = None,
    position_index: int | None = None,
    char_offset: int | None = None,
    caption: str | None = None,
    step_id: str | None = None,
) -> ImageProvenance:
    """Construct an :class:`ImageProvenance`, validating the upstream URL."""
    return ImageProvenance(
        image_id=image_id,
        served_url=served_url,
        source_url=safe_external_url(source_url),
        document_id=document_id,
        position_index=position_index,
        char_offset=char_offset,
        # Stripped, so a whitespace-only caption reads as absent. MRAMG-Manual
        # stores "" for all 2,599 of its images, and a corpus that stored " "
        # instead would otherwise look like it had captions.
        caption=((caption or "").strip() or None),
        step_id=step_id,
    )


__all__ = [
    "DANGEROUS_URL_SCHEMES",
    "MAX_SOURCE_ID_LENGTH",
    "SAFE_URL_SCHEMES",
    "SOURCE_ID_PATTERN",
    "ImageProvenance",
    "Provenance",
    "SourceKind",
    "SourceLocator",
    "build_image_provenance",
    "build_provenance",
    "safe_external_url",
    "safe_source_id",
]

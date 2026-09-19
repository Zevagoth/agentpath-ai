"""What any corpus must be able to hand over, and nothing about any one corpus.

The records here are deliberately close to what a benchmark actually ships, rather
than to what this system would find convenient. That is the point: a corpus is
evidence about the architecture only if the architecture bends to the corpus.

Three rules the shapes encode.

**Text is never rewritten.** :attr:`CorpusDocument.text` is the corpus's own bytes,
markers and all. Chunking records *offsets into* that text rather than copies of
it, so no transformation can silently become the thing an answer is graded
against. A corpus whose text we had normalised would be a corpus we were partly
grading ourselves.

**Order and position survive.** A multimodal document is a sequence, not a bag: an
image belongs after one paragraph and before another, and an evaluation that asks
"did you return the right image" is unanswerable if that is lost.
:attr:`CorpusImage.position_index` and :attr:`CorpusImage.char_offset` carry it.

**Ground truth is separate from everything derived.** :class:`CorpusQuestion` holds
only what the benchmark authored. Nothing in this package may write to it, and the
evaluation reads it rather than recomputing it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentpath.corpus import HEADING_SEGMENT_SEPARATOR, MAX_SECTION_CHARS, heading_label

#: The marker MRAMG uses for "an image belongs here". Declared in the base module
#: because the *concept* - a placeholder in text that an image fills - is not
#: specific to one corpus, and a second corpus using a different token should
#: normalise to the same positional model rather than to the same string.
IMAGE_PLACEHOLDER = "<PIC>"


@dataclass(frozen=True, slots=True)
class CorpusImage:
    """One image, and where it sits in its document.

    ``caption`` is optional and frequently absent: every one of MRAMG-Manual's
    2,599 captions is an empty string, which is why image ranking there cannot be
    caption-based. See ``docs/RETRIEVAL_MODES.md``.
    """

    image_id: str
    document_id: str
    #: 0-based ordinal among the document's images, i.e. which placeholder it fills.
    position_index: int
    #: Character offset of the placeholder in :attr:`CorpusDocument.text`.
    char_offset: int
    #: Repository-relative path to the bytes on disk, once fetched.
    local_path: str | None = None
    #: The upstream URL the corpus recorded, unvalidated here - the provenance
    #: layer validates before anything is rendered as a link.
    source_url: str | None = None
    caption: str | None = None
    sha256: str | None = None
    byte_size: int | None = None

    @property
    def has_caption(self) -> bool:
        return bool((self.caption or "").strip())


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One document, with its text exactly as the corpus published it."""

    document_id: str
    title: str
    #: Verbatim. Placeholders left in place.
    text: str
    #: In document order.
    images: tuple[CorpusImage, ...] = ()
    source_url: str | None = None
    #: Opaque id under which the API serves this document as its own "original".
    source_id: str | None = None

    @property
    def placeholder_count(self) -> int:
        return self.text.count(IMAGE_PLACEHOLDER)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(image.image_id for image in self.images)


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """One retrieval unit: a span of a document, named by its offsets.

    ``text`` is a slice of the document rather than a rewrite, and ``char_start``
    and ``char_end`` are what make a citation openable at the right place. The
    images listed are the ones whose placeholder falls inside this span, which is
    how "the winning passage's images" becomes a well-defined set.
    """

    chunk_id: str
    document_id: str
    text: str
    char_start: int
    char_end: int
    #: Heading path if the corpus has headings, else ``None``.
    section: str | None = None
    #: Images whose placeholder sits inside this span, in document order.
    image_ids: tuple[str, ...] = ()
    #: Position of this chunk among its document's chunks, 0-based.
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class CorpusQuestion:
    """One benchmark question and its authored ground truth.

    Read-only by contract. Stage 2B1's instruction was explicit: the questions,
    the answers and the correct images are not to be modified, so nothing in this
    package writes to these fields and the evaluation consumes them as given.
    """

    question_id: str
    question: str
    #: The authored answer, verbatim, placeholders included.
    ground_truth_answer: str
    #: Documents the benchmark says contain the answer.
    provenance_document_ids: tuple[str, ...]
    #: Images the benchmark says belong in the answer, in order.
    ground_truth_image_ids: tuple[str, ...]
    #: Difficulty or category label, when the corpus supplies one. Used for
    #: stratifying the split; ``None`` means the split falls back to a plain shuffle.
    difficulty: str | None = None

    @property
    def expected_image_count(self) -> int:
        return len(self.ground_truth_image_ids)


@dataclass(frozen=True, slots=True)
class FetchedArtefact:
    """One file the fetch step put on disk, recorded for reproducibility."""

    name: str
    path: str
    byte_size: int
    sha256: str
    #: Where it came from, for the record written beside it.
    source: str
    #: True when a previous run had already produced it unchanged.
    reused: bool = False


@dataclass
class FetchReport:
    """What a fetch actually did, so a second run can be compared to the first."""

    corpus_id: str
    artefacts: list[FetchedArtefact] = field(default_factory=list)
    #: Bytes pulled over the network on *this* run. Zero on a fully cached run,
    #: which is how idempotence is demonstrated rather than asserted.
    downloaded_bytes: int = 0
    #: Bytes that were already present and verified.
    reused_bytes: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(item.byte_size for item in self.artefacts)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class CorpusBundle:
    """A whole corpus, normalised, ready to index or evaluate against.

    Held in memory because the numbers allow it: MRAMG-Manual is 40 documents and
    1.58 million characters of text. A corpus that did not fit would need a
    streaming bundle, and the adapter protocol is where that would be added.
    """

    corpus_id: str
    #: Human-readable, for the console's corpus selector.
    title: str
    #: Where the corpus came from and under what licence. Rendered in the UI, so
    #: attribution is not left to a README nobody opens.
    attribution: str
    licence: str
    documents: tuple[CorpusDocument, ...] = ()
    chunks: tuple[CorpusChunk, ...] = ()
    questions: tuple[CorpusQuestion, ...] = ()
    #: Free-form record of how this bundle was produced.
    manifest: Mapping[str, object] = field(default_factory=dict)

    # -- lookups ------------------------------------------------------------ #

    @property
    def images(self) -> tuple[CorpusImage, ...]:
        return tuple(image for document in self.documents for image in document.images)

    def document(self, document_id: str) -> CorpusDocument | None:
        return next((item for item in self.documents if item.document_id == document_id), None)

    def image(self, image_id: str) -> CorpusImage | None:
        return next((item for item in self.images if item.image_id == image_id), None)

    def chunks_for(self, document_id: str) -> tuple[CorpusChunk, ...]:
        return tuple(item for item in self.chunks if item.document_id == document_id)

    # -- self-description --------------------------------------------------- #

    @property
    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "images": len(self.images),
            "distinct_images": len({image.image_id for image in self.images}),
            "questions": len(self.questions),
            "images_with_caption": sum(1 for image in self.images if image.has_caption),
            "images_with_source_url": sum(1 for image in self.images if image.source_url),
        }

    def integrity_problems(self) -> list[str]:
        """Structural faults that would make an evaluation meaningless.

        Checked at load rather than trusted, because every one of these would
        quietly produce a number instead of an error: a question pointing at a
        document that is not here would score zero recall and look like a
        retrieval failure, and a document whose placeholders and image list
        disagree would misplace every image after the first mismatch.
        """
        problems: list[str] = []
        document_ids = {document.document_id for document in self.documents}
        image_ids = {image.image_id for image in self.images}

        for document in self.documents:
            if document.placeholder_count != len(document.images):
                problems.append(
                    f"document {document.document_id} has {document.placeholder_count} "
                    f"placeholders but {len(document.images)} images"
                )
            for index, image in enumerate(document.images):
                if image.position_index != index:
                    problems.append(
                        f"document {document.document_id} image {image.image_id} claims position "
                        f"{image.position_index} but sits at {index}"
                    )
                    break

        for question in self.questions:
            for document_id in question.provenance_document_ids:
                if document_id not in document_ids:
                    problems.append(f"question {question.question_id} cites unknown document {document_id}")
            for image_id in question.ground_truth_image_ids:
                if image_id not in image_ids:
                    problems.append(f"question {question.question_id} cites unknown image {image_id}")

        for chunk in self.chunks:
            document = self.document(chunk.document_id)
            if document is None:
                problems.append(f"chunk {chunk.chunk_id} belongs to unknown document {chunk.document_id}")
                continue
            if document.text[chunk.char_start : chunk.char_end] != chunk.text:
                problems.append(
                    f"chunk {chunk.chunk_id} text does not match its document at [{chunk.char_start}:{chunk.char_end}]"
                )
        return problems


@runtime_checkable
class CorpusAdapter(Protocol):
    """How a corpus gets in. Two steps, deliberately separate.

    ``fetch`` touches the network and the filesystem and nothing else. ``load``
    touches the filesystem and produces a :class:`CorpusBundle`. Keeping them apart
    is what lets the normalisation be tested against committed fixtures with no
    network, and what lets a re-index skip the download entirely.
    """

    @property
    def corpus_id(self) -> str:
        """Stable machine id, e.g. ``mramg-manual``. Also the isolation key."""
        ...

    @property
    def title(self) -> str: ...

    def fetch(self, *, force: bool = False) -> FetchReport:
        """Put the raw artefacts on disk. Idempotent and resumable."""
        ...

    def load(self) -> CorpusBundle:
        """Normalise what is on disk. No network."""
        ...


def slice_chunks(
    document: CorpusDocument,
    *,
    target_chars: int,
    overlap_chars: int,
    heading_pattern: str | None = None,
) -> tuple[CorpusChunk, ...]:
    """Cut a document into overlapping spans, preferring paragraph boundaries.

    Offsets, not copies: every chunk records where it came from, so
    :meth:`CorpusBundle.integrity_problems` can prove the slice still matches the
    document and a citation can open at the right character.

    Boundaries are chosen at the last blank line inside the budget, falling back to
    the last single newline and then to a hard cut. Cutting mid-sentence is worse
    than a slightly short chunk: the embedder sees a fragment, and a reader sent to
    that offset lands mid-thought. ``overlap_chars`` then carries the tail of one
    chunk into the head of the next so a fact sitting on a boundary is retrievable
    from both sides.

    An image is assigned to the chunk whose span contains its placeholder, using
    the *primary* span rather than the overlap, so an image is not counted twice.
    """
    text = document.text
    if not text:
        return ()

    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_chars, length)
        if end < length:
            window_floor = start + max(target_chars // 2, 1)
            boundary = text.rfind("\n\n", window_floor, end)
            if boundary == -1:
                boundary = text.rfind("\n", window_floor, end)
            if boundary != -1:
                end = boundary
        spans.append((start, end))
        if end >= length:
            break
        start = max(end - overlap_chars, end) if overlap_chars <= 0 else max(end - overlap_chars, start + 1)
        # The overlap must not make the next chunk start before this one did.
        if spans and start <= spans[-1][0]:
            start = spans[-1][1]

    # Scanned once for the whole document rather than once per chunk: the largest
    # MRAMG manual is 168k characters and 164 chunks, and re-scanning it for each
    # would be 164 passes to answer the same question.
    headings = _heading_positions(text, heading_pattern) if heading_pattern else ()

    chunks: list[CorpusChunk] = []
    for ordinal, (span_start, span_end) in enumerate(spans):
        # Primary span excludes the overlap that was carried in from the previous
        # chunk, so each image lands in exactly one chunk.
        primary_start = spans[ordinal - 1][1] if ordinal > 0 else span_start
        owned = tuple(image.image_id for image in document.images if primary_start <= image.char_offset < span_end)
        chunks.append(
            CorpusChunk(
                chunk_id=f"{document.document_id}#c{ordinal:03d}",
                document_id=document.document_id,
                text=text[span_start:span_end],
                char_start=span_start,
                char_end=span_end,
                section=_governing_heading(headings, span_start, span_end),
                image_ids=owned,
                ordinal=ordinal,
            )
        )
    return tuple(chunks)


def _heading_positions(text: str, pattern: str) -> tuple[tuple[int, str], ...]:
    """Every heading in *text* as ``(offset, label)``, in document order.

    The captured text is trimmed to the heading itself by
    :func:`agentpath.corpus.heading_label`, because what is wanted is a label a reader
    can glance at beside a citation, not the block the heading introduces. That function
    lives on the reading side because the hybrid retrieval path needs the identical rule
    and cannot import this package.
    """
    import re

    positions: list[tuple[int, str]] = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        label = heading_label(match.group(0))
        if label:
            positions.append((match.start(), label))
    return tuple(positions)


def _governing_heading(headings: tuple[tuple[int, str], ...], span_start: int, span_end: int) -> str | None:
    """The heading a reader of this span would be under.

    The last heading starting at or before the span, which is the one still in
    force. A chunk that *opens* with a heading counts as being under it - an
    earlier version searched only the text strictly before the span's start, so the
    first chunk of every document came back with no section at all even when its
    first line was the document's title.

    When no heading precedes the span, the first heading *inside* it is used, since
    that is the only label the span offers.
    """
    best: str | None = None
    for offset, heading in headings:
        if offset <= span_start:
            best = heading
            continue
        if best is None and offset < span_end:
            return heading
        break
    return best


def locate_placeholders(text: str, placeholder: str = IMAGE_PLACEHOLDER) -> tuple[int, ...]:
    """Character offsets of every placeholder, in order."""
    offsets: list[int] = []
    cursor = text.find(placeholder)
    while cursor != -1:
        offsets.append(cursor)
        cursor = text.find(placeholder, cursor + len(placeholder))
    return tuple(offsets)


def context_window(
    text: str,
    offset: int,
    *,
    before: int,
    after: int,
) -> str:
    """The text around *offset*, which is all a caption-less image has to offer.

    Used to build the searchable text for an image. Trimmed to whitespace on both
    ends so a window does not open or close mid-word.
    """
    start = max(0, offset - before)
    end = min(len(text), offset + after)
    window = text[start:end]
    if start > 0:
        space = window.find(" ")
        if 0 <= space < 40:
            window = window[space + 1 :]
    if end < len(text):
        space = window.rfind(" ")
        if space > len(window) - 40:
            window = window[:space]
    return " ".join(window.split())


def iter_jsonl(path: Path) -> Iterable[object]:
    """Read a JSON-lines file, skipping blank lines."""
    import json

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a JSON-lines file, creating parents."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    # Re-exported from ``agentpath.corpus``, where the single definition lives: both the
    # writing side (here) and the hybrid retrieval path need the same heading rule, and
    # the reader cannot import this package.
    "HEADING_SEGMENT_SEPARATOR",
    "IMAGE_PLACEHOLDER",
    "MAX_SECTION_CHARS",
    "CorpusAdapter",
    "CorpusBundle",
    "CorpusChunk",
    "CorpusDocument",
    "CorpusImage",
    "CorpusQuestion",
    "FetchReport",
    "FetchedArtefact",
    "context_window",
    "heading_label",
    "iter_jsonl",
    "locate_placeholders",
    "slice_chunks",
    "write_jsonl",
]

"""External corpora, as the API sees them: documents to open, images to serve.

The read-only counterpart to ``agentpath_ingestion.corpora``. It exists as a
separate, narrower module for the reason :mod:`agentpath.procedures` and
:mod:`agentpath.screenshots` do: ``tests/test_visual_steps.py`` asserts that
``agentpath.api.app`` pulls in no ``agentpath_ingestion`` module, and that import
boundary is what guarantees no corpus normalisation, checksum walk or network call
can happen while a request is in flight. The two sides meet at files written by
``make corpus-export``.

Two jobs, and the safety model is the same for both.

**Serving a document as its own "original".** MRAMG records no per-document URL, so
the only original this system can offer is the local copy. It is addressed by an
opaque id - ``mramg-manual-50000`` - which is a key in a registry built at startup.

**Serving an image.** 2,599 files, 147 MiB. Too much to hold in memory, so unlike
:mod:`agentpath.screenshots` this one keeps paths rather than bytes. That
reintroduces the risk that module avoids by construction, so it is closed off
explicitly:

1. every path is resolved once, at startup, and checked to be inside the corpus's
   own images directory - a symlink or a ``..`` in the exported manifest is rejected
   then, not at request time;
2. a request supplies only an image id, which must match
   :data:`~agentpath.provenance.SOURCE_ID_PATTERN` and must already be a key in the
   registry. There is no string concatenation, no ``os.path.join`` and no ``open``
   on a request-derived value anywhere on the request path;
3. an unknown id and a malformed id get the byte-identical 404, so the route is not
   an oracle for what exists on disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from agentpath.provenance import (
    ImageProvenance,
    Provenance,
    SourceLocator,
    build_image_provenance,
    build_provenance,
    safe_source_id,
)

logger = structlog.get_logger(__name__)

#: Bundle shape this reader understands. Written by
#: ``agentpath_ingestion.corpora.export``; a mismatch is refused rather than
#: guessed at, because a missing field would become a silently absent citation.
SUPPORTED_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
CHUNKS_NAME = "chunks.jsonl"
QUESTIONS_NAME = "questions.jsonl"

#: Refuse an implausibly large export rather than reading it into memory. The
#: Manual bundle's documents file is 1.7 MiB.
MAX_BUNDLE_BYTES = 256 * 1024 * 1024

#: Extensions an image may have. Anything else in the images directory is ignored
#: rather than served.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

MEDIA_TYPES: Mapping[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True, slots=True)
class CorpusImageRecord:
    """One image the API can serve, with everything provenance needs."""

    image_id: str
    corpus_id: str
    document_id: str
    position_index: int
    char_offset: int
    #: Resolved, validated at load. Never built from request input.
    path: Path
    byte_size: int
    media_type: str
    source_url: str | None = None
    caption: str | None = None

    @property
    def served_url(self) -> str:
        return f"/api/corpus-images/{self.corpus_id}/{self.image_id}"

    def provenance(self) -> ImageProvenance:
        return build_image_provenance(
            image_id=self.image_id,
            served_url=self.served_url,
            source_url=self.source_url,
            document_id=self.document_id,
            position_index=self.position_index,
            char_offset=self.char_offset,
            caption=self.caption,
        )


@dataclass(frozen=True, slots=True)
class CorpusDocumentRecord:
    """One document, held whole so a citation can be opened at its offset."""

    document_id: str
    corpus_id: str
    title: str
    text: str
    source_id: str
    source_url: str | None = None
    image_ids: tuple[str, ...] = ()

    def excerpt(self, char_start: int, char_end: int, *, padding: int = 0) -> str:
        """The text between two offsets, clamped to the document."""
        start = max(0, min(char_start - padding, len(self.text)))
        end = max(start, min(char_end + padding, len(self.text)))
        return self.text[start:end]


@dataclass(frozen=True, slots=True)
class CorpusChunkRecord:
    """One retrieval unit, as offsets into its document."""

    chunk_id: str
    document_id: str
    corpus_id: str
    text: str
    char_start: int
    char_end: int
    section: str | None = None
    image_ids: tuple[str, ...] = ()
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class CorpusQuestionRecord:
    """A benchmark question and its ground truth, read-only."""

    question_id: str
    question: str
    ground_truth_answer: str
    provenance_document_ids: tuple[str, ...]
    ground_truth_image_ids: tuple[str, ...]
    difficulty: str | None = None


@dataclass(frozen=True, slots=True)
class Corpus:
    """One external corpus, loaded and ready to serve."""

    corpus_id: str
    title: str
    licence: str
    attribution: str
    documents: Mapping[str, CorpusDocumentRecord] = field(default_factory=dict)
    #: Keyed by the opaque source id, which is what a link carries.
    by_source_id: Mapping[str, CorpusDocumentRecord] = field(default_factory=dict)
    images: Mapping[str, CorpusImageRecord] = field(default_factory=dict)
    chunks: Mapping[str, CorpusChunkRecord] = field(default_factory=dict)
    questions: tuple[CorpusQuestionRecord, ...] = ()
    build: Mapping[str, object] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "images": len(self.images),
            "questions": len(self.questions),
        }

    @property
    def image_retrieval_mode(self) -> str:
        """How images are ranked in this corpus, named honestly.

        Read from the build record rather than assumed, so a corpus that did ship
        captions would report something different.
        """
        value = self.build.get("image_retrieval_mode")
        return str(value) if isinstance(value, str) else "unknown"

    def document(self, document_id: object) -> CorpusDocumentRecord | None:
        return self.documents.get(document_id) if isinstance(document_id, str) else None

    def chunk(self, chunk_id: object) -> CorpusChunkRecord | None:
        return self.chunks.get(chunk_id) if isinstance(chunk_id, str) else None

    def source(self, source_id: object) -> CorpusDocumentRecord | None:
        """A document by its opaque link id. Total: bad input is an expected input."""
        checked = safe_source_id(source_id)
        return self.by_source_id.get(checked) if checked else None

    def image(self, image_id: object) -> CorpusImageRecord | None:
        """An image by id, validated for shape then looked up in the closed set."""
        checked = safe_source_id(image_id)
        return self.images.get(checked) if checked else None

    def images_for(self, document_id: str) -> tuple[CorpusImageRecord, ...]:
        document = self.document(document_id)
        if document is None:
            return ()
        return tuple(record for image_id in document.image_ids if (record := self.images.get(image_id)))

    # -- provenance --------------------------------------------------------- #

    def provenance_for_chunk(self, chunk_id: str, *, dataset_id: str | None = None) -> Provenance | None:
        """The full trail for one retrieved chunk."""
        chunk = self.chunk(chunk_id)
        if chunk is None:
            return None
        document = self.document(chunk.document_id)
        if document is None:
            return None
        return build_provenance(
            corpus_id=self.corpus_id,
            dataset_id=dataset_id,
            document_id=document.document_id,
            document_title=document.title,
            chunk_id=chunk.chunk_id,
            source_url=document.source_url,
            source_id=document.source_id,
            locator=SourceLocator(
                section=chunk.section,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            ),
            images=tuple(record.provenance() for image_id in chunk.image_ids if (record := self.images.get(image_id))),
        )

    def provenance_for_document(self, document_id: str, *, dataset_id: str | None = None) -> Provenance | None:
        document = self.document(document_id)
        if document is None:
            return None
        return build_provenance(
            corpus_id=self.corpus_id,
            dataset_id=dataset_id,
            document_id=document.document_id,
            document_title=document.title,
            source_url=document.source_url,
            source_id=document.source_id,
        )


@dataclass(frozen=True, slots=True)
class CorpusCatalogue:
    """Every external corpus the API can serve, keyed by id.

    Immutable after construction. ``problems`` records what was rejected, so a
    corpus missing from the catalogue is reported rather than looking like a corpus
    nobody exported.
    """

    corpora: Mapping[str, Corpus] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> CorpusCatalogue:
        return cls(corpora={})

    @classmethod
    def load(cls, root: Path | None = None) -> CorpusCatalogue:
        """Read every exported bundle under *root*.

        A bundle that cannot be trusted is skipped and recorded rather than raised:
        one broken corpus should degrade to "that corpus cannot be searched", not to
        "the API will not start".
        """
        base = root if root is not None else default_external_root()
        loaded: dict[str, Corpus] = {}
        problems: list[str] = []

        if not base.is_dir():
            return cls(corpora={}, problems=(f"external corpus directory {base.name!r} does not exist",))

        for directory in sorted(base.iterdir()):
            bundle = directory / "build"
            if not (bundle / MANIFEST_NAME).is_file():
                continue
            corpus, failures = _load_one(bundle)
            if corpus is not None:
                if corpus.corpus_id in loaded:
                    problems.append(f"{directory.name} declares corpus id {corpus.corpus_id}, already loaded")
                else:
                    loaded[corpus.corpus_id] = corpus
            problems.extend(failures)

        logger.info(
            "corpus.catalogue_loaded",
            corpora=sorted(loaded),
            documents=sum(len(item.documents) for item in loaded.values()),
            images=sum(len(item.images) for item in loaded.values()),
            problems=len(problems),
        )
        for problem in problems:
            logger.warning("corpus.unusable", detail=problem)
        return cls(corpora=dict(sorted(loaded.items())), problems=tuple(problems))

    # -- lookup ------------------------------------------------------------- #

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.corpora)

    def corpus(self, corpus_id: object) -> Corpus | None:
        if not isinstance(corpus_id, str):
            return None
        return self.corpora.get(corpus_id)

    def source(self, source_id: object) -> tuple[Corpus, CorpusDocumentRecord] | None:
        """Find a document by its opaque link id, across every corpus.

        The id carries its corpus (``mramg-manual-50000``), but the lookup does not
        rely on parsing that: it asks each corpus, so an id shape change cannot turn
        into a wrong-corpus hit.
        """
        checked = safe_source_id(source_id)
        if checked is None:
            return None
        for corpus in self.corpora.values():
            document = corpus.source(checked)
            if document is not None:
                return corpus, document
        return None

    def image(self, corpus_id: object, image_id: object) -> tuple[Corpus, CorpusImageRecord] | None:
        corpus = self.corpus(corpus_id)
        if corpus is None:
            return None
        record = corpus.image(image_id)
        return (corpus, record) if record is not None else None

    def __len__(self) -> int:
        return len(self.corpora)

    def __iter__(self) -> Iterator[Corpus]:
        return iter(self.corpora.values())

    def __contains__(self, corpus_id: object) -> bool:
        return self.corpus(corpus_id) is not None


# --------------------------------------------------------------------------- #
# Loading one bundle
# --------------------------------------------------------------------------- #


def _load_one(bundle: Path) -> tuple[Corpus | None, list[str]]:
    name = bundle.parent.name
    try:
        manifest = json.loads((bundle / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"corpus {name}: manifest unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return None, [f"corpus {name}: manifest is not an object"]

    version = manifest.get("format_version")
    if version != SUPPORTED_FORMAT_VERSION:
        return None, [
            f"corpus {name}: bundle format {version!r} is not the supported "
            f"{SUPPORTED_FORMAT_VERSION}; re-run make corpus-export"
        ]

    corpus_id = manifest.get("corpus_id")
    if not isinstance(corpus_id, str) or safe_source_id(corpus_id) is None:
        return None, [f"corpus {name}: unusable corpus_id {corpus_id!r}"]

    total = 0
    for filename in (DOCUMENTS_NAME, CHUNKS_NAME, QUESTIONS_NAME):
        path = bundle / filename
        if not path.is_file():
            return None, [f"corpus {name}: {filename} is missing; re-run make corpus-export"]
        total += path.stat().st_size
    if total > MAX_BUNDLE_BYTES:
        return None, [f"corpus {name}: bundle is {total} bytes, above the {MAX_BUNDLE_BYTES} limit"]

    problems: list[str] = []
    images_root = _resolve_images_root(bundle, manifest, corpus_id, problems)

    documents: dict[str, CorpusDocumentRecord] = {}
    by_source: dict[str, CorpusDocumentRecord] = {}
    images: dict[str, CorpusImageRecord] = {}

    for row in _read_jsonl(bundle / DOCUMENTS_NAME):
        document_id = str(row.get("document_id") or "")
        source_id = safe_source_id(row.get("source_id"))
        if not document_id or source_id is None:
            problems.append(f"corpus {corpus_id}: document {document_id!r} has an unusable source id")
            continue

        image_ids: list[str] = []
        for entry in row.get("images") or []:
            record = _image_record(entry, corpus_id, document_id, images_root, problems)
            if record is None:
                continue
            image_ids.append(record.image_id)
            # A repeated id is the corpus reusing one image; keep the first record.
            images.setdefault(record.image_id, record)

        document = CorpusDocumentRecord(
            document_id=document_id,
            corpus_id=corpus_id,
            title=str(row.get("title") or document_id),
            text=str(row.get("text") or ""),
            source_id=source_id,
            source_url=row.get("source_url") if isinstance(row.get("source_url"), str) else None,
            image_ids=tuple(image_ids),
        )
        documents[document_id] = document
        by_source[source_id] = document

    chunks: dict[str, CorpusChunkRecord] = {}
    for row in _read_jsonl(bundle / CHUNKS_NAME):
        chunk_id = str(row.get("chunk_id") or "")
        document_id = str(row.get("document_id") or "")
        if not chunk_id or document_id not in documents:
            problems.append(f"corpus {corpus_id}: chunk {chunk_id!r} has no known document")
            continue
        chunks[chunk_id] = CorpusChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            corpus_id=corpus_id,
            text=str(row.get("text") or ""),
            char_start=int(row.get("char_start") or 0),
            char_end=int(row.get("char_end") or 0),
            section=row.get("section") if isinstance(row.get("section"), str) else None,
            image_ids=tuple(str(item) for item in row.get("image_ids") or ()),
            ordinal=int(row.get("ordinal") or 0),
        )

    questions = tuple(
        CorpusQuestionRecord(
            question_id=str(row.get("question_id") or ""),
            question=str(row.get("question") or ""),
            ground_truth_answer=str(row.get("ground_truth_answer") or ""),
            provenance_document_ids=tuple(str(item) for item in row.get("provenance_document_ids") or ()),
            ground_truth_image_ids=tuple(str(item) for item in row.get("ground_truth_image_ids") or ()),
            difficulty=row.get("difficulty") if isinstance(row.get("difficulty"), str) else None,
        )
        for row in _read_jsonl(bundle / QUESTIONS_NAME)
    )

    if not documents:
        return None, [*problems, f"corpus {corpus_id}: no usable documents"]

    corpus = Corpus(
        corpus_id=corpus_id,
        title=str(manifest.get("title") or corpus_id),
        licence=str(manifest.get("licence") or "unknown"),
        attribution=str(manifest.get("attribution") or ""),
        documents=documents,
        by_source_id=by_source,
        images=images,
        chunks=chunks,
        questions=questions,
        build=manifest.get("build") if isinstance(manifest.get("build"), dict) else {},
    )
    return corpus, problems


def _resolve_images_root(
    bundle: Path, manifest: Mapping[str, object], corpus_id: str, problems: list[str]
) -> Path | None:
    """The one directory images may be served from, resolved and fixed at load.

    Everything a request can reach must live under this. Resolving it here - once,
    at startup - is what lets the request path be a dictionary lookup with no path
    handling in it at all.
    """
    raw = manifest.get("images_root")
    candidate = Path(str(raw)) if isinstance(raw, str) and raw else bundle.parent / "raw" / "images"
    if not candidate.is_absolute():
        candidate = (bundle.parent.parents[1] / candidate).resolve()
    resolved = candidate.resolve()
    if not resolved.is_dir():
        problems.append(f"corpus {corpus_id}: images directory {resolved.name!r} does not exist")
        return None
    return resolved


def _image_record(
    entry: object,
    corpus_id: str,
    document_id: str,
    images_root: Path | None,
    problems: list[str],
) -> CorpusImageRecord | None:
    if not isinstance(entry, dict) or images_root is None:
        return None
    image_id = safe_source_id(entry.get("image_id"))
    if image_id is None:
        problems.append(f"corpus {corpus_id}: an image id in document {document_id} is unusable")
        return None

    # Built from the *validated id* and the fixed root, never from the manifest's
    # own path string: a manifest carrying "../../etc/passwd" cannot escape here
    # because that string is not used to locate anything.
    path: Path | None = None
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = (images_root / f"{image_id}{suffix}").resolve()
        # Belt to the braces: the id pattern already forbids separators.
        if not candidate.is_relative_to(images_root):
            problems.append(f"corpus {corpus_id}: image {image_id} resolved outside its corpus")
            return None
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        # Not a problem worth refusing the corpus for: the evaluation reports a
        # broken image rate, and an absent file is exactly what that measures.
        return None

    return CorpusImageRecord(
        image_id=image_id,
        corpus_id=corpus_id,
        document_id=document_id,
        position_index=int(entry.get("position_index") or 0),
        char_offset=int(entry.get("char_offset") or 0),
        path=path,
        byte_size=path.stat().st_size,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        source_url=entry.get("source_url") if isinstance(entry.get("source_url"), str) else None,
        caption=entry.get("caption") if isinstance(entry.get("caption"), str) else None,
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def default_external_root() -> Path:
    """``data/external`` in this repository."""
    from agentpath.paths import repo_root

    return repo_root() / "data" / "external"


def render_document_html(
    document: CorpusDocumentRecord,
    corpus: Corpus,
    *,
    char_start: int | None = None,
    char_end: int | None = None,
) -> str:
    """The document as a standalone page that scrolls to the cited span.

    Built here rather than in the route so it can be tested without HTTP, and
    assembled with explicit escaping: the text is third-party content, and the whole
    point of this page is that a reader can trust what they are shown. Every
    interpolated value goes through :func:`html.escape`, the ``<PIC>`` markers are
    replaced by the images they stand for, and there is no script anywhere on the
    page.
    """
    import html

    title = html.escape(document.title)
    corpus_title = html.escape(corpus.title)
    attribution = html.escape(corpus.attribution)

    images = {record.image_id: record for record in corpus.images_for(document.document_id)}
    ordered = [images.get(image_id) for image_id in document.image_ids]

    pieces: list[str] = []
    cursor = 0
    marker = "<PIC>"
    index = 0
    text = document.text
    while True:
        found = text.find(marker, cursor)
        if found == -1:
            pieces.append(_escaped_span(text[cursor:], cursor, char_start, char_end))
            break
        pieces.append(_escaped_span(text[cursor:found], cursor, char_start, char_end))
        record = ordered[index] if index < len(ordered) else None
        if record is not None:
            alt = html.escape(record.caption or f"Figure {index + 1} of {title}")
            pieces.append(
                f'<figure id="image-{html.escape(record.image_id)}">'
                f'<img src="{html.escape(record.served_url)}" alt="{alt}" loading="lazy">'
                f"<figcaption>{html.escape(record.image_id)} "
                f"&middot; image {index + 1} of {len(ordered)} in this document</figcaption>"
                f"</figure>"
            )
        else:
            pieces.append('<p class="missing">[image not available]</p>')
        index += 1
        cursor = found + len(marker)

    highlighted = "".join(pieces)
    anchor = f"char={char_start}" if char_start is not None else ""
    position = (
        f"Opened at character {char_start}." if char_start is not None else "Opened at the start of the document."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{title} - {corpus_title}</title>
<style>
 body {{ margin:0; background:#0f1720; color:#e8eef4;
        font:15px/1.65 ui-sans-serif, system-ui, "Noto Sans Arabic", sans-serif; }}
 header {{ position:sticky; top:0; background:#16212c; border-bottom:1px solid #2a3b4a; padding:14px 22px; }}
 h1 {{ margin:0; font-size:17px; }}
 .meta {{ color:#9bb0c1; font-size:12.5px; margin-top:4px; }}
 main {{ max-width:900px; margin:0 auto; padding:24px 22px 80px; }}
 pre {{ white-space:pre-wrap; word-wrap:break-word; font:inherit; margin:0; }}
 mark {{ background:rgba(45,212,191,.22); color:#e8eef4; outline:1px solid rgba(45,212,191,.55); }}
 figure {{ margin:18px 0; padding:10px; border:1px solid #2a3b4a; border-radius:8px; background:#16212c; }}
 figure img {{ display:block; max-width:100%; height:auto; border-radius:4px; }}
 figcaption {{ margin-top:7px; font-size:11.5px; color:#6f8798; font-family:ui-monospace,monospace; }}
 .missing {{ color:#f87171; font-size:12.5px; }}
 footer {{ border-top:1px solid #2a3b4a; color:#6f8798; font-size:12px; padding:14px 22px; }}
</style>
</head>
<body>
<header>
 <h1>{title}</h1>
 <p class="meta">{corpus_title} &middot; document {html.escape(document.document_id)} &middot; {position}</p>
</header>
<main id="{html.escape(anchor)}"><pre>{highlighted}</pre></main>
<footer>{attribution}</footer>
</body>
</html>
"""


def _escaped_span(piece: str, offset: int, char_start: int | None, char_end: int | None) -> str:
    """Escape *piece*, wrapping the part of it inside the cited span in ``<mark>``."""
    import html

    if char_start is None or char_end is None or char_end <= char_start:
        return html.escape(piece)

    end = offset + len(piece)
    if end <= char_start or offset >= char_end:
        return html.escape(piece)

    local_start = max(0, char_start - offset)
    local_end = min(len(piece), char_end - offset)
    return (
        html.escape(piece[:local_start])
        + "<mark>"
        + html.escape(piece[local_start:local_end])
        + "</mark>"
        + html.escape(piece[local_end:])
    )


__all__ = [
    "IMAGE_SUFFIXES",
    "MAX_BUNDLE_BYTES",
    "SUPPORTED_FORMAT_VERSION",
    "Corpus",
    "CorpusCatalogue",
    "CorpusChunkRecord",
    "CorpusDocumentRecord",
    "CorpusImageRecord",
    "CorpusQuestionRecord",
    "default_external_root",
    "render_document_html",
]

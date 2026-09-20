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
from dataclasses import dataclass, field, replace
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

#: Optional. Written by ``agentpath_ingestion.corpora.roles``; absent for a corpus nobody
#: has classified, in which case every image keeps the ``other`` default.
ROLES_NAME = "roles.jsonl"

#: Roles the reader understands. Anything else in the file is recorded as a problem and
#: the image keeps its default, rather than an unknown string reaching the UI.
KNOWN_IMAGE_ROLES = frozenset({"workflow_diagram", "ui_screenshot", "other"})

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

#: The Stage 2A knowledge base, named so the console's corpus selector can offer it
#: beside the external ones.
#:
#: Deliberately *not* a :class:`Corpus` in this catalogue. It is versioned procedures
#: with product lines, effective dates and step dependencies, served by ``/api/search``
#: and ``/api/answer``; the external corpora are documents with images, served by
#: ``/api/corpus-search``. Forcing both into one type would mean a type where half the
#: fields are null on either side, and the two endpoints would still be two endpoints.
PROCEDURES_CORPUS_ID = "synthetic-procedures"

#: Every RAGFlow dataset this project owns is prefixed, so a corpus id cannot claim a
#: dataset it did not create and every dataset is identifiable in the RAGFlow UI.
#:
#: Defined here, on the API side, rather than in the ingestion package that writes
#: the datasets, because both sides need it and this is the direction the dependency
#: already runs: ``agentpath_ingestion`` imports ``agentpath`` and not the reverse.
#: Two copies of this rule would mean the reader could look for a dataset the writer
#: never created, and the failure would surface as "corpus not indexed".
DATASET_PREFIX = "agentpath-"

#: Used only when a bundle's build record does not name its own window sizes. The
#: real values come from the bundle - see :attr:`Corpus.image_context_sizes` - so the
#: API searches the same window the adapter described. These mirror
#: ``agentpath_ingestion.corpora.mramg.IMAGE_CONTEXT_BEFORE`` / ``_AFTER``: more text
#: before than after, because a manual names a thing and then shows it far more often
#: than the reverse.
DEFAULT_IMAGE_CONTEXT_BEFORE = 600
DEFAULT_IMAGE_CONTEXT_AFTER = 300

#: A run of two or more spaces is where a line break used to be in text flattened out
#: of a PDF, and MRAMG-Manual is entirely such text: 17 to 116 real newlines per
#: document, with paragraphs separated by double spaces instead.
HEADING_SEGMENT_SEPARATOR = "  "

#: A section label longer than this is not a label. See :func:`heading_label`.
MAX_SECTION_CHARS = 120


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
    #: What kind of figure this is, from ``roles.jsonl`` when the corpus has been
    #: classified. Defaults to ``other`` so a corpus with no role file behaves exactly as
    #: before: every image is shown in document order, which is the right default for
    #: everything except a decision diagram.
    role: str = "other"
    #: How the role was decided - ``corpus_metadata``, ``vision_model`` or ``prefilter``.
    #: ``prefilter`` means *nothing looked at this image*, which is not the same claim as
    #: "a model checked and it is not a diagram". Surfaced so the UI cannot overstate it.
    role_method: str = "prefilter"
    role_confidence: float = 0.0
    role_reason: str = ""

    @property
    def served_url(self) -> str:
        return f"/api/corpus-images/{self.corpus_id}/{self.image_id}"

    @property
    def is_workflow_diagram(self) -> bool:
        return self.role == "workflow_diagram"

    @property
    def role_examined(self) -> bool:
        """Whether a model or the corpus itself decided this, rather than nothing."""
        return self.role_method != "prefilter"

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
        """This document's figures, each once, in the order a reader meets them.

        Sorted on ``position_index`` and deduplicated rather than returned in the
        document's raw slot order, because five MRAMG-Manual documents place the *same*
        image id at two different ``<PIC>`` slots - 2,607 slots against 2,599 distinct
        ids. The registry holds one record per id, carrying its first position, so walking
        the raw slot list yields that record a second time with a position that has already
        been passed and the sequence runs backwards. Measured: 4 of 40 documents.

        A figure reused twice is therefore shown once, at its first position. The
        alternative - one entry per slot - would need a per-slot offset the bundle does not
        export, and would show a reader the same picture twice without being able to say
        the second occurrence sits somewhere else.
        """
        document = self.document(document_id)
        if document is None:
            return ()
        seen: set[str] = set()
        records: list[CorpusImageRecord] = []
        for image_id in document.image_ids:
            if image_id in seen:
                continue
            record = self.images.get(image_id)
            if record is None:
                continue
            seen.add(image_id)
            records.append(record)
        return tuple(sorted(records, key=lambda item: (item.position_index, item.image_id)))

    # -- what a caption-less image offers ----------------------------------- #

    @property
    def image_context_sizes(self) -> tuple[int, int]:
        """Characters of document text before and after an image's marker.

        Read from the build record rather than hardcoded, so the window the API
        searches is the window the adapter documented when it named this corpus's
        image retrieval mode. The fallbacks are the adapter's own defaults, used only
        for a bundle built before these keys existed.
        """
        before = self.build.get("image_context_before")
        after = self.build.get("image_context_after")
        return (
            int(before) if isinstance(before, int) else DEFAULT_IMAGE_CONTEXT_BEFORE,
            int(after) if isinstance(after, int) else DEFAULT_IMAGE_CONTEXT_AFTER,
        )

    def image_context(self, image_id: str) -> str:
        """The document text around *image_id*'s position.

        For MRAMG-Manual this is the *entire* searchable content of an image: all
        2,599 captions are empty strings, so there is nothing else to read. See
        ``docs/RETRIEVAL_MODES.md``.

        Computed here from the verbatim document rather than exported alongside the
        images. Two reasons: it is derived data, and a derived copy on disk is a copy
        that can go stale against the text it was derived from; and 2,599 windows of
        ~900 characters would add 2.3 MB to a 6.7 MB bundle to store something a
        dictionary lookup and a slice can produce.

        A caption is preferred when one exists, because a caption is what a human
        wrote about the image and a window is only what happened to sit near it.
        """
        record = self.image(image_id)
        if record is None:
            return ""
        if record.caption and record.caption.strip():
            return record.caption.strip()
        document = self.document(record.document_id)
        if document is None:
            return ""
        before, after = self.image_context_sizes
        return _context_window(document.text, record.char_offset, before=before, after=after)

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

    # Roles are optional: a bundle exported before classification, or a corpus nobody has
    # classified, simply leaves every image at the "other" default. Absence is not a
    # problem to report, because the display path does not depend on it.
    roles_path = bundle / ROLES_NAME
    if roles_path.is_file():
        images = _apply_roles(roles_path, images, corpus_id, problems)

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


def _apply_roles(
    path: Path,
    images: dict[str, CorpusImageRecord],
    corpus_id: str,
    problems: list[str],
) -> dict[str, CorpusImageRecord]:
    """Overlay ``roles.jsonl`` onto the image registry.

    An overlay rather than a source: the registry is built from the documents, so a role
    file naming an image the corpus does not have is a stale file, not a new image. That
    is reported and ignored - inventing an image from a role row would create one the
    route cannot serve.
    """
    applied = 0
    for row in _read_jsonl(path):
        image_id = str(row.get("image_id") or "")
        record = images.get(image_id)
        if record is None:
            problems.append(f"corpus {corpus_id}: roles name image {image_id!r}, which the documents do not")
            continue
        role = str(row.get("role") or "other")
        if role not in KNOWN_IMAGE_ROLES:
            problems.append(f"corpus {corpus_id}: image {image_id!r} has unknown role {role!r}; left as other")
            role = "other"
        confidence = row.get("confidence")
        images[image_id] = replace(
            record,
            role=role,
            role_method=str(row.get("method") or "prefilter"),
            role_confidence=float(confidence) if isinstance(confidence, int | float) else 0.0,
            role_reason=str(row.get("reason") or ""),
        )
        applied += 1
    logger.info("corpus.roles_applied", corpus=corpus_id, images=applied)
    return images


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


def heading_label(raw: str) -> str:
    """Turn a matched markdown heading into a label short enough to be one.

    Three things happen, in order: the ``#`` markers go, because a reader shown
    "# Components" is being shown the corpus's markup rather than its section name;
    the text is cut at the first double space, which is where a line break used to be
    in flattened text; and what is left is capped.

    The bug this exists for, measured: ``^#{1,6} .+$`` matches to the end of a *line*,
    and in these documents a line holds a heading and the entire section it introduces.
    Over the 1,506 exported chunks the median "heading" was 943 characters and the
    longest 5,124. The same fault then appeared independently on the hybrid retrieval
    path, which derives a label from RAGFlow's chunk text - which is why this is one
    function rather than two copies of a rule.

    It is a no-op on well-formed markdown, where a heading contains no double space and
    is far shorter than the cap.
    """
    heading = raw.strip().lstrip("#").strip()
    head, separator, _rest = heading.partition(HEADING_SEGMENT_SEPARATOR)
    if separator:
        heading = head.strip()
    return heading[:MAX_SECTION_CHARS].strip()


def dataset_name_for(corpus_id: str) -> str:
    """The RAGFlow dataset that holds *corpus_id*, and only *corpus_id*.

    The isolation key. One corpus, one dataset, and a retrieval call names exactly one
    dataset - which is why mixed results are unreachable rather than guarded against.
    """
    return f"{DATASET_PREFIX}{corpus_id}"


def default_external_root() -> Path:
    """``data/external`` in this repository."""
    from agentpath.paths import repo_root

    return repo_root() / "data" / "external"


def _context_window(text: str, offset: int, *, before: int, after: int) -> str:
    """The text around *offset*, trimmed so it does not start or end mid-word.

    Whitespace is collapsed because the result is scored as a bag of words and shown
    as a one-line explanation; neither use wants the document's line breaks. The
    40-character guards stop the trim from eating a whole word when the nearest space
    is far away - which happens at the edges of tables and long identifiers.

    The reference definition of this window is
    ``agentpath_ingestion.corpora.base.context_window``; the API cannot import that
    package (``tests/test_visual_steps.py`` asserts the import graph), so the sizes
    travel through the bundle instead of the code being shared.
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
    "DATASET_PREFIX",
    "DEFAULT_IMAGE_CONTEXT_AFTER",
    "DEFAULT_IMAGE_CONTEXT_BEFORE",
    "HEADING_SEGMENT_SEPARATOR",
    "IMAGE_SUFFIXES",
    "MAX_BUNDLE_BYTES",
    "MAX_SECTION_CHARS",
    "PROCEDURES_CORPUS_ID",
    "SUPPORTED_FORMAT_VERSION",
    "Corpus",
    "CorpusCatalogue",
    "CorpusChunkRecord",
    "CorpusDocumentRecord",
    "CorpusImageRecord",
    "CorpusQuestionRecord",
    "dataset_name_for",
    "default_external_root",
    "heading_label",
    "render_document_html",
]

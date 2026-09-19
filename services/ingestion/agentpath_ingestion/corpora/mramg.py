"""MRAMG-Bench's Manual subset as an AgentPath corpus.

The first implementation of :class:`~agentpath_ingestion.corpora.base.CorpusAdapter`,
and an external one on purpose: the synthetic procedures were written here, so they
cannot tell us whether this architecture works on a knowledge base it did not
author.

What the upstream ships, measured rather than assumed
----------------------------------------------------
``doc_manual.jsonl``  40 lines, each a JSON **array** ``[doc_id, text, image_ids]``
``manual_mqa.jsonl``  390 lines, each an object with ``id``, ``question``,
                      ``provenance``, ``ground_truth`` and ``images_list``
``IMAGE.zip``         1.53 GiB covering all six subsets; Manual's 2,599 images are
                      132.0 MiB of it, under ``IMAGE/images/MANUAL/``
captions              ``IMAGE/images_info/manual_imgs_collection.json``, 2,599
                      entries, **every caption an empty string**, 2,300 with a URL

Licence: Apache-2.0, stated in the repository's own ``LICENSE.txt``.

Three consequences that shaped this module
------------------------------------------
**Images are fetched selectively.** Pulling 1.53 GiB to use 8.5% of it would be
wasteful and was explicitly ruled out, so :class:`RemoteZip` reads the archive's
index over byte ranges and fetches only Manual's members - one coalesced request of
133.8 MiB. See ``remote_zip.py``.

**There are no captions to rank images by.** Every caption is empty, so the planned
caption-based image retrieval is not available for this subset. What *is* available
is position: the text carries 2,607 ``<PIC>`` markers and the image id lists carry
2,607 ids, aligned in all 40 documents. Each image therefore gets a searchable text
built from the window around its marker, and the method is named
*context-window image retrieval* rather than anything that implies a model looked
at a picture.

**The text is not touched.** Documents are stored with their markers in place and
chunks are recorded as offsets into that text, so the corpus this system is graded
on is the corpus the benchmark published.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

from agentpath_ingestion.corpora.base import (
    CorpusBundle,
    CorpusDocument,
    CorpusImage,
    CorpusQuestion,
    FetchedArtefact,
    FetchReport,
    context_window,
    locate_placeholders,
    slice_chunks,
)
from agentpath_ingestion.corpora.remote_zip import RemoteZip, ZipMember

logger = structlog.get_logger(__name__)

CORPUS_ID = "mramg-manual"
CORPUS_TITLE = "MRAMG-Bench - Manual"
CORPUS_LICENCE = "Apache-2.0"
CORPUS_ATTRIBUTION = (
    "MRAMG-Bench (MRAMG/MRAMG-Bench on Hugging Face), Manual subset. "
    "Licensed Apache-2.0. Text, images, questions and ground truth are the "
    "benchmark authors' work and are used unmodified."
)

REPO = "MRAMG/MRAMG-Bench"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
IMAGE_ARCHIVE_URL = f"{BASE_URL}/IMAGE.zip"

#: Small files pulled whole. Each is well under a megabyte and a half.
TEXT_ARTEFACTS = ("LICENSE.txt", "doc_manual.jsonl", "manual_mqa.jsonl")

#: Members of IMAGE.zip that are needed regardless of which images are used.
CAPTION_MEMBER = "IMAGE/images_info/manual_imgs_collection.json"

#: Where Manual's images live inside the archive.
IMAGE_MEMBER_PREFIX = "IMAGE/images/MANUAL/"

#: Chunking. The median document is 26.7k characters and the largest 168k, so they
#: must be cut. 1,400 characters sits near the sweet spot for bge-m3's effective
#: window while still holding a whole procedure step or two, and 200 characters of
#: overlap means a sentence on a boundary is reachable from either side.
CHUNK_TARGET_CHARS = 1_400
CHUNK_OVERLAP_CHARS = 200

#: Markdown-ish headings appear throughout these manuals ("# Personal safety
#: equipment"), which gives chunks a section label.
#:
#: Not quite for free, as first assumed. Because these documents are flattened - a
#: "line" holds a heading and the whole section it introduces - ``.+$`` matches far
#: past the heading, and the label has to be trimmed at the segment boundary. That is
#: done by ``base._heading_positions``; see ``HEADING_SEGMENT_SEPARATOR``.
HEADING_PATTERN = r"^#{1,6} .+$"

#: How much text around a ``<PIC>`` marker describes the image. Asymmetric because
#: a manual names a thing and *then* shows it far more often than the reverse, so
#: the caption-like sentence is usually just above the marker.
IMAGE_CONTEXT_BEFORE = 600
IMAGE_CONTEXT_AFTER = 300


class MramgError(RuntimeError):
    """The corpus could not be fetched or normalised."""


@dataclass(frozen=True, slots=True)
class MramgPaths:
    """Where this corpus lives on disk. All gitignored: it is someone else's data."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def images(self) -> Path:
        return self.raw / "images"

    @property
    def build(self) -> Path:
        return self.root / "build"

    @property
    def manifest(self) -> Path:
        return self.build / "manifest.json"

    def image_path(self, image_id: str) -> Path:
        return self.images / f"{image_id}.jpg"


def default_root(repo_root: Path) -> Path:
    return repo_root / "data" / "external" / "mramg"


class MramgManualAdapter:
    """Fetch and normalise MRAMG-Manual.

    Satisfies :class:`~agentpath_ingestion.corpora.base.CorpusAdapter`. Held to
    that protocol by a test, so the next corpus has a shape to copy rather than a
    precedent to guess at.
    """

    def __init__(self, root: Path, *, client: httpx.Client | None = None) -> None:
        self.paths = MramgPaths(root=root)
        self._client = client

    # -- identity ----------------------------------------------------------- #

    @property
    def corpus_id(self) -> str:
        return CORPUS_ID

    @property
    def title(self) -> str:
        return CORPUS_TITLE

    # -- fetch -------------------------------------------------------------- #

    def fetch(self, *, force: bool = False) -> FetchReport:
        """Put every raw artefact on disk, skipping what is already verified.

        Idempotent, resumable and reproducible in the ways that matter:

        * **Idempotent** because each artefact is checked for presence and size
          before anything is requested, and a second run reports
          ``downloaded_bytes == 0``.
        * **Resumable** because images are written individually as they arrive, so
          an interrupted run resumes at the first missing image rather than
          restarting the archive read.
        * **Reproducible** because every artefact's sha256 is recorded in the
          manifest, and each image is additionally verified against the CRC-32 the
          archive itself stores.
        """
        report = FetchReport(corpus_id=CORPUS_ID)
        self.paths.raw.mkdir(parents=True, exist_ok=True)
        self.paths.images.mkdir(parents=True, exist_ok=True)

        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=300.0, follow_redirects=True)
        try:
            for name in TEXT_ARTEFACTS:
                report.artefacts.append(self._fetch_text_artefact(client, name, force=force, report=report))

            wanted_ids = self._referenced_image_ids()
            self._fetch_archive_members(client, wanted_ids, force=force, report=report)
        finally:
            if owns_client:
                client.close()

        logger.info(
            "corpus.fetch.done",
            corpus=CORPUS_ID,
            artefacts=len(report.artefacts),
            downloaded_bytes=report.downloaded_bytes,
            reused_bytes=report.reused_bytes,
            problems=len(report.problems),
        )
        return report

    def _fetch_text_artefact(
        self, client: httpx.Client, name: str, *, force: bool, report: FetchReport
    ) -> FetchedArtefact:
        target = self.paths.raw / name
        if target.exists() and not force:
            body = target.read_bytes()
            report.reused_bytes += len(body)
            return self._artefact(name, target, body, source=f"{BASE_URL}/{name}", reused=True)

        response = client.get(f"{BASE_URL}/{name}")
        response.raise_for_status()
        body = response.content
        target.write_bytes(body)
        report.downloaded_bytes += len(body)
        logger.info("corpus.fetch.artefact", corpus=CORPUS_ID, name=name, bytes=len(body))
        return self._artefact(name, target, body, source=f"{BASE_URL}/{name}")

    def _fetch_archive_members(
        self,
        client: httpx.Client,
        wanted_ids: set[str],
        *,
        force: bool,
        report: FetchReport,
    ) -> None:
        """Fetch the caption collection and any missing images from IMAGE.zip."""
        caption_target = self.paths.raw / "manual_imgs_collection.json"
        missing = {image_id for image_id in wanted_ids if force or not self.paths.image_path(image_id).exists()}

        if caption_target.exists() and not force and not missing:
            body = caption_target.read_bytes()
            report.reused_bytes += len(body)
            report.artefacts.append(
                self._artefact(
                    "manual_imgs_collection.json",
                    caption_target,
                    body,
                    source=f"{IMAGE_ARCHIVE_URL}!{CAPTION_MEMBER}",
                    reused=True,
                )
            )
            for image_id in sorted(wanted_ids):
                report.reused_bytes += self.paths.image_path(image_id).stat().st_size
            logger.info("corpus.fetch.images_cached", corpus=CORPUS_ID, images=len(wanted_ids))
            # Still record the image set. Omitting it made a fully cached run report
            # 2.5 MiB "total on disk" for a corpus holding 147 MiB of images.
            report.artefacts.append(self._image_set_artefact(wanted_ids, report, reused=True))
            return

        with RemoteZip(IMAGE_ARCHIVE_URL, client=client) as archive:
            members = {member.name: member for member in archive.members()}

            caption_member = members.get(CAPTION_MEMBER)
            if caption_member is None:
                raise MramgError(f"{CAPTION_MEMBER} is not in the archive")

            image_members: list[ZipMember] = []
            for member in members.values():
                if not member.name.startswith(IMAGE_MEMBER_PREFIX):
                    continue
                if member.stem in missing:
                    image_members.append(member)

            unmatched = missing - {member.stem for member in image_members}
            if unmatched:
                report.problems.append(
                    f"{len(unmatched)} referenced images are absent from the archive: {sorted(unmatched)[:5]}"
                )

            to_fetch = [caption_member, *image_members] if not caption_target.exists() or force else image_members
            planned = archive.plan_bytes(to_fetch) if to_fetch else 0
            logger.info(
                "corpus.fetch.plan",
                corpus=CORPUS_ID,
                archive_bytes=archive.archive_size,
                members=len(to_fetch),
                planned_bytes=planned,
                share_of_archive=round(100 * planned / max(archive.archive_size, 1), 2),
            )

            started = time.perf_counter()
            for member, body in archive.extract(to_fetch):
                if member.name == CAPTION_MEMBER:
                    caption_target.write_bytes(body)
                    report.artefacts.append(
                        self._artefact(
                            "manual_imgs_collection.json",
                            caption_target,
                            body,
                            source=f"{IMAGE_ARCHIVE_URL}!{CAPTION_MEMBER}",
                        )
                    )
                    continue
                # Written one at a time so an interrupted run resumes here.
                self.paths.image_path(member.stem).write_bytes(body)

            report.downloaded_bytes += archive.bytes_fetched
            logger.info(
                "corpus.fetch.images",
                corpus=CORPUS_ID,
                written=len(image_members),
                bytes=archive.bytes_fetched,
                seconds=round(time.perf_counter() - started, 1),
            )

        report.artefacts.append(self._image_set_artefact(wanted_ids, report))

    def _image_set_artefact(
        self, wanted_ids: set[str], report: FetchReport, *, reused: bool = False
    ) -> FetchedArtefact:
        """One artefact row for the whole image set.

        2,599 rows would drown the manifest, and per-file integrity is already
        covered: every image was checked against the archive's own CRC-32 as it was
        extracted. What this adds is a single digest over the set, so two machines
        can compare whether they hold the same images.
        """
        present = sorted(image_id for image_id in wanted_ids if self.paths.image_path(image_id).exists())
        digest = hashlib.sha256()
        total = 0
        for image_id in present:
            data = self.paths.image_path(image_id).read_bytes()
            digest.update(image_id.encode("utf-8"))
            digest.update(data)
            total += len(data)

        if len(present) != len(wanted_ids):
            report.problems.append(f"{len(wanted_ids) - len(present)} of {len(wanted_ids)} images are still missing")
        return FetchedArtefact(
            name=f"images/ ({len(present)} files)",
            path=str(self.paths.images),
            byte_size=total,
            sha256=digest.hexdigest(),
            source=f"{IMAGE_ARCHIVE_URL}!{IMAGE_MEMBER_PREFIX}*",
            reused=reused,
        )

    def _artefact(self, name: str, path: Path, body: bytes, *, source: str, reused: bool = False) -> FetchedArtefact:
        return FetchedArtefact(
            name=name,
            path=str(path),
            byte_size=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            source=source,
            reused=reused,
        )

    # -- load --------------------------------------------------------------- #

    def _raw_documents(self) -> list[tuple[str, str, list[str]]]:
        path = self.paths.raw / "doc_manual.jsonl"
        if not path.exists():
            raise MramgError(f"{path} is missing; run: make corpus-fetch CORPUS=mramg-manual")
        rows: list[tuple[str, str, list[str]]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                # The upstream stores a 3-element array, not an object.
                if not isinstance(payload, list) or len(payload) != 3:
                    raise MramgError(f"doc_manual.jsonl line {number} is not a 3-element array")
                document_id, text, image_ids = payload
                rows.append((str(document_id), str(text), [str(item) for item in image_ids]))
        return rows

    def _referenced_image_ids(self) -> set[str]:
        ids: set[str] = set()
        for _document_id, _text, image_ids in self._raw_documents():
            ids.update(image_ids)
        for question in self._raw_questions():
            ids.update(question.ground_truth_image_ids)
        return ids

    def _raw_questions(self) -> list[CorpusQuestion]:
        path = self.paths.raw / "manual_mqa.jsonl"
        if not path.exists():
            raise MramgError(f"{path} is missing; run: make corpus-fetch CORPUS=mramg-manual")
        questions: list[CorpusQuestion] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                questions.append(
                    CorpusQuestion(
                        question_id=str(row["id"]),
                        question=str(row["question"]),
                        ground_truth_answer=str(row["ground_truth"]),
                        provenance_document_ids=tuple(str(item) for item in row.get("provenance") or ()),
                        ground_truth_image_ids=tuple(str(item) for item in row.get("images_list") or ()),
                        # The upstream carries no difficulty label for Manual. The
                        # split therefore stratifies on the one property that does
                        # vary and matters: how many images the answer needs.
                        difficulty=None,
                    )
                )
        return questions

    def _captions(self) -> dict[str, dict[str, str]]:
        path = self.paths.raw / "manual_imgs_collection.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def load(self) -> CorpusBundle:
        """Normalise what is on disk into a bundle. No network, no rewriting."""
        captions = self._captions()
        documents: list[CorpusDocument] = []
        chunks = []

        for document_id, text, image_ids in self._raw_documents():
            offsets = locate_placeholders(text)
            if len(offsets) != len(image_ids):
                raise MramgError(
                    f"document {document_id}: {len(offsets)} placeholders but {len(image_ids)} "
                    "image ids; the positional mapping would be wrong"
                )

            images: list[CorpusImage] = []
            for index, (image_id, offset) in enumerate(zip(image_ids, offsets, strict=True)):
                meta = captions.get(image_id) or {}
                path = self.paths.image_path(image_id)
                exists = path.exists()
                images.append(
                    CorpusImage(
                        image_id=image_id,
                        document_id=document_id,
                        position_index=index,
                        char_offset=offset,
                        local_path=str(path.relative_to(self.paths.root.parents[2])) if exists else None,
                        source_url=(meta.get("image_url") or None),
                        caption=(meta.get("image_caption") or None),
                        sha256=_sha256_file(path) if exists else None,
                        byte_size=path.stat().st_size if exists else None,
                    )
                )

            document = CorpusDocument(
                document_id=document_id,
                title=_title_of(text, document_id),
                text=text,
                images=tuple(images),
                # The corpus records no per-document URL; its images carry upstream
                # URLs, the documents do not. So the original this system can offer
                # is the local copy, served by id.
                source_url=None,
                source_id=f"{CORPUS_ID}-{document_id}",
            )
            documents.append(document)
            chunks.extend(
                slice_chunks(
                    document,
                    target_chars=CHUNK_TARGET_CHARS,
                    overlap_chars=CHUNK_OVERLAP_CHARS,
                    heading_pattern=HEADING_PATTERN,
                )
            )

        questions = tuple(self._raw_questions())
        bundle = CorpusBundle(
            corpus_id=CORPUS_ID,
            title=CORPUS_TITLE,
            attribution=CORPUS_ATTRIBUTION,
            licence=CORPUS_LICENCE,
            documents=tuple(documents),
            chunks=tuple(chunks),
            questions=questions,
            manifest={
                "source_repo": REPO,
                "source_url": f"https://huggingface.co/datasets/{REPO}",
                "licence": CORPUS_LICENCE,
                "chunk_target_chars": CHUNK_TARGET_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "image_context_before": IMAGE_CONTEXT_BEFORE,
                "image_context_after": IMAGE_CONTEXT_AFTER,
                "image_retrieval_mode": "context-window",
                "captions_available": sum(1 for image in (i for d in documents for i in d.images) if image.has_caption),
            },
        )

        problems = bundle.integrity_problems()
        if problems:
            raise MramgError("corpus failed its integrity check: " + "; ".join(problems[:5]))
        logger.info("corpus.loaded", corpus=CORPUS_ID, **bundle.counts)
        return bundle

    # -- what an image offers when it has no caption ------------------------ #

    def image_search_text(self, bundle: CorpusBundle, image: CorpusImage) -> str:
        """The text that makes *image* findable.

        Its caption when it has one, and otherwise the window of document text
        around its placeholder. For MRAMG-Manual that is always the window, because
        no image in the subset has a caption.
        """
        if image.has_caption:
            return str(image.caption)
        document = bundle.document(image.document_id)
        if document is None:
            return ""
        return context_window(
            document.text,
            image.char_offset,
            before=IMAGE_CONTEXT_BEFORE,
            after=IMAGE_CONTEXT_AFTER,
        )


#: A heading shorter than this is a model number or a fragment rather than a
#: title, so the next segment is joined onto it. Tuned against all 40 documents:
#: it turns "225B" into "225B Operator's manual" and "2 in 1" into "2 in 1 Steam
#: Mop & Steam Cleaner", while leaving "Dishwasher" and "Introduction" alone.
MIN_TITLE_CHARS = 8

#: Upstream flattened these manuals, and a run of two or more spaces is where a
#: line or heading break used to be. It is the only segment boundary available.
SEGMENT_SEPARATOR = "  "


def _title_of(text: str, document_id: str) -> str:
    """The document's own opening heading, taken verbatim.

    These documents arrive flattened: headings, their text and the body all sit on
    one line separated by runs of two or more spaces, and the first line often
    begins with a ``<PIC>`` marker because the manual opens with a cover image.
    An earlier version skipped any line containing a marker and fell through to
    the document id for every document that did - which was most of them.

    So the marker is removed first, the first ``#`` heading is located anywhere in
    the opening text, and its segment is taken. Nothing is composed or translated;
    a document with no usable heading keeps its id rather than being given an
    invented name.
    """
    import re

    opening = text[:2_000].replace("<PIC>", " ")
    match = re.search(r"#{1,6}\s+(.+)", opening, re.DOTALL)
    candidate = match.group(1) if match else opening

    segments = [
        segment.strip().lstrip("#").strip()
        for segment in re.split(r" {2,}|\n", candidate)
        if segment.strip().lstrip("#").strip()
    ]
    if not segments:
        return f"document {document_id}"

    title = segments[0]
    # A very short first segment is a model number; join the next one onto it.
    if len(title) < MIN_TITLE_CHARS and len(segments) > 1:
        title = f"{title} {segments[1]}"
    return title[:160]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(131_072), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CAPTION_MEMBER",
    "CHUNK_OVERLAP_CHARS",
    "CHUNK_TARGET_CHARS",
    "CORPUS_ATTRIBUTION",
    "CORPUS_ID",
    "CORPUS_LICENCE",
    "CORPUS_TITLE",
    "HEADING_PATTERN",
    "IMAGE_ARCHIVE_URL",
    "IMAGE_CONTEXT_AFTER",
    "IMAGE_CONTEXT_BEFORE",
    "MramgError",
    "MramgManualAdapter",
    "MramgPaths",
    "default_root",
]

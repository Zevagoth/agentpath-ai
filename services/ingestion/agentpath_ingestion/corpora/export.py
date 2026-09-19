"""Write a normalised corpus to disk in a form the API can read.

The API is forbidden from importing this package - ``tests/test_visual_steps.py``
asserts on the import graph, because that is what guarantees YAML loading, image
hashing and network calls cannot happen during a request. So the two sides meet at
files instead of at objects, exactly as ``agentpath.procedures`` meets the
procedure YAML and ``agentpath.screenshots`` meets the capture sidecars.

What gets written, and why in this shape:

``manifest.json``   one small file with the corpus identity, licence, attribution
                    and counts. Read first, so the API can register a corpus and
                    render its attribution without parsing the documents.
``documents.jsonl`` one line per document, carrying the text **verbatim** and its
                    images with their positions. This is what the source viewer
                    serves and what a reader opens to check a citation.
``chunks.jsonl``    one line per retrieval unit, as offsets into the document.
``questions.jsonl`` the benchmark's own questions and ground truth, untouched.

JSON Lines rather than one big JSON document because the document file is a few
megabytes and a reader that wants one document should not have to hold all forty in
memory to find it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from agentpath_ingestion.corpora.base import CorpusBundle, write_jsonl

logger = structlog.get_logger(__name__)

MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
CHUNKS_NAME = "chunks.jsonl"
QUESTIONS_NAME = "questions.jsonl"

#: Bumped when the on-disk shape changes in a way a reader must notice. The API
#: refuses a bundle it does not recognise rather than guessing at missing fields.
BUNDLE_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExportReport:
    corpus_id: str
    directory: str
    documents: int
    chunks: int
    questions: int
    images: int
    bytes_written: int


def export_bundle(bundle: CorpusBundle, target: Path, *, images_root: Path | None = None) -> ExportReport:
    """Write *bundle* into *target*, replacing whatever was there for this corpus.

    Replacing rather than merging: a corpus is rebuilt from its raw artefacts, so a
    stale document left behind from a previous shape would be a document the API
    serves and the evaluation has never heard of. Only this corpus's directory is
    touched, which is what makes "delete and rebuild one corpus" safe.
    """
    target.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "corpus_id": bundle.corpus_id,
        "title": bundle.title,
        "licence": bundle.licence,
        "attribution": bundle.attribution,
        "counts": bundle.counts,
        # Where the image bytes live, repository-relative, so the API can build its
        # own closed-set registry without being told separately.
        "images_root": str(images_root) if images_root else None,
        "build": dict(bundle.manifest),
    }
    manifest_path = target / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_jsonl(
        target / DOCUMENTS_NAME,
        [
            {
                "document_id": document.document_id,
                "title": document.title,
                # Verbatim, markers included.
                "text": document.text,
                "source_url": document.source_url,
                "source_id": document.source_id,
                "images": [
                    {
                        "image_id": image.image_id,
                        "position_index": image.position_index,
                        "char_offset": image.char_offset,
                        "local_path": image.local_path,
                        "source_url": image.source_url,
                        "caption": image.caption,
                        "byte_size": image.byte_size,
                        "sha256": image.sha256,
                    }
                    for image in document.images
                ],
            }
            for document in bundle.documents
        ],
    )

    write_jsonl(
        target / CHUNKS_NAME,
        [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "section": chunk.section,
                "image_ids": list(chunk.image_ids),
                "ordinal": chunk.ordinal,
            }
            for chunk in bundle.chunks
        ],
    )

    write_jsonl(
        target / QUESTIONS_NAME,
        [
            {
                "question_id": question.question_id,
                "question": question.question,
                "ground_truth_answer": question.ground_truth_answer,
                "provenance_document_ids": list(question.provenance_document_ids),
                "ground_truth_image_ids": list(question.ground_truth_image_ids),
                "difficulty": question.difficulty,
            }
            for question in bundle.questions
        ],
    )

    written = sum(
        (target / name).stat().st_size for name in (MANIFEST_NAME, DOCUMENTS_NAME, CHUNKS_NAME, QUESTIONS_NAME)
    )
    report = ExportReport(
        corpus_id=bundle.corpus_id,
        directory=str(target),
        documents=len(bundle.documents),
        chunks=len(bundle.chunks),
        questions=len(bundle.questions),
        images=len(bundle.images),
        bytes_written=written,
    )
    # `asdict`, not `__dict__`: these records use `slots=True` and so have none.
    logger.info("corpus.exported", **asdict(report))
    return report


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "CHUNKS_NAME",
    "DOCUMENTS_NAME",
    "MANIFEST_NAME",
    "QUESTIONS_NAME",
    "ExportReport",
    "export_bundle",
]

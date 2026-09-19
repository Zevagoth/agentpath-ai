"""Index an external corpus into its own RAGFlow dataset.

Isolation is the requirement, and it is enforced by the coarsest mechanism
available rather than by a metadata filter: **one RAGFlow dataset per corpus.**
``agentpath-northwind-procedures`` holds the synthetic procedures and
``agentpath-mramg-manual`` holds the benchmark, and a retrieval call names exactly
one dataset. A shared index filtered by a ``corpus_id`` field would put the
separation one forgotten ``WHERE`` clause away from being wrong, and "results from
the wrong corpus" is the specific failure Stage 2B1 was told to prevent.

What gets uploaded, and why not the chunks
-----------------------------------------
One file per corpus document, carrying the document's text verbatim. RAGFlow parses
and chunks it.

The alternative was to upload the 1,506 chunks this repository computed, as manual
chunks, so retrieval would return exactly the spans whose offsets the provenance
layer already knows. It was rejected for two measured reasons. RAGFlow deletes
manual chunks whenever a document is re-parsed - Stage 2A hit that and documents it
in ``visual_chunks.py`` - so the index would be one re-parse away from empty. And
the chunk API is one request per chunk, making a rebuild 1,506 round trips.

The cost of letting RAGFlow chunk is that its spans are not ours, so a retrieved
passage arrives without offsets. That is recovered at query time by locating the
passage's text in the document it came from, which is cheap and exact for text we
never rewrote - see ``agentpath.corpus``. A passage that cannot be located is
reported rather than guessed at.

Metadata carries the corpus
---------------------------
``corpus_id``, ``document_id``, ``source_id`` and ``title`` travel as RAGFlow
document metadata, so a retrieved passage can be attributed without a second
lookup, and so a stray result from another corpus would be visibly labelled rather
than silently blend in.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from agentpath.corpus import DATASET_PREFIX, dataset_name_for
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath_ingestion.corpora.base import CorpusBundle
from agentpath_ingestion.ingest import PARSER_CONFIG, SHA_FIELD, IngestionError, IngestionService

logger = structlog.get_logger(__name__)

#: Written beside the exported bundle. A report of the last run, not a source of
#: truth: RAGFlow's own listing decides what exists, exactly as the visual-chunk
#: manifest does.
INGEST_MANIFEST_NAME = "ingest.json"

# ``DATASET_PREFIX`` and ``dataset_name_for`` are imported from ``agentpath.corpus``
# and re-exported here, where callers expect them. The rule that a corpus's dataset is
# ``agentpath-{corpus_id}`` has to be identical on the writing side and the reading
# side, and the reader cannot import this package, so the single definition lives
# there. A second copy would let the API look for a dataset ingestion never created.


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    corpus_document_id: str
    filename: str
    ragflow_document_id: str
    action: str
    byte_size: int


@dataclass
class CorpusIngestReport:
    corpus_id: str
    dataset_name: str
    dataset_id: str = ""
    embedding_model: str = ""
    outcomes: list[DocumentOutcome] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.action] = counts.get(outcome.action, 0) + 1
        return counts

    @property
    def document_ids(self) -> dict[str, str]:
        """Corpus document id to RAGFlow document id."""
        return {item.corpus_document_id: item.ragflow_document_id for item in self.outcomes}

    @property
    def ok(self) -> bool:
        return not self.problems


class CorpusIngestionService:
    """Upload one corpus into its own dataset, idempotently."""

    def __init__(
        self,
        client: RAGFlowClient,
        *,
        embedding_model: str,
        parse_timeout: float = 1_800.0,
        poll_interval: float = 3.0,
    ) -> None:
        self._client = client
        self._embedding_model = embedding_model
        self._parse_timeout = parse_timeout
        self._poll_interval = poll_interval

    async def ingest(self, bundle: CorpusBundle, *, force: bool = False) -> CorpusIngestReport:
        """Bring the corpus's dataset in line with *bundle*.

        Idempotent through a content hash on each document, the same mechanism the
        procedures pipeline uses: a document whose text has not changed is left
        alone, so a second run parses nothing. ``force`` replaces every document.

        Documents in the dataset that the bundle no longer contains are deleted, so
        a corpus can genuinely be rebuilt rather than accumulating.
        """
        started = time.perf_counter()
        dataset_name = dataset_name_for(bundle.corpus_id)
        report = CorpusIngestReport(corpus_id=bundle.corpus_id, dataset_name=dataset_name)

        # Dataset creation, the embedding-model guard and the parse poll are all
        # identical to the procedures pipeline, so they are reused rather than
        # reimplemented - including the refusal to use a dataset built with a
        # different embedding model.
        plumbing = IngestionService(
            self._client,
            dataset_name=dataset_name,
            embedding_model=self._embedding_model,
            parse_timeout=self._parse_timeout,
            poll_interval=self._poll_interval,
        )
        dataset_id = await plumbing.ensure_dataset()
        report.dataset_id = dataset_id
        report.embedding_model = self._embedding_model

        existing = {str(doc.get("name")): doc for doc in await self._client.list_documents(dataset_id)}
        expected: set[str] = set()
        to_parse: list[str] = []

        for document in bundle.documents:
            filename = _filename_for(bundle.corpus_id, document.document_id)
            expected.add(filename)
            payload = document.text.encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            current = existing.get(filename)

            if current is not None and not force and _recorded_sha(current) == digest:
                report.outcomes.append(
                    DocumentOutcome(
                        corpus_document_id=document.document_id,
                        filename=filename,
                        ragflow_document_id=str(current.get("id")),
                        action="unchanged",
                        byte_size=len(payload),
                    )
                )
                continue

            action = "created"
            if current is not None:
                # RAGFlow has no replace-content call, so the old document goes and
                # a new one arrives. Scoped to this one id.
                await self._client.delete_documents(dataset_id, [str(current.get("id"))])
                action = "updated"

            ragflow_id = await self._client.upload_document(dataset_id, filename, payload)
            await self._client.update_document(
                dataset_id,
                ragflow_id,
                meta_fields={
                    "corpus_id": bundle.corpus_id,
                    "document_id": document.document_id,
                    "source_id": document.source_id or "",
                    "title": document.title,
                    "licence": bundle.licence,
                    SHA_FIELD: digest,
                },
                parser_config=PARSER_CONFIG,
            )
            to_parse.append(ragflow_id)
            report.outcomes.append(
                DocumentOutcome(
                    corpus_document_id=document.document_id,
                    filename=filename,
                    ragflow_document_id=ragflow_id,
                    action=action,
                    byte_size=len(payload),
                )
            )
            logger.info(
                "corpus.ingest.uploaded",
                corpus=bundle.corpus_id,
                document=document.document_id,
                action=action,
                bytes=len(payload),
            )

        stale = [str(doc.get("id")) for name, doc in existing.items() if name not in expected]
        if stale:
            await self._client.delete_documents(dataset_id, stale)
            report.removed = stale
            logger.info("corpus.ingest.removed", corpus=bundle.corpus_id, count=len(stale))

        if to_parse:
            logger.info("corpus.ingest.parsing", corpus=bundle.corpus_id, documents=len(to_parse))
            await self._client.parse_documents(dataset_id, to_parse)
            try:
                await plumbing.wait_for_parsing(dataset_id, to_parse)
            except IngestionError as exc:
                report.problems.append(str(exc))

        report.seconds = round(time.perf_counter() - started, 1)
        logger.info(
            "corpus.ingest.done",
            corpus=bundle.corpus_id,
            dataset=dataset_name,
            **report.actions,
            removed=len(report.removed),
            seconds=report.seconds,
        )
        return report

    async def chunk_counts(self, report: CorpusIngestReport) -> dict[str, int]:
        """Chunks RAGFlow produced per document, read back from the engine.

        Read rather than assumed: the number RAGFlow settled on is the number that
        will be retrieved over, and it is not the number this repository computed.
        """
        counts: dict[str, int] = {}
        for outcome in report.outcomes:
            # Paginated: RAGFlow caps a page at 100 and the largest manual produced
            # 97, so a single call was one busy document away from undercounting.
            counts[outcome.corpus_document_id] = await self._client.count_chunks(
                report.dataset_id, outcome.ragflow_document_id
            )
        return counts

    async def delete_corpus(self, corpus_id: str) -> int:
        """Remove every document of one corpus, leaving other datasets untouched.

        The isolation guarantee in its destructive direction: this names one dataset
        and can only affect that one.
        """
        dataset_id = await self._client.find_dataset_id(dataset_name_for(corpus_id))
        if not dataset_id:
            return 0
        documents = await self._client.list_documents(dataset_id)
        ids = [str(doc.get("id")) for doc in documents if doc.get("id")]
        if ids:
            await self._client.delete_documents(dataset_id, ids)
        logger.info("corpus.ingest.deleted", corpus=corpus_id, documents=len(ids))
        return len(ids)


def _filename_for(corpus_id: str, document_id: str) -> str:
    """Stable, unique filename. RAGFlow keys documents by name within a dataset."""
    return f"{corpus_id}-{document_id}.md"


def _recorded_sha(document: dict[str, Any]) -> str:
    meta = document.get("meta_fields") or {}
    return str(meta.get(SHA_FIELD) or "")


def write_ingest_manifest(
    report: CorpusIngestReport, target_dir: Path, chunk_counts: dict[str, int] | None = None
) -> Path:
    """Record what the last run indexed, for a human and for the evaluation.

    Gitignored, like every other artefact naming RAGFlow ids: those ids belong to
    one local instance and mean nothing anywhere else.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_id": report.corpus_id,
        "dataset_name": report.dataset_name,
        "dataset_id": report.dataset_id,
        "embedding_model": report.embedding_model,
        "actions": report.actions,
        "seconds": report.seconds,
        "documents": [
            {
                "corpus_document_id": item.corpus_document_id,
                "filename": item.filename,
                "ragflow_document_id": item.ragflow_document_id,
                "action": item.action,
                "byte_size": item.byte_size,
                "ragflow_chunks": (chunk_counts or {}).get(item.corpus_document_id),
            }
            for item in report.outcomes
        ],
        "removed_document_ids": report.removed,
        "problems": report.problems,
    }
    path = target_dir / INGEST_MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DATASET_PREFIX",
    "INGEST_MANIFEST_NAME",
    "CorpusIngestReport",
    "CorpusIngestionService",
    "DocumentOutcome",
    "dataset_name_for",
    "write_ingest_manifest",
]

"""Ingest the rendered procedures into RAGFlow.

The path this owns: rendered Markdown -> RAGFlow document -> metadata -> parsed,
embedded chunks -> verified chunk counts.

Two properties matter more than speed here:

* **Idempotence.** Re-running must converge, not duplicate. Each document carries
  the SHA-256 of the text it was created from, so a re-run can tell "unchanged"
  from "edited" and only touch what actually changed.
* **Honesty about progress.** Parsing is asynchronous inside RAGFlow. The CLI
  waits for RAGFlow to report DONE rather than assuming success, and surfaces
  RAGFlow's own progress messages when something stalls.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError
from agentpath_ingestion.dataset import LoadedProcedure, build_dir, load_procedures
from agentpath_ingestion.render import RenderedProcedure, render
from agentpath_ingestion.visual_chunks import (
    VisualChunkReport,
    attach_visual_chunks,
    inspect_visual_chunks,
    write_visual_chunk_manifest,
)

logger = structlog.get_logger(__name__)

#: Metadata key holding the hash of the rendered text, used for change detection.
SHA_FIELD = "content_sha256"

DATASET_DESCRIPTION = (
    "AgentPath AI - fictional Northwind Travel customer-service procedures. "
    "Synthetic demo data; contains no real company policy."
)

#: Chunking: ~512 tokens split on newlines suits these documents, where each
#: section (steps, exceptions, escalation) is a few hundred tokens and should stay
#: intact so a retrieved chunk is a self-contained instruction.
PARSER_CONFIG: dict[str, Any] = {
    "chunk_token_num": 512,
    "delimiter": "\n",
    "auto_keywords": 0,
    "auto_questions": 0,
    "layout_recognize": "Plain Text",
}


class IngestionError(RuntimeError):
    """Ingestion could not complete."""


@dataclass
class DocumentOutcome:
    """What happened to one document during ingestion."""

    procedure_id: str
    filename: str
    document_id: str
    action: str  # created | updated | unchanged
    chunk_count: int = 0
    token_count: int = 0
    status: str = "unknown"
    progress_messages: tuple[str, ...] = ()


@dataclass
class IngestionReport:
    """Result of a full ingestion run."""

    dataset_id: str
    dataset_name: str
    embedding_model: str
    outcomes: list[DocumentOutcome] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    #: State of the image-bearing screenshot chunks. ``None`` only when the run
    #: never got as far as looking.
    visual: VisualChunkReport | None = None

    @property
    def document_ids(self) -> dict[str, str]:
        """Procedure id -> RAGFlow document id, for everything that has one."""
        return {o.procedure_id: o.document_id for o in self.outcomes if o.document_id}

    @property
    def created(self) -> list[DocumentOutcome]:
        return [o for o in self.outcomes if o.action == "created"]

    @property
    def updated(self) -> list[DocumentOutcome]:
        return [o for o in self.outcomes if o.action == "updated"]

    @property
    def unchanged(self) -> list[DocumentOutcome]:
        return [o for o in self.outcomes if o.action == "unchanged"]

    @property
    def total_chunks(self) -> int:
        return sum(o.chunk_count for o in self.outcomes)

    @property
    def all_parsed(self) -> bool:
        return bool(self.outcomes) and all(o.chunk_count > 0 for o in self.outcomes)


class IngestionService:
    """Creates the dataset, uploads documents, and waits for parsing."""

    def __init__(
        self,
        client: RAGFlowClient,
        *,
        dataset_name: str,
        embedding_model: str,
        parse_timeout: float = 600.0,
        poll_interval: float = 3.0,
    ) -> None:
        self._client = client
        self._dataset_name = dataset_name
        self._embedding_model = embedding_model
        self._parse_timeout = parse_timeout
        self._poll_interval = poll_interval

    # -- dataset ----------------------------------------------------------- #

    async def ensure_dataset(self) -> str:
        """Return the dataset ID, creating the dataset when absent.

        The embedding model is fixed at creation time in RAGFlow and cannot be
        changed afterwards, so a mismatch is reported rather than silently
        tolerated: it would mean queries are embedded by a different model than
        the documents.
        """
        existing = await self._client.find_dataset_id(self._dataset_name)
        if existing:
            datasets = await self._client.list_datasets(name=self._dataset_name)
            for dataset in datasets:
                if dataset.get("id") == existing:
                    actual = str(dataset.get("embedding_model") or "")
                    if actual and self._embedding_model.split("@")[0] not in actual:
                        raise IngestionError(
                            f"dataset {self._dataset_name!r} was created with embedding model "
                            f"{actual!r}, but configuration asks for {self._embedding_model!r}. "
                            "RAGFlow cannot change a dataset's embedding model; delete the dataset "
                            "in the RAGFlow UI and re-run ingestion."
                        )
            return existing

        dataset_id = await self._client.create_dataset(
            name=self._dataset_name,
            description=DATASET_DESCRIPTION,
            embedding_model=self._embedding_model,
            chunk_method="naive",
            parser_config=PARSER_CONFIG,
        )
        logger.info("ingest.dataset_created", dataset=self._dataset_name, dataset_id=dataset_id)
        return dataset_id

    # -- ingestion --------------------------------------------------------- #

    async def ingest(self, *, force: bool = False) -> IngestionReport:
        """Bring RAGFlow's content in line with the rendered procedures."""
        started = time.perf_counter()
        loaded = load_procedures()
        rendered = [render(item) for item in loaded]
        dataset_id = await self.ensure_dataset()

        existing = {doc.get("name"): doc for doc in await self._client.list_documents(dataset_id)}

        report = IngestionReport(
            dataset_id=dataset_id,
            dataset_name=self._dataset_name,
            embedding_model=self._embedding_model,
        )
        to_parse: list[str] = []

        for document in rendered:
            current = existing.get(document.filename)
            if current is None:
                outcome = await self._create(dataset_id, document)
                to_parse.append(outcome.document_id)
            elif force or self._has_changed(current, document):
                outcome = await self._replace(dataset_id, current, document)
                to_parse.append(outcome.document_id)
            else:
                outcome = DocumentOutcome(
                    procedure_id=document.procedure_id,
                    filename=document.filename,
                    document_id=str(current.get("id")),
                    action="unchanged",
                    chunk_count=int(current.get("chunk_count") or 0),
                    token_count=int(current.get("token_count") or 0),
                    status=str(current.get("run") or "unknown"),
                )
                # A document that never finished parsing must be retried even when
                # its text is unchanged, otherwise it stays invisible to search.
                # Re-uploaded rather than re-parsed: a bare re-parse keeps the
                # document id, so any screenshot orphaned in storage by the
                # previous attempt still occupies the chunk id this run would
                # write to, and the image could not be attached. A re-upload moves
                # every chunk id to a free key.
                if outcome.chunk_count == 0:
                    outcome = await self._replace(dataset_id, current, document)
                    to_parse.append(outcome.document_id)
            report.outcomes.append(outcome)

        if to_parse:
            await self._client.parse_documents(dataset_id, to_parse)
            logger.info("ingest.parse_started", documents=len(to_parse))
            await self._wait_for_parsing(dataset_id, to_parse)

        # Strictly after parsing. RAGFlow's parse deletes every chunk of a
        # document, manual ones included, so anything attached earlier in this
        # method would have been thrown away by the call above.
        report.visual = await self._attach_visual_chunks(dataset_id, loaded, report)

        # Counted last, so the reported chunk totals include the visual chunks.
        await self._refresh_counts(dataset_id, report)
        report.elapsed_seconds = time.perf_counter() - started
        return report

    async def _attach_visual_chunks(
        self, dataset_id: str, loaded: list[LoadedProcedure], report: IngestionReport
    ) -> VisualChunkReport:
        """Attach the screenshot chunks and record what was written.

        A screenshot problem must not lose an otherwise good ingestion run, but it
        must not pass unnoticed either: the failure is turned into a report the
        CLI prints and exits non-zero on, rather than an exception that hides the
        document outcomes above it.
        """
        visual = await attach_visual_chunks(
            self._client,
            dataset_id,
            document_ids=report.document_ids,
            procedures=loaded,
        )
        try:
            path = write_visual_chunk_manifest(visual, build_dir())
        except OSError as exc:
            # The manifest is a report, not the source of truth, so failing to
            # write it must not fail an ingestion that already succeeded.
            logger.warning("ingest.visual_manifest_unwritten", error=str(exc))
        else:
            logger.info("ingest.visual_manifest", chunks=len(visual.outcomes), file=path.name)
        return visual

    async def _create(self, dataset_id: str, document: RenderedProcedure) -> DocumentOutcome:
        document_id = await self._client.upload_document(
            dataset_id, document.filename, document.markdown.encode("utf-8")
        )
        await self._apply_metadata(dataset_id, document_id, document)
        logger.info("ingest.uploaded", procedure=document.procedure_id, document_id=document_id)
        return DocumentOutcome(
            procedure_id=document.procedure_id,
            filename=document.filename,
            document_id=document_id,
            action="created",
        )

    async def _replace(self, dataset_id: str, current: dict[str, Any], document: RenderedProcedure) -> DocumentOutcome:
        """Replace an edited document.

        RAGFlow has no "replace content" call, so the old document is removed and
        a new one uploaded. Scoped to exactly this one document ID.
        """
        await self._client.delete_documents(dataset_id, [str(current.get("id"))])
        outcome = await self._create(dataset_id, document)
        return DocumentOutcome(
            procedure_id=outcome.procedure_id,
            filename=outcome.filename,
            document_id=outcome.document_id,
            action="updated",
        )

    async def _apply_metadata(self, dataset_id: str, document_id: str, document: RenderedProcedure) -> None:
        """Attach filterable metadata plus the content hash."""
        meta_fields = dict(document.metadata)
        meta_fields[SHA_FIELD] = document.sha256
        await self._client.update_document(
            dataset_id,
            document_id,
            meta_fields=meta_fields,
            parser_config=PARSER_CONFIG,
        )

    @staticmethod
    def _has_changed(current: dict[str, Any], document: RenderedProcedure) -> bool:
        meta = current.get("meta_fields") or {}
        recorded = str(meta.get(SHA_FIELD) or "")
        return recorded != document.sha256

    # -- parsing ----------------------------------------------------------- #

    async def wait_for_parsing(self, dataset_id: str, document_ids: list[str]) -> None:
        """Poll until every document reports DONE, or fail with RAGFlow's reason.

        Public because the external-corpus pipeline reuses it: the poll, its timeout
        and its error wording are the same problem for any dataset, and a second
        copy would be a second thing to keep correct.
        """
        await self._wait_for_parsing(dataset_id, document_ids)

    async def _wait_for_parsing(self, dataset_id: str, document_ids: list[str]) -> None:
        """Poll until every document reports DONE, or fail with RAGFlow's reason."""
        pending = set(document_ids)
        deadline = time.monotonic() + self._parse_timeout

        while pending and time.monotonic() < deadline:
            await asyncio.sleep(self._poll_interval)
            documents = {doc.get("id"): doc for doc in await self._client.list_documents(dataset_id)}
            for document_id in list(pending):
                document = documents.get(document_id)
                if document is None:
                    continue
                status = str(document.get("run") or "")
                if status in {"DONE", "3"}:
                    pending.discard(document_id)
                    logger.info(
                        "ingest.parsed",
                        document_id=document_id,
                        chunks=document.get("chunk_count"),
                        seconds=document.get("process_duration"),
                    )
                elif status in {"FAIL", "4"}:
                    messages = _progress_messages(document)
                    raise IngestionError(
                        f"RAGFlow failed to parse {document.get('name')}: "
                        + (messages[-1] if messages else "no reason reported")
                    )

        if pending:
            raise IngestionError(
                f"parsing did not finish within {self._parse_timeout:.0f}s for "
                f"{len(pending)} document(s). Check: make ragflow-logs"
            )

    async def _refresh_counts(self, dataset_id: str, report: IngestionReport) -> None:
        documents = {doc.get("name"): doc for doc in await self._client.list_documents(dataset_id)}
        for index, outcome in enumerate(report.outcomes):
            document = documents.get(outcome.filename)
            if not document:
                continue
            report.outcomes[index] = DocumentOutcome(
                procedure_id=outcome.procedure_id,
                filename=outcome.filename,
                document_id=outcome.document_id,
                action=outcome.action,
                chunk_count=int(document.get("chunk_count") or 0),
                token_count=int(document.get("token_count") or 0),
                status=str(document.get("run") or "unknown"),
                progress_messages=_progress_messages(document),
            )

    # -- verification ------------------------------------------------------ #

    async def verify(self) -> IngestionReport:
        """Report what is actually in RAGFlow, without changing anything."""
        loaded = load_procedures()
        rendered = {item.filename: item for item in (render(item) for item in loaded)}
        dataset_id = await self._client.find_dataset_id(self._dataset_name)
        if not dataset_id:
            raise IngestionError(f"dataset {self._dataset_name!r} does not exist in RAGFlow. Run: make ingest")

        report = IngestionReport(
            dataset_id=dataset_id,
            dataset_name=self._dataset_name,
            embedding_model=self._embedding_model,
        )
        for document in await self._client.list_documents(dataset_id):
            name = str(document.get("name"))
            expected = rendered.get(name)
            meta = document.get("meta_fields") or {}
            action = "unchanged"
            if expected is None:
                action = "unexpected"
            elif str(meta.get(SHA_FIELD) or "") != expected.sha256:
                action = "stale"
            report.outcomes.append(
                DocumentOutcome(
                    procedure_id=str(meta.get("procedure_id") or name.removesuffix(".md")),
                    filename=name,
                    document_id=str(document.get("id")),
                    action=action,
                    chunk_count=int(document.get("chunk_count") or 0),
                    token_count=int(document.get("token_count") or 0),
                    status=str(document.get("run") or "unknown"),
                    progress_messages=_progress_messages(document),
                )
            )

        missing = set(rendered) - {o.filename for o in report.outcomes}
        for filename in sorted(missing):
            item = rendered[filename]
            report.outcomes.append(
                DocumentOutcome(
                    procedure_id=item.procedure_id,
                    filename=filename,
                    document_id="",
                    action="missing",
                    status="absent",
                )
            )

        # Read-only, like the rest of verify: it lists chunks, reads image ids by
        # id, and asks storage whether an object key is occupied. It writes
        # nothing, so a blocked screenshot is reported here rather than repaired.
        report.visual = await inspect_visual_chunks(
            self._client,
            dataset_id,
            document_ids=report.document_ids,
            procedures=loaded,
        )
        return report


def _progress_messages(document: dict[str, Any]) -> tuple[str, ...]:
    raw = document.get("progress_msg")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    if isinstance(raw, str) and raw.strip():
        return tuple(line for line in raw.splitlines() if line.strip())
    return ()


async def smoke_search(client: RAGFlowClient, dataset_id: str, question: str) -> list[dict[str, Any]]:
    """One retrieval call, used by ``verify`` to prove the index answers."""
    try:
        data = await client.retrieve(
            question=question,
            dataset_ids=[dataset_id],
            page_size=3,
            include_metadata=True,
        )
    except RAGFlowError as exc:
        raise IngestionError(f"smoke search failed: {exc}") from exc
    return list(data.get("chunks") or [])

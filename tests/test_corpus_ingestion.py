"""Indexing an external corpus, and the isolation that keeps it separate.

Isolation is the claim Stage 2B1 has to be able to defend: a question asked of the
benchmark must not be answered from the synthetic procedures, and vice versa. The
mechanism is one RAGFlow dataset per corpus, so the tests here are mostly about
proving that the mechanism holds in the directions it could fail - a write landing
in the wrong dataset, a delete reaching further than its corpus, a second corpus
overwriting the first because two documents share a filename.

The RAGFlow used here is a fake that keeps a separate document store per dataset
id, because that is the property under test. It refuses a page larger than 100 for
the same reason the real one does: a cap that only exists in production is a cap
that gets discovered in production.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath_ingestion.corpora.base import CorpusBundle, CorpusDocument, CorpusImage
from agentpath_ingestion.corpora.ingest import (
    DATASET_PREFIX,
    INGEST_MANIFEST_NAME,
    CorpusIngestionService,
    CorpusIngestReport,
    DocumentOutcome,
    dataset_name_for,
    write_ingest_manifest,
)
from agentpath_ingestion.ingest import SHA_FIELD, IngestionError

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "http://ragflow.test:9380"
API_KEY = "ragflow-test-key"

EMBEDDING_MODEL = "BAAI/bge-m3@LocalAI"

#: The procedures dataset, named the way the real pipeline names it. Present in
#: these tests only as the thing that must stay untouched.
PROCEDURES_DATASET = "agentpath-northwind-procedures"


# --------------------------------------------------------------------------- #
# A fake RAGFlow with real dataset boundaries
# --------------------------------------------------------------------------- #


class FakeRAGFlow:
    """Enough of RAGFlow to drive ``CorpusIngestionService`` end to end.

    Documents live in ``self.store[dataset_id]``, keyed by RAGFlow id, so a write
    that reaches the wrong dataset is visible rather than absorbed. Chunk counts
    are per document and served through a real pagination protocol.
    """

    #: Taken from the real client rather than restated, so the fake's refusal and
    #: the production loop can never disagree about where the cap is.
    MAX_CHUNK_PAGE_SIZE = RAGFlowClient.MAX_CHUNK_PAGE_SIZE

    def __init__(
        self,
        *,
        datasets: dict[str, str] | None = None,
        embedding_model: str = EMBEDDING_MODEL,
        chunks_per_document: int = 3,
        parse_status: str = "DONE",
    ) -> None:
        #: dataset name -> dataset id. Pre-seeded datasets already "exist".
        self.datasets: dict[str, str] = dict(datasets or {})
        self.store: dict[str, dict[str, dict[str, Any]]] = {dataset_id: {} for dataset_id in self.datasets.values()}
        self.embedding_model = embedding_model
        self.chunks_per_document = chunks_per_document
        self.parse_status = parse_status
        self.calls: list[str] = []
        self.created_datasets: list[dict[str, Any]] = []
        self._next_document = 0
        self._next_dataset = 0

    # -- helpers used by tests ----------------------------------------- #

    def bucket(self, dataset_name: str) -> dict[str, dict[str, Any]]:
        return self.store.get(self.datasets.get(dataset_name, ""), {})

    def names_in(self, dataset_name: str) -> list[str]:
        return sorted(str(doc["name"]) for doc in self.bucket(dataset_name).values())

    def seed(self, dataset_name: str, *, name: str, corpus_id: str) -> str:
        """Put a document in a dataset without going through ingestion."""
        dataset_id = self.datasets.setdefault(dataset_name, f"pre-{len(self.datasets)}")
        self.store.setdefault(dataset_id, {})
        self._next_document += 1
        document_id = f"seed-{self._next_document}"
        self.store[dataset_id][document_id] = {
            "id": document_id,
            "name": name,
            "run": "DONE",
            "meta_fields": {"corpus_id": corpus_id},
        }
        return document_id

    # -- datasets ------------------------------------------------------ #

    async def find_dataset_id(self, name: str) -> str | None:
        return self.datasets.get(name)

    async def list_datasets(self, *, name: str | None = None) -> list[dict[str, Any]]:
        return [
            {"id": dataset_id, "name": dataset_name, "embedding_model": self.embedding_model}
            for dataset_name, dataset_id in self.datasets.items()
            if name is None or dataset_name == name
        ]

    async def create_dataset(self, **fields: Any) -> str:
        self._next_dataset += 1
        dataset_id = f"ds-{self._next_dataset}"
        name = str(fields.get("name"))
        self.datasets[name] = dataset_id
        self.store[dataset_id] = {}
        self.created_datasets.append(dict(fields))
        self.calls.append(f"create-dataset:{name}")
        return dataset_id

    # -- documents ----------------------------------------------------- #

    async def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        return [dict(doc) for doc in self.store.get(dataset_id, {}).values()]

    async def upload_document(self, dataset_id: str, filename: str, content: bytes) -> str:
        self._next_document += 1
        document_id = f"doc-{self._next_document}"
        self.calls.append(f"upload:{dataset_id}:{filename}")
        self.store.setdefault(dataset_id, {})[document_id] = {
            "id": document_id,
            "name": filename,
            "run": "UNSTART",
            "meta_fields": {},
            "content": content,
        }
        return document_id

    async def update_document(self, dataset_id: str, document_id: str, **fields: Any) -> None:
        self.calls.append(f"update:{dataset_id}:{document_id}")
        self.store[dataset_id][document_id]["meta_fields"] = dict(fields.get("meta_fields") or {})
        if "parser_config" in fields:
            self.store[dataset_id][document_id]["parser_config"] = fields["parser_config"]

    async def delete_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        self.calls.append(f"delete:{dataset_id}:{','.join(sorted(document_ids))}")
        for document_id in document_ids:
            self.store.get(dataset_id, {}).pop(document_id, None)

    async def parse_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        self.calls.append(f"parse:{dataset_id}:{','.join(sorted(document_ids))}")
        for document_id in document_ids:
            document = self.store[dataset_id][document_id]
            document["run"] = self.parse_status
            document["chunk_count"] = self.chunks_per_document
            if self.parse_status == "FAIL":
                document["progress_msg"] = "embedder said no"

    # -- chunks -------------------------------------------------------- #

    async def list_chunks(
        self, dataset_id: str, document_id: str, *, page: int = 1, page_size: int = 100
    ) -> dict[str, Any]:
        if page_size > self.MAX_CHUNK_PAGE_SIZE:
            raise AssertionError(f"page_size must be less than or equal to {self.MAX_CHUNK_PAGE_SIZE}")
        total = int(self.store[dataset_id][document_id].get("chunk_count") or 0)
        start = (page - 1) * page_size
        rows = [{"id": f"{document_id}-c{index}"} for index in range(start, min(start + page_size, total))]
        self.calls.append(f"list-chunks:{document_id}:p{page}")
        return {"total": total, "chunks": rows}

    async def count_chunks(self, dataset_id: str, document_id: str) -> int:
        """Delegates to the real paginating implementation, not a shortcut.

        ``CorpusIngestionService.chunk_counts`` calls this, and the thing worth
        testing is the loop, so the loop is the thing that runs.
        """
        return await RAGFlowClient.count_chunks(self, dataset_id, document_id)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #


def _document(document_id: str, text: str = "First paragraph.\n\nSecond paragraph.") -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        title=f"Manual {document_id}",
        text=text,
        source_id=f"src-{document_id}",
    )


def _bundle(
    *documents: CorpusDocument,
    corpus_id: str = "mramg-manual",
    licence: str = "Apache-2.0",
) -> CorpusBundle:
    return CorpusBundle(
        corpus_id=corpus_id,
        title="MRAMG-Manual",
        attribution="MRAMG-Bench",
        licence=licence,
        documents=documents or (_document("50000"),),
    )


def _service(client: FakeRAGFlow) -> CorpusIngestionService:
    # Zero poll interval: the parse loop sleeps before its first read, and these
    # tests have nothing to wait for.
    return CorpusIngestionService(
        client,  # type: ignore[arg-type]
        embedding_model=EMBEDDING_MODEL,
        parse_timeout=5.0,
        poll_interval=0.0,
    )


# --------------------------------------------------------------------------- #
# One dataset per corpus
# --------------------------------------------------------------------------- #


def test_every_corpus_gets_its_own_prefixed_dataset_name() -> None:
    assert dataset_name_for("mramg-manual") == "agentpath-mramg-manual"
    assert dataset_name_for("northwind-procedures") == PROCEDURES_DATASET
    assert dataset_name_for("mramg-manual") != dataset_name_for("northwind-procedures")


def test_the_prefix_is_applied_so_a_corpus_id_cannot_collide_with_a_foreign_dataset() -> None:
    """A corpus called ``x`` must not be able to claim a dataset called ``x``."""
    assert dataset_name_for("x").startswith(DATASET_PREFIX)
    assert dataset_name_for("x") != "x"


async def test_ingestion_creates_the_corpus_dataset_when_absent() -> None:
    client = FakeRAGFlow()
    report = await _service(client).ingest(_bundle())

    assert report.dataset_name == "agentpath-mramg-manual"
    assert client.created_datasets[0]["name"] == "agentpath-mramg-manual"
    assert client.created_datasets[0]["embedding_model"] == EMBEDDING_MODEL
    assert report.ok


async def test_ingestion_reuses_an_existing_dataset_rather_than_making_a_second() -> None:
    client = FakeRAGFlow(datasets={"agentpath-mramg-manual": "ds-existing"})
    report = await _service(client).ingest(_bundle())

    assert report.dataset_id == "ds-existing"
    assert not client.created_datasets


async def test_a_dataset_built_with_another_embedding_model_is_refused() -> None:
    """Inherited from the procedures pipeline, and worth asserting here too.

    Querying a dataset whose documents were embedded by a different model returns
    plausible-looking nonsense, so it has to fail loudly.
    """
    client = FakeRAGFlow(
        datasets={"agentpath-mramg-manual": "ds-old"},
        embedding_model="text-embedding-3-small@OpenAI",
    )

    with pytest.raises(IngestionError, match="embedding model"):
        await _service(client).ingest(_bundle())


# --------------------------------------------------------------------------- #
# Isolation: writes stay inside their corpus
# --------------------------------------------------------------------------- #


async def test_the_procedures_dataset_is_untouched_by_a_corpus_ingest() -> None:
    client = FakeRAGFlow(datasets={PROCEDURES_DATASET: "ds-procedures"})
    client.seed(PROCEDURES_DATASET, name="NW-HTL-014.md", corpus_id="northwind-procedures")
    client.seed(PROCEDURES_DATASET, name="NW-REF-018.md", corpus_id="northwind-procedures")

    await _service(client).ingest(_bundle(_document("50000"), _document("50001")))

    assert client.names_in(PROCEDURES_DATASET) == ["NW-HTL-014.md", "NW-REF-018.md"]
    assert client.names_in("agentpath-mramg-manual") == [
        "mramg-manual-50000.md",
        "mramg-manual-50001.md",
    ]
    assert not any(call.startswith("delete:ds-procedures") for call in client.calls)


async def test_two_corpora_land_in_two_datasets_without_mixing() -> None:
    client = FakeRAGFlow()
    service = _service(client)

    await service.ingest(_bundle(_document("50000"), corpus_id="mramg-manual"))
    await service.ingest(_bundle(_document("50000"), corpus_id="other-corpus"))

    assert client.names_in("agentpath-mramg-manual") == ["mramg-manual-50000.md"]
    assert client.names_in("agentpath-other-corpus") == ["other-corpus-50000.md"]
    assert client.datasets["agentpath-mramg-manual"] != client.datasets["agentpath-other-corpus"]


async def test_a_shared_document_id_across_corpora_does_not_overwrite() -> None:
    """MRAMG numbers documents from 50000; nothing stops another corpus doing so.

    The filename carries the corpus id for exactly this reason, and the datasets
    are separate anyway, so the same id in two corpora is two documents.
    """
    client = FakeRAGFlow()
    service = _service(client)

    first = await service.ingest(_bundle(_document("50000", text="alpha"), corpus_id="mramg-manual"))
    second = await service.ingest(_bundle(_document("50000", text="beta"), corpus_id="other-corpus"))

    assert first.document_ids["50000"] != second.document_ids["50000"]
    assert first.outcomes[0].filename != second.outcomes[0].filename


async def test_every_upload_carries_its_corpus_id_in_metadata() -> None:
    """Belt to the dataset's braces: a stray result would be visibly labelled."""
    client = FakeRAGFlow()
    bundle = _bundle(_document("50000"))
    await _service(client).ingest(bundle)

    meta = next(iter(client.bucket("agentpath-mramg-manual").values()))["meta_fields"]
    assert meta["corpus_id"] == "mramg-manual"
    assert meta["document_id"] == "50000"
    assert meta["source_id"] == "src-50000"
    assert meta["title"] == "Manual 50000"
    assert meta["licence"] == "Apache-2.0"
    assert meta[SHA_FIELD]


async def test_a_document_without_a_source_id_records_an_empty_string_not_none() -> None:
    """RAGFlow metadata is a flat string map; ``None`` would arrive as "None"."""
    client = FakeRAGFlow()
    document = CorpusDocument(document_id="50000", title="T", text="body", source_id=None)
    await _service(client).ingest(_bundle(document))

    meta = next(iter(client.bucket("agentpath-mramg-manual").values()))["meta_fields"]
    assert meta["source_id"] == ""


# --------------------------------------------------------------------------- #
# Idempotence
# --------------------------------------------------------------------------- #


async def test_a_second_run_over_unchanged_text_parses_nothing() -> None:
    """The expensive half is parsing; an unchanged corpus must not pay it twice."""
    client = FakeRAGFlow()
    service = _service(client)
    bundle = _bundle(_document("50000"), _document("50001"))

    first = await service.ingest(bundle)
    client.calls.clear()
    second = await service.ingest(bundle)

    assert first.actions == {"created": 2}
    assert second.actions == {"unchanged": 2}
    assert not any(call.startswith("parse:") for call in client.calls)
    assert not any(call.startswith("upload:") for call in client.calls)


async def test_a_second_run_keeps_the_same_ragflow_document_ids() -> None:
    """An unchanged document must not be re-created; its id is cited elsewhere."""
    client = FakeRAGFlow()
    service = _service(client)
    bundle = _bundle(_document("50000"))

    first = await service.ingest(bundle)
    second = await service.ingest(bundle)

    assert first.document_ids == second.document_ids


async def test_changed_text_replaces_the_document_and_reparses() -> None:
    client = FakeRAGFlow()
    service = _service(client)

    await service.ingest(_bundle(_document("50000", text="original")))
    client.calls.clear()
    report = await service.ingest(_bundle(_document("50000", text="revised")))

    assert report.actions == {"updated": 1}
    assert any(call.startswith("delete:") for call in client.calls)
    assert any(call.startswith("upload:") for call in client.calls)
    assert any(call.startswith("parse:") for call in client.calls)
    assert client.names_in("agentpath-mramg-manual") == ["mramg-manual-50000.md"]


async def test_force_replaces_documents_whose_text_did_not_change() -> None:
    client = FakeRAGFlow()
    service = _service(client)
    bundle = _bundle(_document("50000"))

    await service.ingest(bundle)
    report = await service.ingest(bundle, force=True)

    assert report.actions == {"updated": 1}


async def test_a_document_dropped_from_the_bundle_is_removed_from_the_dataset() -> None:
    """A corpus must be rebuildable, not merely extendable."""
    client = FakeRAGFlow()
    service = _service(client)

    await service.ingest(_bundle(_document("50000"), _document("50001")))
    report = await service.ingest(_bundle(_document("50000")))

    assert len(report.removed) == 1
    assert client.names_in("agentpath-mramg-manual") == ["mramg-manual-50000.md"]


async def test_removing_a_stale_document_does_not_reach_the_other_dataset() -> None:
    client = FakeRAGFlow(datasets={PROCEDURES_DATASET: "ds-procedures"})
    client.seed(PROCEDURES_DATASET, name="NW-HTL-014.md", corpus_id="northwind-procedures")
    service = _service(client)

    await service.ingest(_bundle(_document("50000"), _document("50001")))
    await service.ingest(_bundle(_document("50000")))

    assert client.names_in(PROCEDURES_DATASET) == ["NW-HTL-014.md"]


# --------------------------------------------------------------------------- #
# Parse failures are reported, not swallowed
# --------------------------------------------------------------------------- #


async def test_a_parse_failure_is_recorded_and_the_report_is_not_ok() -> None:
    """The first live run failed this way - the embedder rejected a batch - and
    reported three chunks as if it had worked. The failure has to surface."""
    client = FakeRAGFlow(parse_status="FAIL")
    report = await _service(client).ingest(_bundle())

    assert not report.ok
    assert report.problems
    assert "mramg-manual-50000.md" in report.problems[0]


async def test_a_parse_timeout_is_reported_rather_than_hanging() -> None:
    client = FakeRAGFlow(parse_status="RUNNING")
    service = CorpusIngestionService(
        client,  # type: ignore[arg-type]
        embedding_model=EMBEDDING_MODEL,
        parse_timeout=0.05,
        poll_interval=0.01,
    )
    report = await service.ingest(_bundle())

    assert not report.ok
    assert "did not finish" in report.problems[0]


# --------------------------------------------------------------------------- #
# Chunk counts are read back from the engine
# --------------------------------------------------------------------------- #


async def test_chunk_counts_come_from_ragflow_not_from_our_own_chunking() -> None:
    client = FakeRAGFlow(chunks_per_document=7)
    service = _service(client)
    report = await service.ingest(_bundle(_document("50000"), _document("50001")))

    counts = await service.chunk_counts(report)

    assert counts == {"50000": 7, "50001": 7}


async def test_chunk_counts_paginate_past_the_hundred_row_cap() -> None:
    """A document with more than 100 chunks was one busy manual away from real.

    The largest MRAMG manual produced 97, so a single page would have undercounted
    the next one. The fake refuses a larger page exactly as RAGFlow does, so asking
    for one would fail rather than silently return 100.
    """
    client = FakeRAGFlow(chunks_per_document=243)
    service = _service(client)
    report = await service.ingest(_bundle(_document("50000")))

    counts = await service.chunk_counts(report)

    assert counts == {"50000": 243}
    pages = [call for call in client.calls if call.startswith("list-chunks:")]
    assert pages == ["list-chunks:doc-1:p1", "list-chunks:doc-1:p2", "list-chunks:doc-1:p3"]


async def test_a_document_whose_count_is_an_exact_multiple_reads_one_empty_page() -> None:
    """The loop stops on a short page, so a full last page needs one more read."""
    client = FakeRAGFlow(chunks_per_document=200)
    service = _service(client)
    report = await service.ingest(_bundle(_document("50000")))

    assert await service.chunk_counts(report) == {"50000": 200}
    assert len([call for call in client.calls if call.startswith("list-chunks:")]) == 3


@respx.mock
async def test_the_real_client_paginates_with_the_documented_page_size() -> None:
    """Against the wire, not the fake: the cap is the server's, so the request
    that goes out must respect it."""
    url = f"{BASE_URL}/api/v1/datasets/ds/documents/doc/chunks"
    seen: list[dict[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        page = int(request.url.params["page"])
        rows = 100 if page == 1 else 12
        return httpx.Response(200, json={"code": 0, "data": {"chunks": [{"id": f"c{page}-{i}"} for i in range(rows)]}})

    respx.get(url).mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        total = await client.count_chunks("ds", "doc")

    assert total == 112
    assert [item["page"] for item in seen] == ["1", "2"]
    assert {item["page_size"] for item in seen} == {"100"}
    assert all(int(item["page_size"]) <= RAGFlowClient.MAX_CHUNK_PAGE_SIZE for item in seen)


# --------------------------------------------------------------------------- #
# Deleting one corpus
# --------------------------------------------------------------------------- #


async def test_dropping_a_corpus_empties_only_its_own_dataset() -> None:
    client = FakeRAGFlow(datasets={PROCEDURES_DATASET: "ds-procedures"})
    client.seed(PROCEDURES_DATASET, name="NW-HTL-014.md", corpus_id="northwind-procedures")
    service = _service(client)
    await service.ingest(_bundle(_document("50000"), _document("50001")))

    removed = await service.delete_corpus("mramg-manual")

    assert removed == 2
    assert client.names_in("agentpath-mramg-manual") == []
    assert client.names_in(PROCEDURES_DATASET) == ["NW-HTL-014.md"]


async def test_dropping_an_unknown_corpus_deletes_nothing() -> None:
    client = FakeRAGFlow(datasets={PROCEDURES_DATASET: "ds-procedures"})
    client.seed(PROCEDURES_DATASET, name="NW-HTL-014.md", corpus_id="northwind-procedures")

    removed = await _service(client).delete_corpus("does-not-exist")

    assert removed == 0
    assert client.names_in(PROCEDURES_DATASET) == ["NW-HTL-014.md"]
    assert not any(call.startswith("delete:") for call in client.calls)


async def test_dropping_an_empty_corpus_issues_no_delete_call() -> None:
    client = FakeRAGFlow(datasets={"agentpath-mramg-manual": "ds-empty"})

    assert await _service(client).delete_corpus("mramg-manual") == 0
    assert not any(call.startswith("delete:") for call in client.calls)


async def test_a_corpus_can_be_dropped_and_rebuilt() -> None:
    client = FakeRAGFlow()
    service = _service(client)
    bundle = _bundle(_document("50000"))

    await service.ingest(bundle)
    await service.delete_corpus("mramg-manual")
    report = await service.ingest(bundle)

    assert report.actions == {"created": 1}
    assert client.names_in("agentpath-mramg-manual") == ["mramg-manual-50000.md"]


# --------------------------------------------------------------------------- #
# The run record
# --------------------------------------------------------------------------- #


def _report_with(chunk_counts: dict[str, int] | None = None) -> CorpusIngestReport:
    return CorpusIngestReport(
        corpus_id="mramg-manual",
        dataset_name="agentpath-mramg-manual",
        dataset_id="ds-1",
        embedding_model=EMBEDDING_MODEL,
        outcomes=[
            DocumentOutcome(
                corpus_document_id="50000",
                filename="mramg-manual-50000.md",
                ragflow_document_id="doc-1",
                action="created",
                byte_size=1234,
            )
        ],
        seconds=12.5,
    )


async def test_the_manifest_records_what_ragflow_actually_indexed(tmp_path: Path) -> None:
    path = write_ingest_manifest(_report_with(), tmp_path, {"50000": 97})

    assert path.name == INGEST_MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["corpus_id"] == "mramg-manual"
    assert payload["dataset_name"] == "agentpath-mramg-manual"
    assert payload["actions"] == {"created": 1}
    assert payload["documents"][0]["ragflow_chunks"] == 97
    assert payload["documents"][0]["byte_size"] == 1234
    assert payload["problems"] == []


async def test_the_manifest_reports_an_absent_chunk_count_as_null(tmp_path: Path) -> None:
    """Better an explicit null than a zero that reads as "RAGFlow indexed nothing"."""
    payload = json.loads(write_ingest_manifest(_report_with(), tmp_path).read_text(encoding="utf-8"))

    assert payload["documents"][0]["ragflow_chunks"] is None


async def test_the_manifest_describes_no_machine(tmp_path: Path) -> None:
    text = write_ingest_manifest(_report_with(), tmp_path, {"50000": 3}).read_text(encoding="utf-8")

    assert str(REPO_ROOT) not in text
    assert "/home/" not in text


def test_the_report_counts_actions_and_maps_document_ids() -> None:
    report = CorpusIngestReport(corpus_id="c", dataset_name="d")
    report.outcomes = [
        DocumentOutcome("1", "c-1.md", "doc-1", "created", 10),
        DocumentOutcome("2", "c-2.md", "doc-2", "unchanged", 20),
        DocumentOutcome("3", "c-3.md", "doc-3", "unchanged", 30),
    ]

    assert report.actions == {"created": 1, "unchanged": 2}
    assert report.document_ids == {"1": "doc-1", "2": "doc-2", "3": "doc-3"}
    assert report.ok


# --------------------------------------------------------------------------- #
# What is uploaded is the corpus's own text
# --------------------------------------------------------------------------- #


async def test_the_uploaded_bytes_are_the_document_text_verbatim() -> None:
    """Placeholders included. Offsets are recovered against this text later, so a
    normalisation here would silently invalidate every citation."""
    client = FakeRAGFlow()
    text = "Step one.\n\n<PIC>\n\nStep two.  Trailing   spaces   kept.\n"
    await _service(client).ingest(_bundle(_document("50000", text=text)))

    stored = next(iter(client.bucket("agentpath-mramg-manual").values()))["content"]
    assert stored == text.encode("utf-8")
    assert b"<PIC>" in stored


async def test_the_filename_is_stable_across_runs() -> None:
    """RAGFlow keys documents by name within a dataset, so change detection and
    stale-document removal both depend on the name not drifting."""
    client = FakeRAGFlow()
    service = _service(client)

    first = await service.ingest(_bundle(_document("50000", text="a")))
    second = await service.ingest(_bundle(_document("50000", text="b")))

    assert first.outcomes[0].filename == second.outcomes[0].filename == "mramg-manual-50000.md"


async def test_a_document_carrying_images_is_still_uploaded_as_text_only() -> None:
    """The image bytes are served by the API from disk, not stored in RAGFlow: the
    index holds text, which is what ``docs/RETRIEVAL_MODES.md`` says it holds."""
    client = FakeRAGFlow()
    document = CorpusDocument(
        document_id="50000",
        title="With figures",
        text="before <PIC> after",
        images=(
            CorpusImage(
                image_id="Manual00_1",
                document_id="50000",
                position_index=0,
                char_offset=7,
                local_path="data/external/mramg/raw/images/Manual00_1.jpg",
            ),
        ),
        source_id="src-50000",
    )
    await _service(client).ingest(_bundle(document))

    stored = next(iter(client.bucket("agentpath-mramg-manual").values()))
    assert stored["content"] == b"before <PIC> after"
    assert "Manual00_1" not in json.dumps(stored["meta_fields"])

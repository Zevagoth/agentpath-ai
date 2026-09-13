"""Tests for the retrieval layer: client, mapping, aggregation, evidence.

All of it runs against recorded RAGFlow payloads, so the suite needs no running
stack and stays fast. The payload shapes below are taken from the real responses
observed against RAGFlow v0.27.2 rather than invented, including the quirks:
business errors arrive inside HTTP 200, and a missing dataset is reported as a
permission error.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentpath.domain import (
    EvidenceLevel,
    Language,
    Passage,
    PreparedQuery,
    RetrievalMode,
)
from agentpath.evidence import ABSTENTION_MESSAGE, detect_scenario_overlap, grade_evidence
from agentpath.retrieval.aggregate import aggregate_passages
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable, Retriever
from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError, RAGFlowUnavailable
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever

BASE_URL = "http://ragflow.test:9380"
API_KEY = "ragflow-test-key"

DATASET_LIST = {
    "code": 0,
    "data": [
        {
            "id": "dataset-1",
            "name": "agentpath-northwind-procedures",
            "embedding_model": "BAAI/bge-m3@agentpath-embedder@HuggingFace",
        }
    ],
}


def _chunk(
    *,
    chunk_id: str,
    procedure_id: str,
    similarity: float,
    term: float = 0.3,
    vector: float = 0.8,
    content: str = "When the property confirms it cannot host a confirmed guest, secure alternatives.",
    version: str = "2.1",
    document: str | None = None,
) -> dict:
    """One chunk in RAGFlow's response shape."""
    return {
        "id": chunk_id,
        "content": content,
        "document_id": f"doc-{procedure_id}",
        "document_keyword": document or f"{procedure_id}.md",
        "dataset_id": "dataset-1",
        "similarity": similarity,
        "term_similarity": term,
        "vector_similarity": vector,
        "highlight": "the property <em>cannot host</em> a confirmed guest",
        "image_id": "",
        "document_metadata": {
            "procedure_id": procedure_id,
            "product_line": "hotels",
            "version": version,
            "effective_date": "2026-03-01",
            "scenario_tags": "overbooking,walked_guest",
            "title": "Property unable to accommodate a confirmed guest",
        },
    }


def _prepared(text: str = "hotel has no room") -> PreparedQuery:
    return PreparedQuery(
        original=text,
        normalized=text,
        language=Language.ENGLISH,
        search_text=text,
        scenario_tags=("overbooking",),
    )


def _retriever(client: RAGFlowClient, **kwargs) -> RAGFlowRetriever:
    return RAGFlowRetriever(
        client,
        dataset_name="agentpath-northwind-procedures",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Client transport
# --------------------------------------------------------------------------- #


@respx.mock
async def test_client_unwraps_the_business_envelope() -> None:
    """Most endpoints wrap their payload in {code, data}."""
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        datasets = await client.list_datasets()
    assert datasets[0]["id"] == "dataset-1"


@respx.mock
async def test_healthz_is_read_without_the_envelope() -> None:
    """healthz answers with the component map directly, not wrapped."""
    respx.get(f"{BASE_URL}/api/v1/system/healthz").mock(
        return_value=httpx.Response(
            200, json={"db": "ok", "doc_engine": "ok", "redis": "ok", "status": "ok", "storage": "ok"}
        )
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert (await client.healthz())["doc_engine"] == "ok"


@respx.mock
async def test_health_names_degraded_components() -> None:
    respx.get(f"{BASE_URL}/api/v1/system/healthz").mock(
        return_value=httpx.Response(200, json={"db": "ok", "doc_engine": "down", "status": "ok"})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        health = await _retriever(client).health()
    assert health.reachable is True
    assert "doc_engine" in health.detail


@respx.mock
async def test_client_raises_on_business_error_inside_http_200() -> None:
    """RAGFlow reports failures with HTTP 200 and a non-zero code."""
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(200, json={"code": 100, "message": "embedder unreachable"})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowError, match="embedder unreachable"):
            await client.retrieve(question="q", dataset_ids=["dataset-1"])


@respx.mock
async def test_client_reports_unreachable_service_distinctly() -> None:
    respx.get(f"{BASE_URL}/api/v1/system/healthz").mock(side_effect=httpx.ConnectError("refused"))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowUnavailable):
            await client.healthz()


@respx.mock
async def test_client_rejects_a_bad_api_key_with_a_clear_message() -> None:
    respx.get(f"{BASE_URL}/api/v1/system/healthz").mock(return_value=httpx.Response(401))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowError, match="API key"):
            await client.healthz()


@respx.mock
async def test_dataset_lookup_treats_permission_code_as_a_miss() -> None:
    """RAGFlow answers "no dataset by that name" with a permission error.

    Treating that as fatal made first-run ingestion impossible, so the miss must
    fall through to the unfiltered listing.
    """
    route = respx.get(f"{BASE_URL}/api/v1/datasets")
    route.side_effect = [
        httpx.Response(200, json={"code": 102, "message": "User 'x' lacks permission for dataset 'y'"}),
        httpx.Response(200, json=DATASET_LIST),
    ]
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.find_dataset_id("agentpath-northwind-procedures") == "dataset-1"


@respx.mock
async def test_dataset_lookup_returns_none_when_absent() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": []})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.find_dataset_id("missing") is None


@respx.mock
async def test_upload_sends_multipart_without_a_json_content_type() -> None:
    """A default JSON content type destroys the multipart boundary.

    RAGFlow then rejects the body with "No file part!", which is what happened
    before the client stopped setting Content-Type globally.
    """
    captured: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"code": 0, "data": [{"id": "doc-1"}]})

    respx.post(f"{BASE_URL}/api/v1/datasets/dataset-1/documents").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.upload_document("dataset-1", "NW-HTL-014.md", b"# doc") == "doc-1"
    assert captured["content_type"].startswith("multipart/form-data")


@respx.mock
async def test_delete_documents_is_a_noop_without_ids() -> None:
    """Guard: an empty list must not become "delete everything"."""
    route = respx.delete(f"{BASE_URL}/api/v1/datasets/dataset-1/documents")
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.delete_documents("dataset-1", [])
    assert not route.called


# --------------------------------------------------------------------------- #
# Retriever: request shape
# --------------------------------------------------------------------------- #


@respx.mock
async def test_retriever_sends_hybrid_parameters() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": [], "total": 0}})

    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        retriever = _retriever(client, vector_similarity_weight=0.6, similarity_threshold=0.15)
        await retriever.retrieve(_prepared(), limit=3)

    assert captured["vector_similarity_weight"] == 0.6
    assert captured["similarity_threshold"] == 0.15
    assert captured["include_metadata"] is True
    assert captured["highlight"] is True
    # The LLM keyword expansion flag must stay off: it costs a chat-model round
    # trip and our lexicon covers it deterministically.
    assert captured["keyword"] is False
    assert "rerank_id" not in captured


@respx.mock
async def test_retriever_requests_more_chunks_than_procedures() -> None:
    """Several chunks of one procedure rank together, so the page must be wider."""
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": []}})

    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await _retriever(client, page_size=12).retrieve(_prepared(), limit=3)
    assert captured["page_size"] >= 9


@respx.mock
async def test_retriever_passes_rerank_model_when_configured() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": []}})

    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        retriever = _retriever(client, rerank_model="BAAI/bge-reranker-v2-m3@inst@HuggingFace")
        await retriever.retrieve(_prepared(), limit=3)
    assert captured["rerank_id"] == "BAAI/bge-reranker-v2-m3@inst@HuggingFace"


@respx.mock
async def test_product_line_filter_becomes_a_metadata_condition() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": []}})

    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await _retriever(client).retrieve(_prepared(), limit=3, product_line="hotels")

    condition = captured["metadata_condition"]
    assert condition["logic"] == "and"
    assert condition["conditions"][0] == {
        "name": "product_line",
        "comparison_operator": "=",
        "value": "hotels",
    }


@respx.mock
async def test_no_filter_is_sent_without_an_explicit_product_line() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": []}})

    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await _retriever(client).retrieve(_prepared(), limit=3)
    assert "metadata_condition" not in captured


# --------------------------------------------------------------------------- #
# Retriever: response mapping
# --------------------------------------------------------------------------- #


@respx.mock
async def test_chunks_map_onto_passages_with_both_score_components() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "chunks": [_chunk(chunk_id="c1", procedure_id="NW-HTL-014", similarity=0.59)],
                    "total": 1,
                },
            },
        )
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await _retriever(client).retrieve(_prepared(), limit=3)

    assert result.mode is RetrievalMode.HYBRID
    passage = result.passages[0]
    assert passage.procedure_id == "NW-HTL-014"
    assert passage.title == "Property unable to accommodate a confirmed guest"
    assert passage.score == pytest.approx(0.59)
    assert passage.lexical_score == pytest.approx(0.3)
    assert passage.semantic_score == pytest.approx(0.8)
    assert passage.version == "2.1"
    assert passage.scenario_tags == ("overbooking", "walked_guest")
    assert "cannot host" in passage.snippet
    assert result.engine_ms >= 0


@respx.mock
async def test_procedure_id_falls_back_to_the_document_name() -> None:
    """Metadata can be absent; attribution must still be correct."""
    chunk = _chunk(chunk_id="c1", procedure_id="NW-REF-018", similarity=0.5)
    del chunk["document_metadata"]
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"chunks": [chunk]}})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await _retriever(client).retrieve(_prepared(), limit=3)
    assert result.passages[0].procedure_id == "NW-REF-018"


@respx.mock
async def test_empty_result_is_not_an_error() -> None:
    """Nothing found is a legitimate answer and must not raise."""
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"chunks": [], "total": 0}})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await _retriever(client).retrieve(_prepared("unrelated question"), limit=3)
    assert result.passages == ()


async def test_empty_query_skips_the_engine_entirely() -> None:
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        retriever = _retriever(client)
        prepared = PreparedQuery(
            original="", normalized="", language=Language.UNKNOWN, search_text=""
        )
        result = await retriever.retrieve(prepared, limit=3)
    assert result.passages == ()
    assert result.diagnostics["reason"] == "empty query"


@respx.mock
async def test_missing_dataset_produces_an_actionable_error() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": []})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RetrievalError, match="make ingest"):
            await _retriever(client).retrieve(_prepared(), limit=3)


@respx.mock
async def test_connection_failure_surfaces_as_unavailable() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(return_value=httpx.Response(200, json=DATASET_LIST))
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=httpx.ConnectError("refused"))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RetrievalUnavailable):
            await _retriever(client).retrieve(_prepared(), limit=3)


@respx.mock
async def test_health_reports_reachable_on_a_401() -> None:
    """A rejected key still proves the service is answering."""
    respx.get(f"{BASE_URL}/api/v1/system/healthz").mock(return_value=httpx.Response(401))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        health = await _retriever(client).health()
    assert health.reachable is True


def test_ragflow_retriever_satisfies_the_port() -> None:
    client = RAGFlowClient(BASE_URL, API_KEY)
    assert isinstance(_retriever(client), Retriever)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _passage(procedure_id: str, score: float, chunk_id: str = "c") -> Passage:
    return Passage(
        chunk_id=chunk_id,
        document_id=f"doc-{procedure_id}",
        procedure_id=procedure_id,
        title=f"Title {procedure_id}",
        content="content",
        score=score,
        product_line="hotels",
        version="1.0",
        effective_date="2026-01-01",
        scenario_tags=("overbooking",),
    )


def test_passages_group_into_procedures_ranked_by_best_passage() -> None:
    passages = [
        _passage("NW-HTL-014", 0.61, "a"),
        _passage("NW-HTL-021", 0.55, "b"),
        _passage("NW-HTL-014", 0.42, "c"),
    ]
    hits = aggregate_passages(passages, limit=3)

    assert [hit.procedure_id for hit in hits] == ["NW-HTL-014", "NW-HTL-021"]
    assert hits[0].rank == 1
    assert hits[0].score == pytest.approx(0.61, abs=1e-4)
    assert hits[0].matched_passage_count == 2
    # Passages inside a hit are best-first, so best_passage is the snippet to show.
    assert hits[0].best_passage.chunk_id == "a"


def test_procedure_score_is_the_best_passage_not_a_sum() -> None:
    """Three weak passages must not outrank one strong one."""
    passages = [
        _passage("NW-CXL-032", 0.2, "a"),
        _passage("NW-CXL-032", 0.2, "b"),
        _passage("NW-CXL-032", 0.2, "c"),
        _passage("NW-HTL-014", 0.5, "d"),
    ]
    hits = aggregate_passages(passages, limit=3)
    assert hits[0].procedure_id == "NW-HTL-014"


def test_more_matching_sections_breaks_an_exact_tie() -> None:
    passages = [
        _passage("NW-HTL-021", 0.5, "a"),
        _passage("NW-HTL-014", 0.5, "b"),
        _passage("NW-HTL-014", 0.4, "c"),
    ]
    hits = aggregate_passages(passages, limit=3)
    assert hits[0].procedure_id == "NW-HTL-014"


def test_aggregation_respects_the_limit() -> None:
    passages = [_passage(f"NW-TST-{index:03d}", 0.9 - index / 100, str(index)) for index in range(6)]
    assert len(aggregate_passages(passages, limit=3)) == 3


def test_aggregating_nothing_yields_nothing() -> None:
    assert aggregate_passages([], limit=3) == ()


def test_ranking_is_stable_for_equal_scores() -> None:
    """Reproducible benchmark numbers require a deterministic tie-break."""
    first = aggregate_passages([_passage("NW-B", 0.5, "x"), _passage("NW-A", 0.5, "y")], limit=3)
    second = aggregate_passages([_passage("NW-A", 0.5, "y"), _passage("NW-B", 0.5, "x")], limit=3)
    assert [h.procedure_id for h in first] == [h.procedure_id for h in second]


# --------------------------------------------------------------------------- #
# Evidence grading and abstention
# --------------------------------------------------------------------------- #

THRESHOLDS = {"strong_score": 0.45, "strong_margin": 0.05, "partial_score": 0.28}


def test_no_hits_means_abstain() -> None:
    level, reason = grade_evidence([], **THRESHOLDS)
    assert level is EvidenceLevel.INSUFFICIENT
    assert "similarity threshold" in reason


def test_clear_winner_is_strong() -> None:
    hits = aggregate_passages([_passage("NW-HTL-014", 0.62, "a"), _passage("NW-HTL-021", 0.31, "b")], limit=3)
    level, reason = grade_evidence(hits, **THRESHOLDS)
    assert level is EvidenceLevel.STRONG
    assert "NW-HTL-014" in reason


def test_single_high_scoring_hit_is_strong() -> None:
    hits = aggregate_passages([_passage("NW-HTL-014", 0.62, "a")], limit=3)
    level, reason = grade_evidence(hits, **THRESHOLDS)
    assert level is EvidenceLevel.STRONG
    assert "only match" in reason


def test_two_close_high_scores_are_only_partial() -> None:
    """An undecided engine must not present one procedure as the answer."""
    hits = aggregate_passages([_passage("NW-HTL-014", 0.50, "a"), _passage("NW-HTL-021", 0.49, "b")], limit=3)
    level, reason = grade_evidence(hits, **THRESHOLDS)
    assert level is EvidenceLevel.PARTIAL
    assert "Verify" in reason


def test_mid_score_is_partial() -> None:
    hits = aggregate_passages([_passage("NW-REF-018", 0.33, "a")], limit=3)
    level, _ = grade_evidence(hits, **THRESHOLDS)
    assert level is EvidenceLevel.PARTIAL


def test_weak_top_score_abstains() -> None:
    hits = aggregate_passages([_passage("NW-REF-018", 0.2, "a")], limit=3)
    level, reason = grade_evidence(hits, **THRESHOLDS)
    assert level is EvidenceLevel.INSUFFICIENT
    assert "under the minimum" in reason


def test_abstention_message_is_the_exact_specified_wording() -> None:
    assert ABSTENTION_MESSAGE == "No confirmed procedure was found in the available knowledge base."


def test_overlapping_scenarios_are_flagged_with_versions() -> None:
    hits = aggregate_passages([_passage("NW-HTL-014", 0.6, "a"), _passage("NW-HTL-021", 0.55, "b")], limit=3)
    notices = detect_scenario_overlap(hits, minimum_score=0.28)
    assert len(notices) == 1
    assert "NW-HTL-014" in notices[0] and "NW-HTL-021" in notices[0]
    assert "overbooking" in notices[0]
    assert "v1.0" in notices[0]


def test_weak_hits_are_not_flagged_as_conflicts() -> None:
    hits = aggregate_passages([_passage("NW-HTL-014", 0.2, "a"), _passage("NW-HTL-021", 0.19, "b")], limit=3)
    assert detect_scenario_overlap(hits, minimum_score=0.28) == ()

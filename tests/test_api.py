"""Tests for the HTTP layer.

A fake retriever stands in for RAGFlow, so these tests cover the contract the web
app depends on - shapes, status codes, and the difference between "no procedure"
and "engine down" - without a running stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agentpath.api.app import create_app
from agentpath.api.deps import ServiceContext
from agentpath.config import Settings
from agentpath.domain import (
    DependencyHealth,
    Passage,
    PreparedQuery,
    RetrievalMode,
    RetrievedPassages,
)
from agentpath.evidence import ABSTENTION_MESSAGE
from agentpath.query.lexicon import load_lexicon
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable
from agentpath.search_service import SearchService


@dataclass
class FakeRetriever:
    """A retriever whose output the test decides."""

    passages: tuple[Passage, ...] = ()
    error: Exception | None = None
    healthy: bool = True
    calls: list[dict] = field(default_factory=list)

    @property
    def mode(self) -> RetrievalMode:
        return RetrievalMode.HYBRID

    async def retrieve(
        self,
        query: PreparedQuery,
        *,
        limit: int,
        product_line: str | None = None,
        procedure_id: str | None = None,
    ) -> RetrievedPassages:
        self.calls.append(
            {
                "search_text": query.search_text,
                "limit": limit,
                "product_line": product_line,
                "procedure_id": procedure_id,
            }
        )
        if self.error:
            raise self.error
        return RetrievedPassages(
            passages=self.passages,
            mode=self.mode,
            engine_ms=12.3,
            diagnostics={"dataset_id": "dataset-1"},
        )

    async def health(self) -> DependencyHealth:
        return DependencyHealth(
            name="ragflow",
            reachable=self.healthy,
            detail="fake" if self.healthy else "unreachable",
            latency_ms=1.0,
        )


def _passage(procedure_id: str, score: float, chunk: str = "c1") -> Passage:
    return Passage(
        chunk_id=chunk,
        document_id=f"doc-{procedure_id}",
        procedure_id=procedure_id,
        title="Property unable to accommodate a confirmed guest",
        content="Secure equivalent-or-better alternative accommodation.",
        score=score,
        lexical_score=0.29,
        semantic_score=0.79,
        highlight="secure <em>equivalent</em> accommodation",
        product_line="hotels",
        version="2.1",
        effective_date="2026-03-01",
        scenario_tags=("overbooking",),
    )


def _settings() -> Settings:
    return Settings(
        RAGFLOW_API_KEY="test-key",
        AGENTPATH_LOG_LEVEL="WARNING",
        RAGFLOW_RERANK_MODEL="",
    )


def _context(retriever: FakeRetriever, settings: Settings | None = None) -> ServiceContext:
    settings = settings or _settings()
    lexicon = load_lexicon()

    class _NoopClient:
        async def aclose(self) -> None:
            return None

    return ServiceContext(
        settings=settings,
        lexicon=lexicon,
        client=_NoopClient(),  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        search=SearchService(retriever=retriever, lexicon=lexicon, settings=settings),  # type: ignore[arg-type]
    )


async def _client(retriever: FakeRetriever, settings: Settings | None = None):
    settings = settings or _settings()
    app = create_app(settings=settings, context=_context(retriever, settings))
    manager = LifespanManager(app)
    await manager.__aenter__()
    transport = httpx.ASGITransport(app=app)
    return (
        httpx.AsyncClient(transport=transport, base_url="http://test"),
        manager,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_search_returns_ranked_procedures_with_citations() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-HTL-014", 0.62), _passage("NW-HTL-021", 0.31, "c2")))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "hotel has no room"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    body = response.json()

    assert body["mode"] == "hybrid"
    assert [p["procedure_id"] for p in body["procedures"]] == ["NW-HTL-014", "NW-HTL-021"]
    top = body["procedures"][0]
    assert top["rank"] == 1
    assert top["version"] == "2.1"
    assert top["citations"], "a recommendation without citations is not shippable"
    assert top["citations"][0]["lexical_score"] == pytest.approx(0.29)
    assert top["citations"][0]["semantic_score"] == pytest.approx(0.79)
    assert body["evidence"]["level"] == "strong"
    assert body["evidence"]["abstained"] is False
    assert body["evidence"]["message"] is None
    assert set(body["timings_ms"]) >= {"understanding", "retrieval", "aggregation", "total"}


async def test_query_understanding_is_reported_back() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-HTL-014", 0.62),))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "overboking at the hotle"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    understanding = response.json()["query"]
    corrections = {(c["before"], c["after"]) for c in understanding["corrections"]}
    assert ("overboking", "overbooking") in corrections
    assert ("hotle", "hotel") in corrections
    assert "overbooking" in understanding["normalized"]
    assert understanding["language"] == "en"
    assert understanding["detected_product_line"] == "hotels"
    assert "overbooking" in understanding["scenario_tags"]


async def test_arabic_query_is_expanded_before_retrieval() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-HTL-014", 0.62),))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "الفندق ما عنده غرف"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    body = response.json()
    assert body["query"]["language"] == "ar"
    assert body["query"]["expansions"], "Arabic queries need English equivalents"
    # The text actually sent to the engine keeps the agent's own words first.
    assert retriever.calls[0]["search_text"].startswith("الفندق ما عنده غرف")
    assert "overbooking" in retriever.calls[0]["search_text"]


async def test_product_line_filter_is_forwarded() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-REF-018", 0.5),))
    client, manager = await _client(retriever)
    try:
        await client.post("/api/search", json={"query": "refund late", "product_line": "refunds"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert retriever.calls[0]["product_line"] == "refunds"


async def test_limit_is_respected() -> None:
    retriever = FakeRetriever(passages=tuple(_passage(f"NW-TST-{i:03d}", 0.6 - i / 100, f"c{i}") for i in range(5)))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "anything", "limit": 2})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert len(response.json()["procedures"]) == 2


# --------------------------------------------------------------------------- #
# Abstention and failure modes
# --------------------------------------------------------------------------- #


async def test_no_results_abstains_with_the_exact_message() -> None:
    retriever = FakeRetriever(passages=())
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "how do I reset my password"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    body = response.json()
    assert response.status_code == 200, "abstaining is a valid answer, not an error"
    assert body["procedures"] == []
    assert body["evidence"]["level"] == "insufficient"
    assert body["evidence"]["abstained"] is True
    assert body["evidence"]["message"] == ABSTENTION_MESSAGE


async def test_weak_match_abstains_rather_than_recommending() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-HTL-014", 0.12),))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "vaguely related question"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.json()["evidence"]["abstained"] is True


async def test_engine_down_is_503_and_never_looks_like_no_results() -> None:
    """The distinction an agent must never be confused about."""
    retriever = FakeRetriever(error=RetrievalUnavailable("connection refused", engine="ragflow"))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "hotel has no room"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "retrieval_unavailable"
    assert "not the same as finding no procedure" in body["message"]
    assert ABSTENTION_MESSAGE not in str(body)


async def test_engine_rejection_is_502() -> None:
    retriever = FakeRetriever(error=RetrievalError("bad request", engine="ragflow"))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "hotel has no room"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 502
    assert response.json()["error"] == "retrieval_failed"


async def test_conflicting_procedures_are_surfaced() -> None:
    retriever = FakeRetriever(passages=(_passage("NW-HTL-014", 0.6), _passage("NW-HTL-021", 0.55, "c2")))
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json={"query": "hotel problem"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    conflicts = response.json()["conflicts"]
    assert conflicts and "NW-HTL-014" in conflicts[0]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 99},
        {"query": "x", "product_line": "trains"},
        {"query": "x", "unexpected": True},
    ],
)
async def test_invalid_requests_are_rejected(payload: dict) -> None:
    retriever = FakeRetriever()
    client, manager = await _client(retriever)
    try:
        response = await client.post("/api/search", json=payload)
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


async def test_health_reports_each_dependency() -> None:
    retriever = FakeRetriever()
    client, manager = await _client(retriever)
    try:
        response = await client.get("/health")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    body = response.json()
    names = {dep["name"] for dep in body["dependencies"]}
    assert "ragflow" in names
    assert "embedder" in names
    assert body["dataset"]
    assert body["mode"] == "hybrid"


async def test_health_is_503_when_ragflow_is_unreachable() -> None:
    retriever = FakeRetriever(healthy=False)
    client, manager = await _client(retriever)
    try:
        response = await client.get("/health")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


async def test_openapi_documents_the_search_contract() -> None:
    retriever = FakeRetriever()
    client, manager = await _client(retriever)
    try:
        response = await client.get("/openapi.json")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    schema = response.json()
    assert "/api/search" in schema["paths"]
    assert "/health" in schema["paths"]


async def test_request_id_is_echoed_for_log_correlation() -> None:
    retriever = FakeRetriever()
    client, manager = await _client(retriever)
    try:
        response = await client.get("/health", headers={"x-request-id": "abc123"})
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.headers["x-request-id"] == "abc123"

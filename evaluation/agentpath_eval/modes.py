"""The three retrieval modes the benchmark compares.

They are layered so the report attributes gains to the right component:

1. ``lexical_baseline`` - keyword overlap only. The reference point.
2. ``ragflow_hybrid``   - RAGFlow's hybrid retrieval on the **raw** query, no
   query understanding. Isolates what the engine contributes.
3. ``agentpath_full``   - the shipping path through ``POST /api/search``:
   query understanding, hybrid retrieval, aggregation, evidence grading. Measured
   over HTTP so the latency is the one an agent would actually experience.

Comparing 2 with 3 is the only way to say whether our own layer earns its place.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from agentpath.config import Settings
from agentpath.domain import EvidenceLevel, Language, PreparedQuery
from agentpath.evidence import grade_evidence
from agentpath.query.lexicon import load_lexicon
from agentpath.retrieval.aggregate import aggregate_passages
from agentpath.retrieval.ports import RetrievalError
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever
from agentpath.search_service import SearchService
from agentpath_eval.baseline import LexicalBaseline
from agentpath_eval.metrics import QueryOutcome
from agentpath_eval.queries import BenchmarkQuery


class Mode(Protocol):
    """One measurable retrieval path."""

    name: str
    description: str

    async def run(self, query: BenchmarkQuery, *, limit: int) -> QueryOutcome: ...

    async def aclose(self) -> None: ...


def _outcome(
    query: BenchmarkQuery,
    *,
    ranked: list[str],
    abstained: bool,
    evidence: str,
    latency_ms: float,
    top_score: float = 0.0,
    notes: str = "",
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query.id,
        query=query.query,
        type=query.type,
        language=query.language,
        expected=query.expected,
        supported=query.supported,
        ranked=tuple(ranked),
        abstained=abstained,
        evidence=evidence,
        latency_ms=latency_ms,
        top_score=top_score,
        notes=notes,
    )


class LexicalBaselineMode:
    """Mode 1: keyword overlap, in process, no engine."""

    name = "lexical_baseline"
    description = (
        "Keyword overlap over the procedure corpus. No embeddings, no synonyms, "
        "no typo correction. Benchmark reference only."
    )

    def __init__(self) -> None:
        self._baseline = LexicalBaseline()

    async def run(self, query: BenchmarkQuery, *, limit: int) -> QueryOutcome:
        started = time.perf_counter()
        ranked, abstained = self._baseline.search(query.query, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000
        return _outcome(
            query,
            ranked=[procedure_id for procedure_id, _ in ranked],
            abstained=abstained,
            evidence="insufficient" if abstained else "keyword_match",
            latency_ms=latency_ms,
            top_score=ranked[0][1] if ranked else 0.0,
        )

    async def aclose(self) -> None:
        return None


class RAGFlowHybridMode:
    """Mode 2: RAGFlow hybrid retrieval on the raw query.

    Query understanding is bypassed by handing the retriever a ``PreparedQuery``
    whose search text is the untouched input. Evidence grading still runs, so the
    abstention column is comparable across modes.
    """

    name = "ragflow_hybrid"
    description = "RAGFlow hybrid retrieval (BM25 + vector) on the raw query, without AgentPath query understanding."

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = RAGFlowClient(
            settings.ragflow_base_url,
            settings.ragflow_api_key.get_secret_value(),
            timeout=settings.ragflow_timeout_seconds,
        )
        self._retriever = RAGFlowRetriever(
            self._client,
            dataset_name=settings.ragflow_dataset_name,
            rerank_model=(settings.ragflow_rerank_model if settings.enable_rerank else None) or None,
            vector_similarity_weight=settings.vector_similarity_weight,
            similarity_threshold=settings.similarity_threshold,
            rerank_candidates=settings.rerank_candidates,
            page_size=settings.page_size,
        )

    async def run(self, query: BenchmarkQuery, *, limit: int) -> QueryOutcome:
        prepared = PreparedQuery(
            original=query.query,
            normalized=query.query,
            language=Language.UNKNOWN,
            search_text=query.query,
        )
        started = time.perf_counter()
        try:
            retrieved = await self._retriever.retrieve(prepared, limit=limit)
        except RetrievalError as exc:
            return _outcome(
                query,
                ranked=[],
                abstained=True,
                evidence="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                notes=str(exc),
            )
        hits = aggregate_passages(retrieved.passages, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000

        evidence, _ = grade_evidence(
            hits,
            strong_score=self._settings.evidence_strong_score,
            strong_margin=self._settings.evidence_strong_margin,
            partial_score=self._settings.evidence_partial_score,
        )
        return _outcome(
            query,
            ranked=[hit.procedure_id for hit in hits],
            abstained=evidence is EvidenceLevel.INSUFFICIENT,
            evidence=evidence.value,
            latency_ms=latency_ms,
            top_score=hits[0].score if hits else 0.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class AgentPathAPIMode:
    """Mode 3: the shipping path over HTTP.

    Measured through the API rather than in process, because the number that
    matters to an agent includes serialisation and the network hop.
    """

    name = "agentpath_full"
    description = "The full AgentPath path over HTTP: understanding, hybrid retrieval, grading."

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120.0)

    async def probe(self) -> None:
        """Fail early with a clear message when the API is not running."""
        try:
            response = await self._client.get(f"{self._base_url}/health")
        except httpx.HTTPError as exc:
            raise RuntimeError(f"AgentPath API is not reachable at {self._base_url}. Start it with: make api") from exc
        if response.status_code not in (200, 503):
            raise RuntimeError(f"unexpected /health status {response.status_code} from {self._base_url}")

    async def run(self, query: BenchmarkQuery, *, limit: int) -> QueryOutcome:
        started = time.perf_counter()
        response = await self._client.post(
            f"{self._base_url}/api/search",
            json={"query": query.query, "limit": limit},
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code != 200:
            return _outcome(
                query,
                ranked=[],
                abstained=True,
                evidence="error",
                latency_ms=latency_ms,
                notes=f"HTTP {response.status_code}: {response.text[:160]}",
            )

        body = response.json()
        procedures = body.get("procedures") or []
        return _outcome(
            query,
            ranked=[procedure["procedure_id"] for procedure in procedures],
            abstained=bool(body["evidence"]["abstained"]),
            evidence=str(body["evidence"]["level"]),
            latency_ms=latency_ms,
            top_score=float(procedures[0]["score"]) if procedures else 0.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def in_process_full_mode(settings: Settings) -> RAGFlowHybridMode:
    """Full pipeline without HTTP, used when the API is not running.

    Same components as mode 3, so retrieval quality is identical; only the
    latency excludes the HTTP hop. The report labels which one was used.
    """

    class _InProcess(RAGFlowHybridMode):
        name = "agentpath_full_in_process"
        description = (
            "The full AgentPath path in process (query understanding, hybrid retrieval, grading), without the HTTP hop."
        )

        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self._service = SearchService(
                retriever=self._retriever,
                lexicon=load_lexicon(settings.lexicon_file),
                settings=settings,
            )

        async def run(self, query: BenchmarkQuery, *, limit: int) -> QueryOutcome:
            started = time.perf_counter()
            result = await self._service.search(query.query, limit=limit)
            latency_ms = (time.perf_counter() - started) * 1000
            return _outcome(
                query,
                ranked=[hit.procedure_id for hit in result.hits],
                abstained=result.abstained,
                evidence=result.evidence.value,
                latency_ms=latency_ms,
                top_score=result.hits[0].score if result.hits else 0.0,
            )

    return _InProcess(settings)

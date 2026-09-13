"""Composition root: build the search service and its dependencies once.

Wiring lives here rather than inside the route handlers so the same object graph
can be built by the CLI and the evaluation harness. Swapping RAGFlow for another
engine is a change to :func:`build_retriever` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

from agentpath.config import Settings
from agentpath.domain import DependencyHealth
from agentpath.query.lexicon import Lexicon, load_lexicon
from agentpath.retrieval.ports import Retriever
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever
from agentpath.search_service import SearchService

logger = structlog.get_logger(__name__)


@dataclass
class ServiceContext:
    """Everything the API needs for its lifetime."""

    settings: Settings
    lexicon: Lexicon
    client: RAGFlowClient
    retriever: Retriever
    search: SearchService

    async def aclose(self) -> None:
        await self.client.aclose()

    async def dependency_health(self) -> list[DependencyHealth]:
        """Check every dependency the search path relies on.

        The model sidecars are checked directly rather than through RAGFlow: when
        retrieval breaks, the first question is always "which of the three is
        down", and a single aggregated flag cannot answer it.
        """
        checks = [await self.retriever.health()]
        checks.append(
            await _probe_http(
                name="embedder",
                url=f"{self.settings.embedder_url}/health",
                detail=self.settings.embedder_model,
            )
        )
        if self.settings.ragflow_rerank_model:
            checks.append(
                await _probe_http(
                    name="reranker",
                    url=f"{self.settings.reranker_url}/health",
                    detail=self.settings.reranker_model,
                )
            )
        return checks


async def _probe_http(*, name: str, url: str, detail: str) -> DependencyHealth:
    import time

    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return DependencyHealth(name=name, reachable=False, detail=f"{detail}: {exc}")
    latency_ms = (time.perf_counter() - started) * 1000
    reachable = response.status_code == httpx.codes.OK
    return DependencyHealth(
        name=name,
        reachable=reachable,
        detail=detail if reachable else f"{detail}: HTTP {response.status_code}",
        latency_ms=latency_ms,
    )


def build_retriever(settings: Settings, client: RAGFlowClient) -> Retriever:
    """Construct the production retriever.

    Reranking requires both the feature flag and a configured model, so enabling
    it is a deliberate act rather than a side effect of a model happening to be
    registered. The benchmark reports both paths so the trade-off stays visible.
    """
    rerank_model = settings.ragflow_rerank_model if settings.enable_rerank else None
    return RAGFlowRetriever(
        client,
        dataset_name=settings.ragflow_dataset_name,
        rerank_model=rerank_model or None,
        vector_similarity_weight=settings.vector_similarity_weight,
        similarity_threshold=settings.similarity_threshold,
        rerank_candidates=settings.rerank_candidates,
        page_size=settings.page_size,
        llm_keyword_expansion=settings.ragflow_llm_keyword_expansion,
        cross_languages=settings.ragflow_cross_languages,
    )


def build_context(settings: Settings) -> ServiceContext:
    """Build the object graph. Cheap and synchronous; no network calls."""
    lexicon = load_lexicon(settings.lexicon_file)
    client = RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    )
    retriever = build_retriever(settings, client)
    search = SearchService(retriever=retriever, lexicon=lexicon, settings=settings)

    logger.info(
        "context.built",
        dataset=settings.ragflow_dataset_name,
        embedding_model=settings.ragflow_embedding_model,
        rerank_model=settings.ragflow_rerank_model or "disabled",
        vector_similarity_weight=settings.vector_similarity_weight,
        lexicon_version=lexicon.version,
        lexicon_concepts=len(lexicon.concepts),
    )
    return ServiceContext(
        settings=settings,
        lexicon=lexicon,
        client=client,
        retriever=retriever,
        search=search,
    )

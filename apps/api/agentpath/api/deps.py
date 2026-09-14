"""Composition root: build the search service and its dependencies once.

Wiring lives here rather than inside the route handlers so the same object graph
can be built by the CLI and the evaluation harness. Swapping RAGFlow for another
engine is a change to :func:`build_retriever` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import structlog

from agentpath.config import Settings
from agentpath.domain import DependencyHealth
from agentpath.generation import AnswerService, OllamaClient, OllamaRuntimeOptions
from agentpath.procedures import ProcedureCatalogue
from agentpath.query.lexicon import Lexicon, load_lexicon
from agentpath.retrieval.ports import Retriever
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever
from agentpath.screenshots import SCREENSHOT_URL_PREFIX, ScreenshotCatalogue
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
    #: The grounded answer service, or ``None`` when generation is not configured.
    #: Optional rather than required so a context built for search alone - the
    #: evaluation harness, the ingestion CLI, most tests - carries no Ollama client
    #: and cannot fail on one. ``POST /api/answer`` returns 503 when it is absent.
    answers: AnswerService | None = None
    #: Defaulted so every existing constructor call - tests, the CLI, the
    #: evaluation harness - keeps working and simply serves no screenshots.
    screenshots: ScreenshotCatalogue = field(default_factory=ScreenshotCatalogue.empty)
    #: The structured procedure source the answer path rebuilds evidence from.
    #: Defaulted for the same reason: a context built for search alone gets a
    #: catalogue in which every citation is unknown, rather than a filesystem read.
    procedures: ProcedureCatalogue = field(default_factory=ProcedureCatalogue.empty)

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


def build_ollama_client(settings: Settings) -> OllamaClient:
    """Construct the generation client from configuration.

    Lives in the composition root, next to :func:`build_retriever`, so the
    mapping from settings to runtime options is written once. The measurement
    harness and the answer service must not each decide what ``num_ctx`` or
    ``think`` mean, or the numbers the harness reports stop describing the thing
    that ships.

    Deliberately *not* built into :class:`ServiceContext` yet: the API has no
    route that generates an answer, and constructing a client the request path
    never touches would only add a dependency to fail on.
    """
    return OllamaClient(
        settings.ollama_host_url,
        model=settings.ollama_chat_model,
        timeout=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
        keep_alive=settings.ollama_keep_alive,
        options=OllamaRuntimeOptions(
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
            seed=settings.ollama_seed,
            num_predict=settings.ollama_num_predict,
        ),
    )


def build_answer_service(
    settings: Settings,
    context: ServiceContext,
    *,
    client: OllamaClient | None = None,
) -> AnswerService:
    """Compose the grounded answer service over an existing context.

    A factory rather than a field on :class:`ServiceContext`, for the same reason
    :func:`build_ollama_client` is: T8 builds the service layer and T10 adds the
    route. Until a route exists, holding a generation client on the context the
    request path uses would add a dependency that every ``/api/search`` request
    carries and no request needs. The catalogue *is* on the context, because it is
    a local read that has to happen at startup rather than per request.

    ``client`` is injectable so the measurement harness and the tests can supply
    a configured or a fake client without rebuilding the object graph.
    """
    return AnswerService(
        search=context.search,
        catalogue=context.procedures,
        client=client or build_ollama_client(settings),
        settings=settings,
        max_blocks=settings.answer_max_evidence_blocks,
        screenshot_url_prefix=SCREENSHOT_URL_PREFIX,
    )


def build_context(settings: Settings) -> ServiceContext:
    """Build the object graph. Cheap and synchronous; no network calls.

    Three local reads happen here and all three are deliberate. The lexicon is one
    YAML file. The screenshot catalogue is one small JSON sidecar and one PNG per
    committed screenshot - three of each in the Stage 2A slice, ~650 KiB and three
    sha256 digests, single-digit milliseconds. The procedure catalogue is the five
    committed procedure YAMLs, ~50 KiB in total.

    All three are read once at startup precisely so that no YAML parsing, no JSON
    parsing and no image hashing can happen while a request is in flight. For the
    procedure catalogue that is not only a latency argument: the answer path
    rebuilds every step it cites from this data, so it has to be resolved and
    validated before any request can depend on it. They are still local filesystem
    reads with no network calls, which is what "cheap and synchronous" is
    promising.
    """
    lexicon = load_lexicon(settings.lexicon_file)
    screenshots = ScreenshotCatalogue.load(settings.screenshots_dir)
    procedures = ProcedureCatalogue.load(settings.procedures_dir)
    client = RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    )
    retriever = build_retriever(settings, client)
    search = SearchService(retriever=retriever, lexicon=lexicon, settings=settings)

    # Constructing the Ollama client is cheap - an httpx client and a settings
    # snapshot, no connection and no model load - so the answer service is built
    # here rather than per request. Whether the daemon is up and the model pulled
    # is a runtime question, answered by /health and by the 503 on /api/answer,
    # not something to discover while building the object graph.
    answers = AnswerService(
        search=search,
        catalogue=procedures,
        client=build_ollama_client(settings),
        settings=settings,
        max_blocks=settings.answer_max_evidence_blocks,
        screenshot_url_prefix=SCREENSHOT_URL_PREFIX,
    )

    logger.info(
        "context.built",
        dataset=settings.ragflow_dataset_name,
        embedding_model=settings.ragflow_embedding_model,
        rerank_model=settings.ragflow_rerank_model or "disabled",
        vector_similarity_weight=settings.vector_similarity_weight,
        lexicon_version=lexicon.version,
        lexicon_concepts=len(lexicon.concepts),
        screenshots=len(screenshots),
        procedures=len(procedures),
        procedure_steps=procedures.step_count,
    )
    return ServiceContext(
        settings=settings,
        lexicon=lexicon,
        client=client,
        retriever=retriever,
        search=search,
        answers=answers,
        screenshots=screenshots,
        procedures=procedures,
    )

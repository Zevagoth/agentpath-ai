"""RAGFlow implementation of the :class:`~agentpath.retrieval.ports.Retriever` port.

This is the production retrieval path. It maps AgentPath's prepared query onto
RAGFlow's ``/api/v1/retrieval`` call and maps RAGFlow's chunks back into
:class:`~agentpath.domain.Passage` objects.

Hybrid search is RAGFlow's own: term (BM25) similarity and vector similarity are
blended as ``(1 - w) * term + w * vector``, then the candidate set is reordered by
the cross-encoder named in ``rerank_id``. Exact policy terms therefore stay
findable through the term half while synonyms and Arabic phrasings arrive through
the vector half.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agentpath.domain import (
    DependencyHealth,
    Passage,
    PreparedQuery,
    RetrievalMode,
    RetrievedPassages,
)
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable
from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError, RAGFlowUnavailable

logger = structlog.get_logger(__name__)


class RAGFlowRetriever:
    """Retrieves passages from a RAGFlow dataset."""

    def __init__(
        self,
        client: RAGFlowClient,
        *,
        dataset_name: str,
        rerank_model: str | None = None,
        vector_similarity_weight: float = 0.6,
        similarity_threshold: float = 0.15,
        rerank_candidates: int = 32,
        page_size: int = 12,
        llm_keyword_expansion: bool = False,
        cross_languages: list[str] | None = None,
    ) -> None:
        self._client = client
        self._dataset_name = dataset_name
        self._rerank_model = rerank_model
        self._vector_similarity_weight = vector_similarity_weight
        self._similarity_threshold = similarity_threshold
        self._rerank_candidates = rerank_candidates
        self._page_size = page_size
        self._llm_keyword_expansion = llm_keyword_expansion
        self._cross_languages = cross_languages or []
        self._dataset_id: str | None = None

    @property
    def mode(self) -> RetrievalMode:
        return RetrievalMode.HYBRID

    # -- dataset resolution ------------------------------------------------ #

    async def dataset_id(self) -> str:
        """Resolve and cache the dataset ID for the configured dataset name."""
        if self._dataset_id:
            return self._dataset_id
        try:
            dataset_id = await self._client.find_dataset_id(self._dataset_name)
        except RAGFlowUnavailable as exc:
            raise RetrievalUnavailable(str(exc), engine="ragflow") from exc
        except RAGFlowError as exc:
            raise RetrievalError(str(exc), engine="ragflow") from exc
        if not dataset_id:
            raise RetrievalError(
                f"RAGFlow dataset {self._dataset_name!r} does not exist. Run: make ingest",
                engine="ragflow",
            )
        self._dataset_id = dataset_id
        return dataset_id

    # -- retrieval --------------------------------------------------------- #

    async def retrieve(
        self,
        query: PreparedQuery,
        *,
        limit: int,
        product_line: str | None = None,
    ) -> RetrievedPassages:
        if query.is_empty:
            return RetrievedPassages(
                passages=(),
                mode=self.mode,
                engine_ms=0.0,
                diagnostics={"reason": "empty query"},
            )

        dataset_id = await self.dataset_id()

        # Ask for more chunks than we need procedures: several chunks of the same
        # procedure routinely rank next to each other, and aggregation collapses
        # them afterwards.
        page_size = max(self._page_size, limit * 3)
        metadata_condition = _metadata_condition(product_line)

        started = time.perf_counter()
        try:
            data = await self._client.retrieve(
                question=query.search_text,
                dataset_ids=[dataset_id],
                page_size=page_size,
                similarity_threshold=self._similarity_threshold,
                vector_similarity_weight=self._vector_similarity_weight,
                rerank_id=self._rerank_model,
                rerank_candidates_count=max(self._rerank_candidates, page_size),
                highlight=True,
                include_metadata=True,
                metadata_condition=metadata_condition,
                llm_keyword_expansion=self._llm_keyword_expansion,
                cross_languages=self._cross_languages or None,
            )
        except RAGFlowUnavailable as exc:
            raise RetrievalUnavailable(str(exc), engine="ragflow", detail=exc.path) from exc
        except RAGFlowError as exc:
            raise RetrievalError(str(exc), engine="ragflow", detail=exc.path) from exc
        engine_ms = (time.perf_counter() - started) * 1000

        chunks = data.get("chunks") or []
        passages = tuple(_to_passage(chunk) for chunk in chunks)

        logger.debug(
            "ragflow.retrieved",
            chunks=len(passages),
            engine_ms=round(engine_ms, 1),
            reranked=bool(self._rerank_model),
            filtered=bool(metadata_condition),
        )

        return RetrievedPassages(
            passages=passages,
            mode=self.mode,
            engine_ms=engine_ms,
            diagnostics={
                "dataset_id": dataset_id,
                "total_candidates": data.get("total", len(passages)),
                "vector_similarity_weight": self._vector_similarity_weight,
                "similarity_threshold": self._similarity_threshold,
                "rerank_model": self._rerank_model,
                "metadata_filter": metadata_condition,
                "llm_keyword_expansion": self._llm_keyword_expansion,
            },
        )

    async def health(self) -> DependencyHealth:
        started = time.perf_counter()
        try:
            components = await self._client.healthz()
        except RAGFlowUnavailable as exc:
            return DependencyHealth(name="ragflow", reachable=False, detail=str(exc))
        except RAGFlowError as exc:
            # An authentication error still proves the service is answering.
            reachable = exc.code == 401
            return DependencyHealth(
                name="ragflow",
                reachable=reachable,
                detail=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        latency_ms = (time.perf_counter() - started) * 1000
        # Report per-component state: "RAGFlow is up but Elasticsearch is not" is
        # a different problem from "RAGFlow is down", and both are reachable here.
        degraded = sorted(
            name
            for name, state in components.items()
            if name != "status" and isinstance(state, str) and state.lower() != "ok"
        )
        if degraded:
            return DependencyHealth(
                name="ragflow",
                reachable=True,
                detail=f"degraded components: {', '.join(degraded)}",
                latency_ms=latency_ms,
            )
        return DependencyHealth(
            name="ragflow",
            reachable=True,
            detail=f"all components ok, dataset {self._dataset_name!r}",
            latency_ms=latency_ms,
        )


def _metadata_condition(product_line: str | None) -> dict[str, Any] | None:
    """Build RAGFlow's metadata filter for a product-line restriction.

    Applied only when the agent explicitly picks a product line. A filter derived
    from automatic detection would silently delete the correct procedure from the
    candidate set whenever detection was wrong.
    """
    if not product_line:
        return None
    return {
        "logic": "and",
        "conditions": [
            {"name": "product_line", "comparison_operator": "=", "value": product_line},
        ],
    }


def _to_passage(chunk: dict[str, Any]) -> Passage:
    """Map one RAGFlow chunk onto a :class:`Passage`.

    RAGFlow renames its internal fields on the way out (``content_with_weight`` ->
    ``content``, ``docnm_kwd`` -> ``document_keyword``); both spellings are
    accepted here so a future rename does not silently produce empty passages.
    """
    metadata = chunk.get("document_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {str(key): str(value) for key, value in metadata.items() if value is not None}

    document_name = str(chunk.get("document_keyword") or chunk.get("docnm_kwd") or "")
    procedure_id = metadata.get("procedure_id") or _procedure_id_from_document_name(document_name)

    content = str(chunk.get("content") or chunk.get("content_with_weight") or "")
    highlight = chunk.get("highlight")

    scenario_tags = tuple(tag for tag in (metadata.get("scenario_tags") or "").split(",") if tag)

    return Passage(
        chunk_id=str(chunk.get("id") or chunk.get("chunk_id") or ""),
        document_id=str(chunk.get("document_id") or chunk.get("doc_id") or ""),
        procedure_id=procedure_id,
        title=metadata.get("title") or _title_from_content(content) or document_name,
        content=content,
        score=_as_float(chunk.get("similarity")),
        lexical_score=_optional_float(chunk.get("term_similarity")),
        semantic_score=_optional_float(chunk.get("vector_similarity")),
        highlight=str(highlight) if highlight else None,
        product_line=metadata.get("product_line"),
        version=metadata.get("version"),
        effective_date=metadata.get("effective_date"),
        scenario_tags=scenario_tags,
        image_id=str(chunk.get("image_id")) if chunk.get("image_id") else None,
        metadata=metadata,
    )


def _procedure_id_from_document_name(document_name: str) -> str:
    """``NW-HTL-014.md`` -> ``NW-HTL-014``.

    Fallback for when document metadata is unavailable, so retrieval still
    attributes passages correctly.
    """
    if not document_name:
        return "unknown"
    stem = document_name.rsplit(".", 1)[0]
    return stem.strip() or "unknown"


def _title_from_content(content: str) -> str | None:
    """Recover the procedure title from a chunk that contains the H1 heading."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            # Rendered as "NW-HTL-014 — Title"; keep the human part.
            for separator in ("—", "-"):
                if separator in heading:
                    candidate = heading.split(separator, 1)[1].strip()
                    if candidate:
                        return candidate
            return heading
    return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

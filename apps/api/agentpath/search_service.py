"""The search service: query understanding, retrieval, aggregation, grading.

This is the whole AgentPath search pipeline in one readable place, and it is
engine-agnostic: it talks to a :class:`~agentpath.retrieval.ports.Retriever`, so
the same code path serves the production RAGFlow retriever and the benchmark's
lexical baseline. That is what makes the benchmark a fair comparison rather than
two different programs.
"""

from __future__ import annotations

import time

import structlog

from agentpath.config import Settings
from agentpath.domain import PreparedQuery, RetrievalMode, SearchResult
from agentpath.evidence import detect_scenario_overlap, grade_evidence
from agentpath.query.lexicon import Lexicon
from agentpath.query.pipeline import QueryPreprocessor
from agentpath.retrieval.aggregate import aggregate_passages
from agentpath.retrieval.ports import Retriever

logger = structlog.get_logger(__name__)


class SearchService:
    """Runs one search from raw text to graded, ranked procedures."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        lexicon: Lexicon,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._preprocessor = QueryPreprocessor(lexicon)
        self._settings = settings

    @property
    def mode(self) -> RetrievalMode:
        return self._retriever.mode

    async def search(
        self,
        query: str,
        *,
        product_line: str | None = None,
        limit: int | None = None,
    ) -> SearchResult:
        """Execute the pipeline and return a graded result.

        Timings are reported per stage because "first results in about two
        seconds" is a product requirement, and an aggregate number would hide
        which stage broke it.
        """
        limit = limit or self._settings.top_procedures
        total_started = time.perf_counter()

        understanding_started = time.perf_counter()
        prepared = self._preprocessor.prepare(query, product_line=product_line)
        understanding_ms = (time.perf_counter() - understanding_started) * 1000

        if prepared.is_empty:
            return self._empty_result(prepared, understanding_ms)

        retrieved = await self._retriever.retrieve(
            prepared,
            limit=limit,
            # Only an explicit filter narrows the candidate set; see
            # RAGFlowRetriever._metadata_condition for why detection does not.
            product_line=prepared.requested_product_line,
        )

        aggregation_started = time.perf_counter()
        hits = aggregate_passages(retrieved.passages, limit=limit)
        aggregation_ms = (time.perf_counter() - aggregation_started) * 1000

        evidence, reason = grade_evidence(
            hits,
            strong_score=self._settings.evidence_strong_score,
            strong_margin=self._settings.evidence_strong_margin,
            partial_score=self._settings.evidence_partial_score,
        )
        conflicts = detect_scenario_overlap(hits, minimum_score=self._settings.evidence_partial_score)

        total_ms = (time.perf_counter() - total_started) * 1000

        logger.info(
            "search.completed",
            mode=self._retriever.mode.value,
            language=prepared.language.value,
            scenario_tags=list(prepared.scenario_tags),
            product_line=prepared.effective_product_line,
            hits=[hit.procedure_id for hit in hits],
            evidence=evidence.value,
            abstained=evidence.value == "insufficient",
            total_ms=round(total_ms, 1),
            engine_ms=round(retrieved.engine_ms, 1),
            query=prepared.original if self._settings.log_query_text else None,
        )

        return SearchResult(
            query=prepared,
            mode=self._retriever.mode,
            hits=hits,
            evidence=evidence,
            evidence_reason=reason,
            timings_ms={
                "understanding": round(understanding_ms, 2),
                "retrieval": round(retrieved.engine_ms, 2),
                "aggregation": round(aggregation_ms, 2),
                "total": round(total_ms, 2),
            },
            diagnostics={
                "passages_considered": len(retrieved.passages),
                "lexicon_version": self._preprocessor.lexicon_version,
                **dict(retrieved.diagnostics),
            },
            conflicts=conflicts,
        )

    def _empty_result(self, prepared: PreparedQuery, understanding_ms: float) -> SearchResult:
        """A blank query is not an error, and it is not an abstention either.

        Nothing was searched, so the reason says exactly that rather than
        implying the knowledge base was consulted and came up short.
        """
        from agentpath.domain import EvidenceLevel

        return SearchResult(
            query=prepared,
            mode=self._retriever.mode,
            hits=(),
            evidence=EvidenceLevel.INSUFFICIENT,
            evidence_reason="No query text was provided, so no search was performed.",
            timings_ms={
                "understanding": round(understanding_ms, 2),
                "retrieval": 0.0,
                "aggregation": 0.0,
                "total": round(understanding_ms, 2),
            },
            diagnostics={"passages_considered": 0},
        )

"""The retrieval port.

Everything above this line (the search service, the API, the CLI, the evaluation
harness) depends only on this protocol. Everything below it is an engine detail.

``RAGFlowRetriever`` is the production implementation. The evaluation harness
adds a lexical baseline implementation for benchmarking only. Swapping RAGFlow
for another engine later means writing one class, not touching the application.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentpath.domain import DependencyHealth, PreparedQuery, RetrievalMode, RetrievedPassages


class RetrievalError(RuntimeError):
    """Retrieval failed for a reason the caller cannot fix by retrying differently."""

    def __init__(self, message: str, *, engine: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.engine = engine
        self.detail = detail


class RetrievalUnavailable(RetrievalError):
    """The engine is unreachable or not ready (connection refused, timeout)."""


@runtime_checkable
class Retriever(Protocol):
    """Retrieves passages for a prepared query."""

    @property
    def mode(self) -> RetrievalMode:
        """Which retrieval path this implementation represents."""
        ...

    async def retrieve(
        self,
        query: PreparedQuery,
        *,
        limit: int,
        product_line: str | None = None,
    ) -> RetrievedPassages:
        """Return candidate passages, best first.

        Implementations must not raise on "nothing found": an empty result is a
        legitimate answer and drives abstention. They should raise
        :class:`RetrievalUnavailable` when the engine itself is broken, so the
        API can distinguish "no procedure exists" from "the search engine is
        down" - two situations that must never look the same to an agent.
        """
        ...

    async def health(self) -> DependencyHealth:
        """Report reachability without performing a search."""
        ...

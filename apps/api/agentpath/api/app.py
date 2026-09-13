"""The AgentPath AI HTTP API.

Endpoints:

* ``POST /api/search`` - the search pipeline: understand, retrieve, grade.
* ``GET  /health``     - reachability of RAGFlow and the model sidecars.

Security posture: this service has no authentication and is bound to
``127.0.0.1`` by configuration. That is acceptable for a local proof of concept
and only because nothing listens on a routable interface. Exposing it would
require an auth layer first; see SECURITY.md.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentpath import __version__
from agentpath.api.deps import ServiceContext, build_context
from agentpath.api.schemas import (
    DependencyStatus,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from agentpath.config import Settings, get_settings
from agentpath.logging_setup import configure_logging
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None, context: ServiceContext | None = None) -> FastAPI:
    """Build the application.

    ``context`` is injectable so tests can supply a fake retriever without
    reaching for monkeypatching or a live RAGFlow.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.context = context or build_context(settings)
        app.state.settings = settings
        logger.info("api.started", host=settings.api_host, port=settings.api_port)
        try:
            yield
        finally:
            if context is None:
                await app.state.context.aclose()
            logger.info("api.stopped")

    app = FastAPI(
        title="AgentPath AI",
        version=__version__,
        summary="Describe the case. Find the right path.",
        description=(
            "Hybrid retrieval over a synthetic customer-service knowledge base. "
            "All knowledge-base content is fictional."
        ),
        lifespan=lifespan,
    )

    # The web app runs on a different loopback port, so the browser needs CORS.
    # Loopback origins only: this must not become a wildcard.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            structlog.contextvars.unbind_contextvars("request_id")
        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round(duration_ms, 1),
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RetrievalUnavailable)
    async def _unavailable(_request: Request, exc: RetrievalUnavailable) -> JSONResponse:
        """The engine is down.

        This must never look like "no procedure found": one means the knowledge
        base has no answer, the other means we could not look. Conflating them
        would teach agents to trust an empty result.
        """
        logger.error("retrieval.unavailable", engine=exc.engine, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "retrieval_unavailable",
                "message": (
                    "The retrieval engine is not reachable, so no search was performed. "
                    "This is not the same as finding no procedure."
                ),
                "detail": str(exc),
                "engine": exc.engine,
            },
        )

    @app.exception_handler(RetrievalError)
    async def _retrieval_error(_request: Request, exc: RetrievalError) -> JSONResponse:
        logger.error("retrieval.failed", engine=exc.engine, detail=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "retrieval_failed",
                "message": "The retrieval engine rejected the request.",
                "detail": str(exc),
                "engine": exc.engine,
            },
        )

    @app.post(
        "/api/search",
        response_model=SearchResponse,
        summary="Find the procedures that match a customer-service case",
    )
    async def search(request: Request, payload: SearchRequest) -> SearchResponse:
        ctx: ServiceContext = request.app.state.context
        result = await ctx.search.search(
            payload.query,
            product_line=payload.product_line,
            limit=payload.limit,
        )
        return SearchResponse.from_result(result)

    @app.get("/health", response_model=HealthResponse, summary="Dependency health")
    async def health(request: Request, response: Response) -> HealthResponse:
        ctx: ServiceContext = request.app.state.context
        checks = await ctx.dependency_health()
        # RAGFlow is the only hard requirement: without it there is no search.
        critical_ok = all(check.reachable for check in checks if check.name == "ragflow")
        if not critical_ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if all(check.reachable for check in checks) else "degraded",
            version=__version__,
            mode=ctx.retriever.mode,
            dataset=ctx.settings.ragflow_dataset_name,
            dependencies=[
                DependencyStatus(
                    name=check.name,
                    reachable=check.reachable,
                    detail=check.detail,
                    latency_ms=round(check.latency_ms, 1) if check.latency_ms is not None else None,
                )
                for check in checks
            ],
        )

    return app


app = create_app()

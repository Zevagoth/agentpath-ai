"""The AgentPath AI HTTP API.

Endpoints:

* ``POST /api/search``                    - the search pipeline: understand, retrieve, grade.
* ``GET  /api/screenshots/{screenshot_id}`` - one committed AgentDesk capture, as PNG.
* ``GET  /health``                        - reachability of RAGFlow and the model sidecars.

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
    AnswerRequest,
    AnswerResponse,
    DependencyStatus,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from agentpath.config import Settings, get_settings
from agentpath.logging_setup import configure_logging
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable
from agentpath.screenshots import (
    MAX_SCREENSHOT_ID_LENGTH,
    SCREENSHOT_MEDIA_TYPE,
    SCREENSHOT_URL_PREFIX,
)

logger = structlog.get_logger(__name__)

#: A rejected screenshot id is logged so a typo is diagnosable, and truncated
#: first so a 4 KiB request path cannot bloat a log line. Tied to the validity
#: bound: anything longer than this could not have been served anyway, so nothing
#: diagnostic is lost.
MAX_LOGGED_ID_LENGTH = MAX_SCREENSHOT_ID_LENGTH


def _if_none_match(header: str | None) -> frozenset[str]:
    """Parse ``If-None-Match`` into the set of entity tags it offers.

    Only the shapes a real cache sends: a comma-separated list of quoted tags,
    optionally weak-prefixed. ``*`` is not special-cased because this route has no
    unsafe methods for it to guard.
    """
    if not header:
        return frozenset()
    tags = set()
    for raw in header.split(","):
        candidate = raw.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:].strip()
        if candidate:
            tags.add(candidate)
    return frozenset(tags)


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
    #
    # This already covers the screenshot route and does not need widening. An
    # `<img src>` load is not a CORS request at all - the browser fetches it and
    # renders it without consulting these headers, and only a `crossorigin`
    # attribute or a canvas read would make the response's origin matter. `GET` is
    # allowed here in any case, so even a `fetch()` for the bytes from the console
    # works. Adding an origin, a method or a header for the sake of images would
    # have bought nothing.
    # Port 3100 is where Playwright serves the production build. Without it the
    # browser specs could not reach this API at all: every request was answered
    # with "Disallowed CORS origin", which surfaced in the console as a failed
    # fetch and so looked exactly like the API being down. Naming both ports keeps
    # the rule narrow - still loopback-only, still no wildcard - while letting the
    # end-to-end tests exercise the same code path a developer uses on 3000.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:3100",
            "http://localhost:3100",
        ],
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
        return SearchResponse.from_result(result, ctx.screenshots)

    @app.post(
        "/api/answer",
        response_model=AnswerResponse,
        summary="Answer a customer-service case with cited procedure steps",
    )
    async def answer(request: Request, payload: AnswerRequest) -> AnswerResponse:
        """Generate a grounded answer, or refuse in a way that says which refusal.

        Always HTTP 200 when the service ran, because the interesting outcomes are
        not transport failures: abstaining, needing a clarification and failing to
        generate are all legitimate results with different meanings, and ``status``
        distinguishes them. The only 503 here is "generation is not configured at
        all", which is a deployment fact rather than a result.

        `/api/search` is untouched by this route. An agent who only wants to find
        the procedure keeps a fast search that never loads a model.
        """
        ctx: ServiceContext = request.app.state.context
        if ctx.answers is None:
            return JSONResponse(  # type: ignore[return-value]
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "generation_unavailable",
                    "message": (
                        "Answer generation is not configured on this deployment. Search is "
                        "unaffected: POST /api/search still works."
                    ),
                },
            )
        result = await ctx.answers.answer(
            payload.query,
            product_line=payload.product_line,
            limit=payload.limit,
            context=payload.case_context,
        )
        return AnswerResponse.from_result(result, ctx.screenshots)

    @app.get(
        SCREENSHOT_URL_PREFIX + "/{screenshot_id}",
        summary="Fetch one instructional screenshot",
        responses={
            200: {"content": {SCREENSHOT_MEDIA_TYPE: {}}, "description": "The PNG."},
            304: {"description": "The client's cached copy is current."},
            404: {"description": "No such screenshot."},
        },
        response_class=Response,
    )
    async def screenshot(request: Request, screenshot_id: str) -> Response:
        """Serve a screenshot by id.

        The security model, in full:

        1. ``screenshot_id`` is validated for shape and length, then looked up in
           the catalogue that :func:`~agentpath.api.deps.build_context` built at
           startup by listing the screenshots directory. Anything that is not a
           key in that catalogue is a 404.
        2. **Request input never reaches a path operation.** There is no ``/``,
           no ``os.path.join`` and no ``open`` on this code path at all - the
           bytes were read at startup and are already in memory. Traversal has
           nothing to traverse: the request only ever selects from a closed set.
        3. Malformed and unknown ids get the byte-identical 404. A distinct 422
           for "well-formed but absent" would be an oracle for enumerating the id
           space, and the request input is not echoed back at all - reflecting it
           would gain a caller nothing it does not already know and would put
           attacker-chosen bytes in a response body.
        4. No absolute path appears in the body, the headers or the log line.

        What actually reaches this handler is narrower than it looks, and the
        tests assert the observed split rather than assuming one. Starlette
        percent-decodes the path *before* routing and the path parameter cannot
        span a ``/``, so ``../../etc/passwd``, ``..%2f..%2f...`` and
        ``%2e%2e%2f...`` never match this route at all - they 404 in the router.
        What does arrive is anything that stays a single decoded segment: a
        double-encoded ``%252e%252e%252f``, backslashes, a NUL byte, a 4000-byte
        id, an uppercase id, a ``.png`` suffix, and a bare ``..`` from ``%2e%2e``.
        Those are the cases step 1 exists for.

        Caching: the sha256 is spent on a strong ``ETag`` rather than on a long
        ``immutable`` lifetime. The bytes are immutable for a given digest, but
        this URL is keyed by the stable id, so regenerating a capture replaces
        what it returns. An hour of freshness plus revalidation gets the repeat
        fetches off the wire without ever serving a stale image.
        """
        ctx: ServiceContext = request.app.state.context
        record = ctx.screenshots.resolve(screenshot_id)
        if record is None:
            logger.info(
                "screenshot.not_found",
                # Bounded and character-restricted by the check that just failed;
                # truncated anyway so a hostile id cannot bloat a log line.
                requested=screenshot_id[:MAX_LOGGED_ID_LENGTH],
                known=len(ctx.screenshots),
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "screenshot_not_found",
                    "message": "No screenshot is available under that id.",
                },
                headers={"X-Content-Type-Options": "nosniff"},
            )

        headers = {
            "ETag": record.etag,
            "Cache-Control": "public, max-age=3600, must-revalidate",
            # The body is a PNG whose type was verified against its magic bytes at
            # startup, so there is no reason for a browser to guess anything else.
            "X-Content-Type-Options": "nosniff",
        }
        if record.etag in _if_none_match(request.headers.get("if-none-match")):
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
        return Response(content=record.image, media_type=SCREENSHOT_MEDIA_TYPE, headers=headers)

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

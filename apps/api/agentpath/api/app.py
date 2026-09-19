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
    CorpusListResponse,
    CorpusSearchRequest,
    CorpusSearchResponse,
    CorpusSummary,
    DependencyStatus,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from agentpath.config import Settings, get_settings
from agentpath.corpus import PROCEDURES_CORPUS_ID, render_document_html
from agentpath.corpus_retrieval.service import default_method
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
        "/api/corpora",
        response_model=CorpusListResponse,
        summary="List the knowledge bases available to search",
    )
    async def corpora(request: Request) -> CorpusListResponse:
        """What can be searched, and under what licence.

        The console's corpus selector reads this rather than hardcoding a list, so a
        corpus that failed to load is absent here and visibly absent there. Licence and
        attribution travel with every entry: this deployment indexes someone else's
        Apache-2.0 data, and a client that can render a result must be able to render
        where it came from.
        """
        ctx: ServiceContext = request.app.state.context
        summaries: list[CorpusSummary] = []
        for corpus in ctx.corpora:
            methods = ctx.corpus_methods.get(corpus.corpus_id) or {}
            if not methods:
                # A corpus with no usable method would be a selector entry that fails
                # on click. It stays out of the list and says why in `problems`.
                continue
            summaries.append(
                CorpusSummary(
                    corpus_id=corpus.corpus_id,
                    title=corpus.title,
                    licence=corpus.licence,
                    attribution=corpus.attribution,
                    counts=corpus.counts,
                    image_retrieval_mode=corpus.image_retrieval_mode,
                    methods=list(methods),
                    default_method=default_method(methods),
                )
            )
        problems = list(ctx.corpora.problems)
        problems.extend(
            f"corpus {corpus.corpus_id!r} loaded but has no available retrieval method"
            for corpus in ctx.corpora
            if not ctx.corpus_methods.get(corpus.corpus_id)
        )
        return CorpusListResponse(
            corpora=summaries,
            procedures_corpus_id=PROCEDURES_CORPUS_ID,
            problems=problems,
        )

    @app.post(
        "/api/corpus-search",
        response_model=CorpusSearchResponse,
        summary="Search one external corpus for documents and images",
        responses={404: {"description": "No such corpus, or that method is not available for it."}},
    )
    async def corpus_search(request: Request, payload: CorpusSearchRequest) -> CorpusSearchResponse:
        """Retrieve documents and images from exactly one corpus.

        `corpus_id` is required and there is no "search everything" value. Mixing two
        corpora in one result is not disabled by a check here - each service holds one
        corpus and one dataset, so there is no code path that could produce it.

        The synthetic procedures are refused by name rather than falling through to a
        generic 404, because "wrong endpoint" and "no such corpus" are different
        mistakes and only one of them is fixed by changing the URL.
        """
        ctx: ServiceContext = request.app.state.context

        if payload.corpus_id == PROCEDURES_CORPUS_ID:
            return JSONResponse(  # type: ignore[return-value]
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "wrong_endpoint_for_corpus",
                    "message": (
                        f"{PROCEDURES_CORPUS_ID!r} is the procedure knowledge base, which is served by "
                        "POST /api/search and POST /api/answer. This endpoint serves external "
                        "document corpora."
                    ),
                },
            )

        corpus = ctx.corpora.corpus(payload.corpus_id)
        if corpus is None:
            logger.info(
                "corpus.search_unknown_corpus",
                requested=payload.corpus_id[:MAX_LOGGED_ID_LENGTH],
                available=list(ctx.corpora.ids),
            )
            return JSONResponse(  # type: ignore[return-value]
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "corpus_not_found",
                    "message": "No corpus under that id.",
                    "available": list(ctx.corpora.ids),
                },
            )

        service = ctx.corpus_method(payload.corpus_id, payload.method)
        if service is None:
            available = sorted(method.value for method in (ctx.corpus_methods.get(payload.corpus_id) or {}))
            return JSONResponse(  # type: ignore[return-value]
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "method_not_available",
                    "message": (
                        f"Method {payload.method.value if payload.method else 'default'!r} is not available "
                        f"for corpus {payload.corpus_id!r}."
                    ),
                    "available": available,
                },
            )

        result = await service.search(
            payload.question,
            limit=payload.limit,
            image_limit=payload.image_limit,
        )
        return CorpusSearchResponse.from_result(result, service)

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

    @app.get(
        "/api/sources/{source_id}",
        summary="Open an external corpus document at the cited position",
        responses={
            200: {"content": {"text/html": {}}, "description": "The document."},
            404: {"description": "No such source."},
        },
        response_class=Response,
    )
    async def source_document(request: Request, source_id: str) -> Response:
        """Serve a corpus document as its own "original", scrolled to the citation.

        This is what the console's `Open original source` button opens for a corpus
        that records no upstream document URL - which is MRAMG-Manual, whose images
        carry URLs but whose documents do not.

        The security model is the screenshot route's, with one deliberate
        difference. There, the bytes are in memory and the request can only pick
        from a closed set. Here the document text is in memory too, but the *images
        referenced by the page* are files, so paths exist somewhere. They are
        resolved once at startup and checked to sit inside the corpus's own images
        directory (`agentpath.corpus._image_record`), and the request supplies only
        an opaque id that must already be a key in that registry. There is no path
        operation on a request-derived value on this code path.

        `char` and `to` are read from the query string rather than the fragment,
        because a fragment never reaches the server. The fragment is still what the
        provenance link carries, so the page also works when opened without them -
        it just starts at the top and says so.
        """
        ctx: ServiceContext = request.app.state.context
        found = ctx.corpora.source(source_id)
        if found is None:
            logger.info(
                "corpus.source_not_found",
                requested=source_id[:MAX_LOGGED_ID_LENGTH],
                known=sum(len(corpus.documents) for corpus in ctx.corpora),
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "source_not_found", "message": "No source document under that id."},
                headers={"X-Content-Type-Options": "nosniff"},
            )

        corpus, document = found
        start = _positive_int(request.query_params.get("char"))
        end = _positive_int(request.query_params.get("to"))
        if start is not None and end is None:
            end = start + 1_400

        body = render_document_html(document, corpus, char_start=start, char_end=end)
        return Response(
            content=body,
            media_type="text/html; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=300, must-revalidate",
                "X-Content-Type-Options": "nosniff",
                # The page embeds only images from this origin and runs no script.
                # Stated in a header as well as in the markup, so a future edit that
                # adds a script tag stops working rather than quietly executing.
                "Content-Security-Policy": (
                    "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
                ),
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get(
        "/api/corpus-images/{corpus_id}/{image_id}",
        summary="Fetch one image from an external corpus",
        responses={
            200: {"content": {"image/jpeg": {}}, "description": "The image."},
            304: {"description": "The client's cached copy is current."},
            404: {"description": "No such image."},
        },
        response_class=Response,
    )
    async def corpus_image(request: Request, corpus_id: str, image_id: str) -> Response:
        """Serve an image by corpus and id.

        Read from disk per request rather than held in memory: MRAMG-Manual is 2,599
        files and 147 MiB, which is the one place this route differs from
        `/api/screenshots`. The path is not built from the request - it was resolved
        and bounds-checked at startup, and the request only selects a key.

        A malformed id, an unknown id and an unknown corpus all return the
        byte-identical 404, so the route cannot be used to enumerate what exists.
        """
        ctx: ServiceContext = request.app.state.context
        found = ctx.corpora.image(corpus_id, image_id)
        if found is None:
            logger.info(
                "corpus.image_not_found",
                corpus=corpus_id[:MAX_LOGGED_ID_LENGTH],
                requested=image_id[:MAX_LOGGED_ID_LENGTH],
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "corpus_image_not_found", "message": "No image under that id."},
                headers={"X-Content-Type-Options": "nosniff"},
            )

        _corpus, record = found
        try:
            payload = record.path.read_bytes()
        except OSError:
            # The file was there at startup and is not now. A 404 is the honest
            # answer; the evaluation's broken-image rate is what measures this.
            logger.warning("corpus.image_unreadable", image=record.image_id)
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "corpus_image_not_found", "message": "No image under that id."},
                headers={"X-Content-Type-Options": "nosniff"},
            )

        etag = f'"{record.image_id}-{record.byte_size}"'
        headers = {
            "ETag": etag,
            "Cache-Control": "public, max-age=3600, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        }
        if etag in _if_none_match(request.headers.get("if-none-match")):
            return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
        return Response(content=payload, media_type=record.media_type, headers=headers)

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


def _positive_int(raw: str | None) -> int | None:
    """A non-negative integer from a query parameter, or ``None``.

    Total rather than validating: these come from a URL a user can edit, so a
    nonsense value means "no position given" and the document opens at the top.
    Bounded so an absurd offset cannot become a large slice operation.
    """
    if raw is None or not raw.isdigit() or len(raw) > 9:
        return None
    return int(raw)


app = create_app()

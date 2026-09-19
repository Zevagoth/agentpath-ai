"""Async HTTP client for the RAGFlow API.

One place owns the RAGFlow wire protocol: authentication, the ``{code, data,
message}`` envelope, timeouts and error translation. Both the retriever and the
ingestion service use it, so a change in RAGFlow's API surface has exactly one
landing site.

Reference: RAGFlow v0.27.2 HTTP API. Verified against the pinned checkout in
``.external/ragflow`` rather than assumed from documentation.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx
import structlog

logger = structlog.get_logger(__name__)

#: RAGFlow returns HTTP 200 with a business code in the body. 0 means success.
SUCCESS_CODE = 0


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """Identity of a chunk RAGFlow just wrote.

    ``image_id`` is RAGFlow's ``img_id`` (``"{dataset_id}-{chunk_id}"``), renamed
    on the way out of the API. It is ``None`` for a text-only chunk. It is the
    handle the retrieval path carries as ``Passage.image_id``.
    """

    chunk_id: str
    image_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkUpsert:
    """Outcome of :meth:`RAGFlowClient.upsert_chunk`.

    ``action`` is one of:

    * ``created`` - the chunk did not exist and was written.
    * ``unchanged`` - an identical chunk was already there; nothing was posted.
    * ``image_attached`` - the text matched an existing chunk that carried no
      image, so only the image was added.

    ``superseded_chunk_ids`` are earlier versions of the same logical chunk that
    were deleted because their content changed.

    ``image_blocked`` means an image was asked for but not written, because an
    object was already stored under this chunk's key and RAGFlow would have
    concatenated onto it. The chunk text is there; the image is not.
    """

    chunk_id: str
    image_id: str | None
    action: str
    superseded_chunk_ids: tuple[str, ...] = ()
    image_blocked: bool = False


def encode_chunk_image(image: bytes) -> str:
    """Encode image bytes the way RAGFlow's ``image_base64`` field demands.

    RAGFlow decodes with ``base64.b64decode(value, validate=True)``, so anything
    outside the standard alphabet is rejected server-side with a bare "Invalid
    ``image_base64``". The two ways to get that error are a data URI
    (``data:image/png;base64,...`` - the prefix is not base64) and URL-safe
    base64. Both are caught here instead, with a message that says what to do,
    and the encoding is done in one place so no caller has to know the rule.
    """
    if isinstance(image, str):
        raise ValueError(
            "image must be raw bytes: the client does the base64 encoding. "
            "Pre-encoded strings and data URIs are rejected because RAGFlow "
            "decodes with validate=True and a 'data:image/...;base64,' prefix fails."
        )
    if not isinstance(image, bytes | bytearray | memoryview):
        raise TypeError(f"image must be raw bytes, got {type(image).__name__}")
    payload = bytes(image)
    if not payload:
        raise ValueError("image is empty; RAGFlow rejects an empty `image_base64`")
    if payload[:5].lower() == b"data:":
        raise ValueError(
            "image looks like a data URI; pass the decoded bytes instead. "
            "RAGFlow validates strict base64 and the 'data:' prefix fails."
        )
    return base64.b64encode(payload).decode("ascii")


class RAGFlowError(RuntimeError):
    """RAGFlow answered, but rejected the request."""

    def __init__(self, message: str, *, code: int | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


class RAGFlowUnavailable(RAGFlowError):
    """RAGFlow could not be reached at all."""


class RAGFlowClient:
    """Minimal, typed client for the endpoints AgentPath AI needs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    # -- lifecycle --------------------------------------------------------- #

    async def __aenter__(self) -> Self:
        self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                # Only authentication belongs in the client defaults. A default
                # Content-Type would override the multipart boundary httpx
                # generates for file uploads, and RAGFlow would reject the body
                # with "No file part!". JSON calls set the header per request.
                headers=self._auth_headers(),
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **self._auth_headers()}

    # -- transport --------------------------------------------------------- #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expect_envelope: bool = True,
    ) -> Any:
        client = self._ensure_client()
        url = path if path.startswith("http") else f"{self._base_url}{path}"
        started = time.perf_counter()
        try:
            response = await client.request(
                method,
                url,
                json=json,
                params=params,
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            raise RAGFlowUnavailable(f"RAGFlow timed out after {self._timeout}s", path=path) from exc
        except httpx.HTTPError as exc:
            raise RAGFlowUnavailable(f"RAGFlow is unreachable: {exc}", path=path) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise RAGFlowError("RAGFlow rejected the API key", code=401, path=path)
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise RAGFlowUnavailable(
                f"RAGFlow returned HTTP {response.status_code}", code=response.status_code, path=path
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGFlowError(f"RAGFlow returned a non-JSON body (HTTP {response.status_code})", path=path) from exc

        logger.debug("ragflow.request", method=method, path=path, status=response.status_code, ms=round(elapsed_ms, 1))

        if not expect_envelope:
            return payload

        if not isinstance(payload, dict) or "code" not in payload:
            raise RAGFlowError(f"unexpected RAGFlow response shape: {type(payload).__name__}", path=path)

        code = payload.get("code")
        if code != SUCCESS_CODE:
            message = payload.get("message") or "unknown RAGFlow error"
            raise RAGFlowError(f"RAGFlow error {code}: {message}", code=code, path=path)

        return payload.get("data")

    # -- system ------------------------------------------------------------ #

    async def healthz(self) -> dict[str, Any]:
        """System health, per component.

        This endpoint answers outside the usual ``{code, data}`` envelope - it
        returns the component map directly, e.g.
        ``{"db": "ok", "doc_engine": "ok", "redis": "ok", "storage": "ok",
        "status": "ok"}`` - so envelope parsing is switched off here.
        """
        payload = await self._request("GET", "/api/v1/system/healthz", expect_envelope=False)
        return payload if isinstance(payload, dict) else {"status": str(payload)}

    # -- retrieval --------------------------------------------------------- #

    async def retrieve(
        self,
        *,
        question: str,
        dataset_ids: list[str],
        page_size: int = 12,
        similarity_threshold: float = 0.15,
        vector_similarity_weight: float = 0.6,
        rerank_id: str | None = None,
        rerank_candidates_count: int = 32,
        highlight: bool = True,
        include_metadata: bool = True,
        metadata_condition: dict[str, Any] | None = None,
        llm_keyword_expansion: bool = False,
        cross_languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Hybrid retrieval over the given datasets.

        Notes on two parameters whose names mislead:

        * ``keyword`` (exposed here as ``llm_keyword_expansion``) does not switch
          on lexical matching. BM25 term scoring is always active, weighted by
          ``1 - vector_similarity_weight``. The flag asks a chat model to extract
          keywords and append them to the question, which costs an LLM round trip
          per search. AgentPath expands deterministically instead.
        * ``cross_languages`` likewise triggers LLM translation of the query.

        ``rerank_candidates_count`` must be at least ``page * page_size``, which
        RAGFlow validates server-side.
        """
        body: dict[str, Any] = {
            "question": question,
            "dataset_ids": dataset_ids,
            "page": 1,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "rerank_candidates_count": max(rerank_candidates_count, page_size),
            "highlight": highlight,
            "keyword": llm_keyword_expansion,
            "include_metadata": include_metadata,
        }
        if rerank_id:
            body["rerank_id"] = rerank_id
        if metadata_condition:
            body["metadata_condition"] = metadata_condition
        if cross_languages:
            body["cross_languages"] = cross_languages

        data = await self._request("POST", "/api/v1/retrieval", json=body)
        return data or {"chunks": [], "total": 0}

    # -- datasets ---------------------------------------------------------- #

    async def list_datasets(self, *, name: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": 1, "page_size": 100}
        if name:
            params["name"] = name
        data = await self._request("GET", "/api/v1/datasets", params=params)
        if isinstance(data, dict):
            return list(data.get("data") or [])
        return list(data or [])

    async def find_dataset_id(self, name: str) -> str | None:
        """Look up a dataset ID by exact name, or ``None`` when absent.

        RAGFlow reports "no dataset by that name" as business code 102 with a
        message about lacking permission for the *name*, which reads like an
        authorization failure but is really a miss. Treating it as fatal would
        make first-run ingestion impossible, so it is swallowed here and the
        unfiltered listing decides.
        """
        try:
            for dataset in await self.list_datasets(name=name):
                if dataset.get("name") == name:
                    return str(dataset["id"])
        except RAGFlowError as exc:
            if exc.code not in {102, 108}:
                raise
            logger.debug("ragflow.dataset_lookup_miss", dataset=name, code=exc.code)

        for dataset in await self.list_datasets():
            if dataset.get("name") == name:
                return str(dataset["id"])
        return None

    async def create_dataset(
        self,
        *,
        name: str,
        description: str,
        embedding_model: str,
        chunk_method: str = "naive",
        parser_config: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "embedding_model": embedding_model,
            "chunk_method": chunk_method,
        }
        if parser_config:
            body["parser_config"] = parser_config
        data = await self._request("POST", "/api/v1/datasets", json=body)
        if not isinstance(data, dict) or "id" not in data:
            raise RAGFlowError(f"dataset creation returned no id: {data!r}")
        return str(data["id"])

    async def update_dataset(self, dataset_id: str, **fields: Any) -> None:
        await self._request("PUT", f"/api/v1/datasets/{dataset_id}", json=fields)

    # -- documents --------------------------------------------------------- #

    async def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents",
            params={"page": 1, "page_size": 100},
        )
        if isinstance(data, dict):
            return list(data.get("docs") or data.get("data") or [])
        return list(data or [])

    async def upload_document(self, dataset_id: str, filename: str, content: bytes) -> str:
        """Upload one document. Multipart, so it bypasses the JSON helper."""
        client = self._ensure_client()
        url = f"{self._base_url}/api/v1/datasets/{dataset_id}/documents"
        try:
            response = await client.post(
                url,
                files={"file": (filename, content, "text/markdown")},
                headers=self._auth_headers(),
            )
        except httpx.HTTPError as exc:
            raise RAGFlowUnavailable(f"upload failed: {exc}", path=url) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGFlowError(f"upload returned a non-JSON body (HTTP {response.status_code})") from exc

        if payload.get("code") != SUCCESS_CODE:
            raise RAGFlowError(f"upload rejected: {payload.get('message')}", code=payload.get("code"))

        data = payload.get("data")
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id"):
                return str(entry["id"])
        raise RAGFlowError(f"upload returned no document id: {payload!r}")

    async def update_document(self, dataset_id: str, document_id: str, **fields: Any) -> None:
        await self._request(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}",
            json=fields,
        )

    async def delete_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        """Delete documents from a dataset.

        Scoped deliberately: callers pass explicit IDs, never a wildcard, so this
        cannot empty a dataset by accident.
        """
        if not document_ids:
            return
        await self._request(
            "DELETE",
            f"/api/v1/datasets/{dataset_id}/documents",
            json={"ids": document_ids},
        )

    async def parse_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        await self._request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/chunks",
            json={"document_ids": document_ids},
        )

    async def list_chunks(
        self, dataset_id: str, document_id: str, *, page: int = 1, page_size: int = 100
    ) -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            params={"page": page, "page_size": page_size},
        )
        return data or {}

    #: RAGFlow refuses a larger page: `ValueError('page_size must be less than or
    #: equal to 100')`. Named here so callers that need every chunk paginate rather
    #: than asking for more and silently receiving one page.
    MAX_CHUNK_PAGE_SIZE = 100

    async def count_chunks(self, dataset_id: str, document_id: str) -> int:
        """Every chunk of one document, counted across pages.

        A single `list_chunks` call returns at most 100, and a document that has
        more would be silently undercounted - which matters because that count is
        reported as "what RAGFlow indexed". The largest MRAMG manual produced 97
        chunks, close enough to the cap that relying on one page was luck.
        """
        total = 0
        page = 1
        while True:
            data = await self.list_chunks(dataset_id, document_id, page=page, page_size=self.MAX_CHUNK_PAGE_SIZE)
            chunks = data.get("chunks") if isinstance(data, dict) else None
            found = len(chunks or [])
            total += found
            if found < self.MAX_CHUNK_PAGE_SIZE:
                return total
            page += 1

    # -- chunk writes ------------------------------------------------------ #
    #
    # Everything below exists so the instructional screenshots can live in
    # RAGFlow as image-bearing chunks. Four server-side facts shape the design.
    # The first two are read from the pinned checkout; the last two were measured
    # against the running stack on a throwaway dataset, because neither is
    # documented and both change what a safe write looks like.
    #
    # 1. ``chunk_id = xxhash64(content + document_id)`` (chunk_api.py l.1108).
    #    The id is a pure function of the text and the document, not a fresh uuid.
    # 2. An image is stored in MinIO under bucket ``dataset_id``, object key
    #    ``chunk_id``, and ``store_chunk_image`` (api/utils/image_utils.py)
    #    *concatenates onto an existing object and re-saves it as JPEG*.
    #    Measured: a 48x24 PNG posted twice for the same content came back as a
    #    48x48 JPEG, and a third post as 48x72. The screenshot stacks on top of
    #    itself instead of replacing.
    # 3. Re-parsing a document deletes every chunk of it, manual ones included -
    #    ``parse`` issues an unfiltered ``delete({"doc_id": id})`` - but leaves
    #    the stored images behind. Measured: after a re-parse the manual chunk was
    #    gone while its image was still served, so re-adding the same content
    #    stacked onto that orphan (48x24 -> 48x48).
    # 4. No API path removes a chunk image. Deleting a chunk by id keeps the
    #    object, and ``delete_all`` keeps it too on this deployment despite
    #    calling ``delete_chunk_images`` server-side (measured: the object was
    #    still served afterwards, and the next add stacked onto it).
    #
    # So the safe rule is: never post an image to an object key that is already
    # occupied. :meth:`upsert_chunk` enforces it by checking the key first, and a
    # clean key is obtained by changing the content or re-uploading the document,
    # both of which move the chunk id.

    async def add_chunk(
        self,
        dataset_id: str,
        document_id: str,
        *,
        content: str,
        important_keywords: Sequence[str] | None = None,
        questions: Sequence[str] | None = None,
        tag_keywords: Sequence[str] | None = None,
        image: bytes | None = None,
    ) -> ChunkRef:
        """Add one chunk to a document, optionally carrying an image.

        ``image`` takes raw bytes. Encoding happens here (see
        :func:`encode_chunk_image`) so no caller can hand RAGFlow a data URI.

        Prefer :meth:`upsert_chunk` for anything that runs more than once: this
        method posts unconditionally, and a repeat post of identical content
        stacks the stored image.
        """
        body: dict[str, Any] = {"content": content}
        if important_keywords:
            body["important_keywords"] = list(important_keywords)
        if questions:
            body["questions"] = list(questions)
        if tag_keywords:
            body["tag_kwd"] = list(tag_keywords)
        if image is not None:
            body["image_base64"] = encode_chunk_image(image)

        data = await self._request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            json=body,
        )
        chunk = (data or {}).get("chunk") if isinstance(data, dict) else None
        if not isinstance(chunk, dict) or not chunk.get("id"):
            raise RAGFlowError(f"chunk creation returned no id: {data!r}")
        return ChunkRef(chunk_id=str(chunk["id"]), image_id=str(chunk.get("image_id") or "") or None)

    async def update_chunk(
        self,
        dataset_id: str,
        document_id: str,
        chunk_id: str,
        *,
        content: str | None = None,
        important_keywords: Sequence[str] | None = None,
        questions: Sequence[str] | None = None,
        tag_keywords: Sequence[str] | None = None,
        image: bytes | None = None,
    ) -> None:
        """Update one chunk in place.

        The id does not move: RAGFlow updates the row under the id it was given
        and does not recompute the hash, so after changing ``content`` the id no
        longer matches ``xxhash64(content + document_id)``. That is why the write
        path identifies chunks by looking them up rather than by recomputing.

        Passing ``image`` re-runs the same concatenating store, so it is only safe
        when nothing is stored under this chunk's key yet - which is not the same
        as the chunk having no image, because a deleted or re-parsed chunk leaves
        its object behind. :meth:`upsert_chunk` checks before it patches.
        """
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if important_keywords is not None:
            body["important_keywords"] = list(important_keywords)
        if questions is not None:
            body["questions"] = list(questions)
        if tag_keywords is not None:
            body["tag_kwd"] = list(tag_keywords)
        if image is not None:
            body["image_base64"] = encode_chunk_image(image)
        if not body:
            return
        await self._request(
            "PATCH",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}",
            json=body,
        )

    async def delete_chunks(self, dataset_id: str, document_id: str, chunk_ids: Sequence[str]) -> None:
        """Delete specific chunks of a document.

        Scoped like :meth:`delete_documents`: explicit IDs only, and an empty list
        is a no-op rather than "delete everything". RAGFlow does have a
        ``delete_all`` flag on this route; it is exposed separately as
        :meth:`delete_all_chunks` so it can never be reached by accident.

        Note that this removes the chunk rows only. Their stored images stay in
        MinIO, keyed by chunk id.
        """
        ids = [str(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return
        await self._request(
            "DELETE",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"chunk_ids": ids},
        )

    async def delete_all_chunks(self, dataset_id: str, document_id: str) -> None:
        """Delete every chunk of one document. Parsed chunks included.

        Exposed separately from :meth:`delete_chunks` so the destructive flag
        cannot be reached by passing an empty list.

        RAGFlow calls ``delete_chunk_images`` on this path, which reads as "the
        images go too". On this deployment they do not: after this call the image
        was still served, and the next add of the same content stacked onto it.
        So this is not a way out of a stacked image; changing the content or
        re-uploading the document is, because both move the chunk id.
        """
        await self._request(
            "DELETE",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            json={"delete_all": True},
        )

    async def get_chunk(self, dataset_id: str, document_id: str, chunk_id: str) -> dict[str, Any] | None:
        """Read one chunk by id, or ``None`` when it does not exist.

        This is the *current* state, which the paged listing is not. RAGFlow
        fetches by id straight from the doc store, while the listing goes through
        search: an ``add`` is visible to both at once because the insert waits for
        a refresh, but a ``PATCH`` is not - the field it changed took ~830 ms to
        appear in the listing on this deployment, and it was there by id
        immediately. So anything that reads a field a PATCH may have just written,
        such as ``img_id``, has to read it here.

        Field names are RAGFlow's internal ones on this route
        (``content_with_weight``, ``img_id``), not the renamed public ones.
        """
        try:
            data = await self._request(
                "GET",
                f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}",
            )
        except RAGFlowError as exc:
            if isinstance(exc, RAGFlowUnavailable) or exc.code != 102:
                raise
            # 102 on this route means "no such chunk", the same overloaded code
            # RAGFlow uses for a dataset-name miss.
            return None
        return data if isinstance(data, dict) else None

    async def fetch_chunk_image(self, image_id: str) -> bytes | None:
        """Read a stored chunk image, or ``None`` when nothing is stored there.

        Answers outside the ``{code, data}`` envelope: a hit is the raw image
        body, a miss is a JSON error with HTTP 200, so the content type decides.

        Two uses. It proves what was actually stored - the bytes posted, or a
        re-encoded JPEG stack - and it tells :meth:`upsert_chunk` whether an
        object key is free before anything is written to it.

        This route needs a RAGFlow credential, so a browser cannot call it. The
        UI is served screenshots by our own API instead.
        """
        if not image_id:
            return None
        client = self._ensure_client()
        url = f"{self._base_url}/api/v1/documents/images/{image_id}"
        try:
            response = await client.get(url, headers=self._auth_headers())
        except httpx.TimeoutException as exc:
            raise RAGFlowUnavailable(f"RAGFlow timed out after {self._timeout}s", path=url) from exc
        except httpx.HTTPError as exc:
            raise RAGFlowUnavailable(f"RAGFlow is unreachable: {exc}", path=url) from exc

        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise RAGFlowError("RAGFlow rejected the API key", code=401, path=url)
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise RAGFlowUnavailable(
                f"RAGFlow returned HTTP {response.status_code}", code=response.status_code, path=url
            )
        if "application/json" in response.headers.get("content-type", ""):
            return None
        return response.content or None

    async def upsert_chunk(
        self,
        dataset_id: str,
        document_id: str,
        *,
        content: str,
        identity_prefix: str,
        important_keywords: Sequence[str] | None = None,
        questions: Sequence[str] | None = None,
        tag_keywords: Sequence[str] | None = None,
        image: bytes | None = None,
    ) -> ChunkUpsert:
        """Write one chunk so that running it again changes nothing.

        ``identity_prefix`` names the logical chunk: every chunk of this document
        whose content starts with it is a version of the same thing. The caller
        owns that uniqueness - for visual steps it is the procedure id plus the
        step id, which the chunk text carries anyway.

        The decision is made from RAGFlow's own state, not from a local record:

        * identical content already present -> nothing is posted, so the stored
          image cannot be stacked onto itself;
        * identical content present but carrying no image -> only the image is
          attached, which converges a half-finished earlier run;
        * content changed -> the new chunk is added and the older versions of the
          same logical chunk are deleted, so a superseded step does not linger as
          a duplicate that retrieval can still return.

        An image is never posted in the same call that creates the chunk. The
        chunk is written text-first, and the image follows only once the object
        key is known to be free (see fact 3 above: a re-parse leaves an orphaned
        image behind whose key a later identical add would stack onto). The extra
        round trip costs one more embedding call per new chunk, which is
        ingestion-time only. When the key is not free the text still lands, the
        image is skipped rather than corrupted, and ``image_blocked`` says so; the
        fix is to re-upload the document, which moves the chunk id to a clean key.

        Identity is resolved by lookup rather than by recomputing
        ``xxhash64(content + document_id)`` locally. Reasons, in order of weight:
        the local hash gives the id of the chunk about to be written and says
        nothing about the superseded version that has to be deleted, which is the
        half of the problem that actually needs solving; reproducing it would add
        a native dependency for a RAGFlow implementation detail this code would
        then be pinned to; and the lookup returns the image state too, which the
        hash cannot. The doc store inserts with ``refresh="wait_for"`` and deletes
        with ``refresh=True``, so a write is visible to the next lookup - measured
        at 11 ms, first attempt, against the live stack. Concurrent writers to one
        document are still not safe; this is an ingestion path, run one at a time.
        """
        existing = await self.list_chunks(dataset_id, document_id)
        chunks = existing.get("chunks") if isinstance(existing, dict) else None
        siblings = [
            chunk
            for chunk in (chunks or [])
            if isinstance(chunk, dict) and str(chunk.get("content") or "").startswith(identity_prefix)
        ]
        match = next((chunk for chunk in siblings if str(chunk.get("content") or "") == content), None)
        superseded = tuple(str(chunk["id"]) for chunk in siblings if chunk.get("id") and chunk is not match)

        image_blocked = False
        if match is not None:
            chunk_id = str(match["id"])
            # Read the image state by id, not from the listing above: a PATCH from
            # an earlier run is visible by id at once but reaches the listing only
            # after the doc store refreshes, and believing the listing would mean
            # re-attaching an image that is already there.
            current = await self.get_chunk(dataset_id, document_id, chunk_id) or {}
            image_id = str(current.get("img_id") or match.get("image_id") or "") or None
            action = "unchanged"
            if image is not None and image_id is None:
                image_id, image_blocked = await self._attach_image(dataset_id, document_id, chunk_id, image)
                action = "image_attached" if image_id else "unchanged"
        else:
            ref = await self.add_chunk(
                dataset_id,
                document_id,
                content=content,
                important_keywords=important_keywords,
                questions=questions,
                tag_keywords=tag_keywords,
            )
            chunk_id, image_id, action = ref.chunk_id, ref.image_id, "created"
            if image is not None:
                image_id, image_blocked = await self._attach_image(dataset_id, document_id, chunk_id, image)

        # Deleted last, and only once the replacement is in place: a failed write
        # then leaves the previous version standing rather than no chunk at all.
        await self.delete_chunks(dataset_id, document_id, superseded)

        logger.debug(
            "ragflow.chunk_upsert",
            action=action,
            chunk_id=chunk_id,
            superseded=len(superseded),
            image_blocked=image_blocked,
        )
        return ChunkUpsert(
            chunk_id=chunk_id,
            image_id=image_id,
            action=action,
            superseded_chunk_ids=superseded,
            image_blocked=image_blocked,
        )

    async def _attach_image(
        self, dataset_id: str, document_id: str, chunk_id: str, image: bytes
    ) -> tuple[str | None, bool]:
        """Attach an image to an existing chunk, unless its object key is taken.

        Returns ``(image_id, blocked)``. ``blocked`` means an object was already
        stored under this chunk's key - left over from a chunk that was deleted or
        re-parsed away - so writing would have concatenated onto it. Nothing is
        written in that case: a chunk without an image is recoverable, a stacked
        JPEG is not.
        """
        image_id = f"{dataset_id}-{chunk_id}"
        if await self.fetch_chunk_image(image_id) is not None:
            logger.warning(
                "ragflow.chunk_image_key_occupied",
                chunk_id=chunk_id,
                reason="an image is already stored under this chunk id; writing would stack onto it",
                remedy="re-upload the document so the chunk id moves to a free key",
            )
            return None, True

        await self.update_chunk(dataset_id, document_id, chunk_id, image=image)
        # Confirm from RAGFlow rather than assuming: the id is predictable, but
        # whether the field landed is not something to take on trust.
        stored = await self.get_chunk(dataset_id, document_id, chunk_id) or {}
        recorded = str(stored.get("img_id") or "") or None
        if recorded is None:
            logger.warning("ragflow.chunk_image_not_recorded", chunk_id=chunk_id)
        return recorded, False

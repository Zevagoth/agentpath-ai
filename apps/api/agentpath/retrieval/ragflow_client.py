"""Async HTTP client for the RAGFlow API.

One place owns the RAGFlow wire protocol: authentication, the ``{code, data,
message}`` envelope, timeouts and error translation. Both the retriever and the
ingestion service use it, so a change in RAGFlow's API surface has exactly one
landing site.

Reference: RAGFlow v0.27.2 HTTP API. Verified against the pinned checkout in
``.external/ragflow`` rather than assumed from documentation.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any, Self

import httpx
import structlog

logger = structlog.get_logger(__name__)

#: RAGFlow returns HTTP 200 with a business code in the body. 0 means success.
SUCCESS_CODE = 0


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
            raise RAGFlowError(
                f"RAGFlow returned a non-JSON body (HTTP {response.status_code})", path=path
            ) from exc

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
        """System health. Also the cheapest way to verify the API key works."""
        return await self._request("GET", "/api/v1/system/healthz")

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

    async def list_chunks(self, dataset_id: str, document_id: str, *, page_size: int = 100) -> dict[str, Any]:
        data = await self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            params={"page": 1, "page_size": page_size},
        )
        return data or {}

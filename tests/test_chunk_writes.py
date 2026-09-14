"""Tests for RAGFlow chunk writes, including image-bearing chunks.

The unit tests run against recorded payloads with ``respx``, so they need no
stack. The shapes and the quirks they encode were observed against the live
RAGFlow v0.27.2 deployment, not invented:

* an image posted twice for the same content comes back as a JPEG twice as tall,
  because RAGFlow concatenates onto the existing object;
* re-parsing a document deletes manually added chunks but leaves their images, so
  a later identical add would stack onto an orphan;
* a field set by ``PATCH`` is readable by chunk id immediately but reaches the
  paged listing only after the doc store refreshes (~830 ms measured).

One integration test at the end proves the whole thing against the real stack
with a PNG generated in the test.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
import zlib

import httpx
import pytest
import respx

from agentpath.retrieval.ragflow_client import (
    ChunkUpsert,
    RAGFlowClient,
    RAGFlowError,
    RAGFlowUnavailable,
    encode_chunk_image,
)
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever

BASE_URL = "http://ragflow.test:9380"
API_KEY = "ragflow-test-key"
DATASET = "dataset-1"
DOCUMENT = "doc-1"

CHUNKS_URL = f"{BASE_URL}/api/v1/datasets/{DATASET}/documents/{DOCUMENT}/chunks"
IDENTITY = "NW-HTL-014 step 1"
CONTENT = f"{IDENTITY} - screen: agentdesk-booking-detail-status. Open the booking and read the status banner."


def synthetic_png(width: int = 16, height: int = 8, rgb: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    """A real, valid PNG built here rather than committed as a fixture.

    Small enough to inline in a request assertion, and a genuine PNG so RAGFlow's
    image pipeline treats it as one.
    """
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _added(chunk_id: str = "abc123", *, image: bool = False) -> dict:
    """RAGFlow's response to a successful add: ``img_id`` renamed to ``image_id``."""
    chunk = {
        "id": chunk_id,
        "content": CONTENT,
        "document_id": DOCUMENT,
        "dataset_id": DATASET,
        "doc_type_kwd": "image" if image else "text",
    }
    if image:
        chunk["image_id"] = f"{DATASET}-{chunk_id}"
    return {"code": 0, "data": {"chunk": chunk}}


def _listing(*chunks: dict) -> dict:
    return {"code": 0, "data": {"total": len(chunks), "chunks": list(chunks)}}


def _listed(chunk_id: str, content: str, *, image_id: str = "") -> dict:
    return {
        "id": chunk_id,
        "content": content,
        "document_id": DOCUMENT,
        "dataset_id": DATASET,
        "image_id": image_id,
        "doc_type_kwd": "image" if image_id else "text",
        "important_keywords": [],
        "questions": [],
    }


def _stored(chunk_id: str, content: str, *, image_id: str = "") -> dict:
    """The by-id read, which answers with RAGFlow's internal field names."""
    return {
        "code": 0,
        "data": {
            "id": chunk_id,
            "content_with_weight": content,
            "doc_id": DOCUMENT,
            "kb_id": DATASET,
            "img_id": image_id,
            "doc_type_kwd": "image" if image_id else "text",
        },
    }


def _no_such_chunk() -> dict:
    return {"code": 102, "message": "Chunk not found!", "data": False}


def _image_response(payload: bytes) -> httpx.Response:
    return httpx.Response(200, content=payload, headers={"Content-Type": "image/png"})


def _no_image_response() -> httpx.Response:
    """A missing object is an error envelope inside HTTP 200, not a 404."""
    return httpx.Response(
        200,
        json={"code": 102, "message": "Image not found."},
        headers={"Content-Type": "application/json"},
    )


# --------------------------------------------------------------------------- #
# Encoding: what RAGFlow will and will not accept
# --------------------------------------------------------------------------- #


def test_image_is_encoded_as_plain_standard_base64() -> None:
    png = synthetic_png()
    encoded = encode_chunk_image(png)
    assert base64.b64decode(encoded, validate=True) == png
    assert not encoded.startswith("data:")


def test_a_data_uri_is_refused_with_an_explanation() -> None:
    """RAGFlow decodes with validate=True, so the prefix fails server-side."""
    with pytest.raises(ValueError, match="data URI"):
        encode_chunk_image(b"data:image/png;base64," + base64.b64encode(synthetic_png()))


def test_a_pre_encoded_string_is_refused() -> None:
    with pytest.raises(ValueError, match="raw bytes"):
        encode_chunk_image(base64.b64encode(synthetic_png()).decode("ascii"))  # type: ignore[arg-type]


def test_an_empty_image_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_chunk_image(b"")


@respx.mock
async def test_nothing_is_posted_when_the_image_is_rejected() -> None:
    route = respx.post(CHUNKS_URL)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(ValueError):
            await client.add_chunk(DATASET, DOCUMENT, content=CONTENT, image=b"data:image/png;base64,AAAA")
    assert not route.called


# --------------------------------------------------------------------------- #
# add / update / delete request shape
# --------------------------------------------------------------------------- #


@respx.mock
async def test_add_chunk_sends_the_image_and_returns_both_ids() -> None:
    png = synthetic_png()
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_added("abc123", image=True))

    respx.post(CHUNKS_URL).mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        ref = await client.add_chunk(
            DATASET,
            DOCUMENT,
            content=CONTENT,
            important_keywords=["overbooking"],
            questions=["what if the hotel has no room?"],
            tag_keywords=["hotels"],
            image=png,
        )

    assert captured["content"] == CONTENT
    assert captured["important_keywords"] == ["overbooking"]
    assert captured["questions"] == ["what if the hotel has no room?"]
    assert captured["tag_kwd"] == ["hotels"]
    assert base64.b64decode(captured["image_base64"], validate=True) == png
    assert ref.chunk_id == "abc123"
    assert ref.image_id == f"{DATASET}-abc123"


@respx.mock
async def test_add_chunk_omits_fields_that_were_not_given() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_added())

    respx.post(CHUNKS_URL).mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        ref = await client.add_chunk(DATASET, DOCUMENT, content=CONTENT)

    assert captured == {"content": CONTENT}
    assert ref.image_id is None


@respx.mock
async def test_add_chunk_without_an_id_in_the_response_is_an_error() -> None:
    respx.post(CHUNKS_URL).mock(return_value=httpx.Response(200, json={"code": 0, "data": {}}))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowError, match="no id"):
            await client.add_chunk(DATASET, DOCUMENT, content=CONTENT)


@respx.mock
async def test_add_chunk_maps_a_business_error_inside_http_200() -> None:
    respx.post(CHUNKS_URL).mock(
        return_value=httpx.Response(200, json={"code": 102, "message": "`content` is required"})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowError, match="`content` is required"):
            await client.add_chunk(DATASET, DOCUMENT, content=" ")


@respx.mock
async def test_add_chunk_reports_an_unreachable_service_distinctly() -> None:
    respx.post(CHUNKS_URL).mock(side_effect=httpx.ConnectError("refused"))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowUnavailable):
            await client.add_chunk(DATASET, DOCUMENT, content=CONTENT)


@respx.mock
async def test_update_chunk_patches_only_the_given_fields() -> None:
    png = synthetic_png()
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0})

    respx.patch(f"{CHUNKS_URL}/abc123").mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.update_chunk(DATASET, DOCUMENT, "abc123", image=png)

    assert set(captured) == {"image_base64"}
    assert base64.b64decode(captured["image_base64"], validate=True) == png


@respx.mock
async def test_update_chunk_with_nothing_to_change_makes_no_request() -> None:
    route = respx.patch(f"{CHUNKS_URL}/abc123")
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.update_chunk(DATASET, DOCUMENT, "abc123")
    assert not route.called


@respx.mock
async def test_delete_chunks_sends_explicit_ids() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "message": "deleted 1 chunks"})

    respx.delete(CHUNKS_URL).mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.delete_chunks(DATASET, DOCUMENT, ["abc123"])
    assert captured == {"chunk_ids": ["abc123"]}


@respx.mock
async def test_delete_chunks_is_a_noop_without_ids() -> None:
    """An empty list must never turn into RAGFlow's delete_all."""
    route = respx.delete(CHUNKS_URL)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.delete_chunks(DATASET, DOCUMENT, [])
    assert not route.called


@respx.mock
async def test_delete_all_chunks_is_a_separate_explicit_call() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "message": "deleted 7 chunks"})

    respx.delete(CHUNKS_URL).mock(side_effect=_handler)
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.delete_all_chunks(DATASET, DOCUMENT)
    assert captured == {"delete_all": True}


# --------------------------------------------------------------------------- #
# Reads used by the write path
# --------------------------------------------------------------------------- #


@respx.mock
async def test_get_chunk_returns_the_current_row() -> None:
    respx.get(f"{CHUNKS_URL}/abc123").mock(
        return_value=httpx.Response(200, json=_stored("abc123", CONTENT, image_id=f"{DATASET}-abc123"))
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        chunk = await client.get_chunk(DATASET, DOCUMENT, "abc123")
    assert chunk is not None
    assert chunk["img_id"] == f"{DATASET}-abc123"


@respx.mock
async def test_get_chunk_treats_the_data_error_code_as_absent() -> None:
    respx.get(f"{CHUNKS_URL}/gone").mock(return_value=httpx.Response(200, json=_no_such_chunk()))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.get_chunk(DATASET, DOCUMENT, "gone") is None


@respx.mock
async def test_get_chunk_still_raises_on_other_errors() -> None:
    respx.get(f"{CHUNKS_URL}/abc123").mock(
        return_value=httpx.Response(200, json={"code": 100, "message": "index missing"})
    )
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        with pytest.raises(RAGFlowError, match="index missing"):
            await client.get_chunk(DATASET, DOCUMENT, "abc123")


@respx.mock
async def test_fetch_chunk_image_returns_the_bytes() -> None:
    png = synthetic_png()
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-abc123").mock(return_value=_image_response(png))
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.fetch_chunk_image(f"{DATASET}-abc123") == png


@respx.mock
async def test_fetch_chunk_image_reads_a_json_error_as_no_image() -> None:
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-abc123").mock(return_value=_no_image_response())
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.fetch_chunk_image(f"{DATASET}-abc123") is None


@respx.mock
async def test_fetch_chunk_image_without_an_id_makes_no_request() -> None:
    route = respx.get(url__startswith=f"{BASE_URL}/api/v1/documents/images/")
    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        assert await client.fetch_chunk_image("") is None
    assert not route.called


# --------------------------------------------------------------------------- #
# Idempotence
# --------------------------------------------------------------------------- #


@respx.mock
async def test_upsert_of_unchanged_content_posts_nothing() -> None:
    """The core guarantee: a second run must not touch the stored image.

    Re-posting identical content is exactly what stacks a screenshot onto itself,
    so the no-op has to be a genuine absence of a write, not a rewrite.
    """
    respx.get(CHUNKS_URL).mock(
        return_value=httpx.Response(200, json=_listing(_listed("abc123", CONTENT, image_id=f"{DATASET}-abc123")))
    )
    respx.get(f"{CHUNKS_URL}/abc123").mock(
        return_value=httpx.Response(200, json=_stored("abc123", CONTENT, image_id=f"{DATASET}-abc123"))
    )
    post = respx.post(CHUNKS_URL)
    patch = respx.patch(url__startswith=f"{CHUNKS_URL}/")
    delete = respx.delete(CHUNKS_URL)

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert result == ChunkUpsert(chunk_id="abc123", image_id=f"{DATASET}-abc123", action="unchanged")
    assert not post.called
    assert not patch.called
    assert not delete.called


@respx.mock
async def test_upsert_trusts_the_by_id_read_over_a_stale_listing() -> None:
    """A PATCH-set ``img_id`` reaches the listing late; believing it re-attaches.

    Measured at ~830 ms on the live stack, which is easily inside one ingestion
    run, and the consequence would be a stacked image.
    """
    respx.get(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_listing(_listed("abc123", CONTENT, image_id=""))))
    respx.get(f"{CHUNKS_URL}/abc123").mock(
        return_value=httpx.Response(200, json=_stored("abc123", CONTENT, image_id=f"{DATASET}-abc123"))
    )
    patch = respx.patch(f"{CHUNKS_URL}/abc123")

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert result.action == "unchanged"
    assert result.image_id == f"{DATASET}-abc123"
    assert not patch.called


@respx.mock
async def test_upsert_attaches_a_missing_image_when_the_key_is_free() -> None:
    """Converges a run that wrote the text and then died before the image."""
    respx.get(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_listing(_listed("abc123", CONTENT))))
    respx.get(f"{CHUNKS_URL}/abc123").side_effect = [
        httpx.Response(200, json=_stored("abc123", CONTENT)),
        httpx.Response(200, json=_stored("abc123", CONTENT, image_id=f"{DATASET}-abc123")),
    ]
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-abc123").mock(return_value=_no_image_response())
    patch = respx.patch(f"{CHUNKS_URL}/abc123").mock(return_value=httpx.Response(200, json={"code": 0}))
    post = respx.post(CHUNKS_URL)

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert result.action == "image_attached"
    assert result.image_id == f"{DATASET}-abc123"
    assert result.image_blocked is False
    assert patch.called
    assert not post.called


@respx.mock
async def test_upsert_refuses_to_write_onto_an_occupied_image_key() -> None:
    """The re-parse orphan: the chunk row is gone, its image is not.

    RAGFlow would concatenate the new screenshot onto the old object and re-encode
    the pair as JPEG. Skipping the image leaves a recoverable state instead.
    """
    respx.get(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_listing()))
    respx.post(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_added("abc123")))
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-abc123").mock(
        return_value=_image_response(synthetic_png())
    )
    patch = respx.patch(f"{CHUNKS_URL}/abc123")

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert result.action == "created"
    assert result.image_blocked is True
    assert result.image_id is None
    assert not patch.called


@respx.mock
async def test_upsert_never_posts_the_image_with_the_content() -> None:
    """The key can only be checked once the id is known, so text goes first."""
    posted: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        posted.update(json.loads(request.content))
        return httpx.Response(200, json=_added("abc123"))

    respx.get(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_listing()))
    respx.post(CHUNKS_URL).mock(side_effect=_handler)
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-abc123").mock(return_value=_no_image_response())
    respx.patch(f"{CHUNKS_URL}/abc123").mock(return_value=httpx.Response(200, json={"code": 0}))
    respx.get(f"{CHUNKS_URL}/abc123").mock(
        return_value=httpx.Response(200, json=_stored("abc123", CONTENT, image_id=f"{DATASET}-abc123"))
    )

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert "image_base64" not in posted
    assert result.action == "created"
    assert result.image_id == f"{DATASET}-abc123"


@respx.mock
async def test_upsert_replaces_a_superseded_version_of_the_same_step() -> None:
    """Changed content means a new chunk id, so the old one has to go.

    Left alone it stays retrievable, and the answer would cite a step that no
    longer exists.
    """
    old_content = f"{IDENTITY} - screen: agentdesk-booking-detail-status. Older wording."
    deleted: dict = {}

    def _delete(request: httpx.Request) -> httpx.Response:
        deleted.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "message": "deleted 1 chunks"})

    respx.get(CHUNKS_URL).mock(
        return_value=httpx.Response(
            200,
            json=_listing(
                _listed("old111", old_content, image_id=f"{DATASET}-old111"),
                _listed("other", "NW-HTL-014 step 4 - a different step, untouched"),
            ),
        )
    )
    respx.post(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_added("new222")))
    respx.get(f"{BASE_URL}/api/v1/documents/images/{DATASET}-new222").mock(return_value=_no_image_response())
    respx.patch(f"{CHUNKS_URL}/new222").mock(return_value=httpx.Response(200, json={"code": 0}))
    respx.get(f"{CHUNKS_URL}/new222").mock(
        return_value=httpx.Response(200, json=_stored("new222", CONTENT, image_id=f"{DATASET}-new222"))
    )
    respx.delete(CHUNKS_URL).mock(side_effect=_delete)

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(
            DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY, image=synthetic_png()
        )

    assert result.action == "created"
    assert result.chunk_id == "new222"
    assert result.superseded_chunk_ids == ("old111",)
    # Only the earlier version of this step, never the neighbouring step.
    assert deleted == {"chunk_ids": ["old111"]}


@respx.mock
async def test_upsert_deletes_the_old_chunk_only_after_the_new_one_lands() -> None:
    """A failed add must leave the previous version standing."""
    order: list[str] = []

    def _record(name: str, response: httpx.Response):
        def _handler(request: httpx.Request) -> httpx.Response:
            order.append(name)
            return response

        return _handler

    respx.get(CHUNKS_URL).mock(
        return_value=httpx.Response(200, json=_listing(_listed("old111", f"{IDENTITY} - older wording")))
    )
    respx.post(CHUNKS_URL).mock(side_effect=_record("add", httpx.Response(200, json=_added("new222"))))
    respx.delete(CHUNKS_URL).mock(
        side_effect=_record("delete", httpx.Response(200, json={"code": 0, "message": "deleted 1 chunks"}))
    )

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        await client.upsert_chunk(DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY)

    assert order == ["add", "delete"]


@respx.mock
async def test_upsert_without_an_image_makes_one_write_and_no_image_calls() -> None:
    respx.get(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_listing()))
    post = respx.post(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_added("abc123")))
    images = respx.get(url__startswith=f"{BASE_URL}/api/v1/documents/images/")

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY)

    assert result.action == "created"
    assert result.image_id is None
    assert post.call_count == 1
    assert not images.called


@respx.mock
async def test_upsert_ignores_chunks_outside_its_identity() -> None:
    """Parsed document chunks must not be mistaken for an earlier version."""
    respx.get(CHUNKS_URL).mock(
        return_value=httpx.Response(
            200,
            json=_listing(
                _listed("parsed1", "## Steps\n1. Open the booking..."),
                _listed("parsed2", "## Escalation\nContact the supplier desk."),
            ),
        )
    )
    respx.post(CHUNKS_URL).mock(return_value=httpx.Response(200, json=_added("abc123")))
    delete = respx.delete(CHUNKS_URL)

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        result = await client.upsert_chunk(DATASET, DOCUMENT, content=CONTENT, identity_prefix=IDENTITY)

    assert result.superseded_chunk_ids == ()
    assert not delete.called


# --------------------------------------------------------------------------- #
# The retrieval path already carries the image id
# --------------------------------------------------------------------------- #


@respx.mock
async def test_an_image_bearing_chunk_reaches_passage_image_id() -> None:
    """No mapping work is needed for T5: the retriever already passes it through."""
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": [{"id": DATASET, "name": "agentpath-northwind-procedures"}],
            },
        )
    )
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "chunks": [
                        {
                            "id": "abc123",
                            "content": CONTENT,
                            "document_id": DOCUMENT,
                            "document_keyword": "NW-HTL-014.md",
                            "dataset_id": DATASET,
                            "similarity": 0.61,
                            "image_id": f"{DATASET}-abc123",
                            "doc_type_kwd": "image",
                            "document_metadata": {"procedure_id": "NW-HTL-014", "version": "2.1"},
                        }
                    ],
                    "total": 1,
                },
            },
        )
    )

    from agentpath.domain import Language, PreparedQuery

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        retriever = RAGFlowRetriever(client, dataset_name="agentpath-northwind-procedures")
        result = await retriever.retrieve(
            PreparedQuery(
                original="hotel has no room",
                normalized="hotel has no room",
                language=Language.ENGLISH,
                search_text="hotel has no room",
            ),
            limit=3,
        )

    assert result.passages[0].image_id == f"{DATASET}-abc123"


@respx.mock
async def test_a_text_chunk_carries_no_image_id() -> None:
    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(
            200,
            json={"code": 0, "data": [{"id": DATASET, "name": "agentpath-northwind-procedures"}]},
        )
    )
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "chunks": [
                        {
                            "id": "text1",
                            "content": CONTENT,
                            "document_id": DOCUMENT,
                            "document_keyword": "NW-HTL-014.md",
                            "similarity": 0.5,
                            "image_id": "",
                        }
                    ]
                },
            },
        )
    )

    from agentpath.domain import Language, PreparedQuery

    async with RAGFlowClient(BASE_URL, API_KEY) as client:
        retriever = RAGFlowRetriever(client, dataset_name="agentpath-northwind-procedures")
        result = await retriever.retrieve(
            PreparedQuery(original="q", normalized="q", language=Language.ENGLISH, search_text="q"),
            limit=3,
        )

    assert result.passages[0].image_id is None


# --------------------------------------------------------------------------- #
# Live stack
# --------------------------------------------------------------------------- #

PROBE_DATASET = "agentpath-t4-integration-throwaway"
PROBE_DOCUMENT = "T4-INTEGRATION.md"
PROBE_BODY = b"""# T4-INTEGRATION - throwaway probe document

## Steps
1. Open the probe console and confirm the reference.
2. Close the probe.
"""
PROBE_IDENTITY = "T4-INTEGRATION step 1"
PROBE_CONTENT = (
    f"{PROBE_IDENTITY} - screen: probe-console. Control: Confirm reference button. "
    "Synthetic demo data, written by the integration test."
)


@pytest.mark.integration
async def test_an_image_bearing_chunk_survives_a_round_trip_on_the_live_stack() -> None:
    """Proves ``image_base64`` works on this deployment, end to end.

    Everything happens in a dataset created and destroyed by the test, so the
    development corpus is never touched. The PNG is generated here rather than
    committed, and the assertions are about what RAGFlow actually stored: the
    bytes posted, byte for byte, and no second copy after a second run.
    """
    from agentpath.config import get_settings

    settings = get_settings()
    if not settings.has_ragflow_credentials:
        pytest.skip("no RAGFlow credentials configured")

    png = synthetic_png(40, 20, (10, 160, 90))
    dataset_id = None

    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=60.0,
    ) as client:
        try:
            await client.healthz()
        except RAGFlowUnavailable:
            pytest.skip("RAGFlow is not running")

        try:
            stale = await client.find_dataset_id(PROBE_DATASET)
            if stale:
                await _delete_dataset(client, stale)
            dataset_id = await client.create_dataset(
                name=PROBE_DATASET,
                description="Throwaway dataset for the T4 integration test. Safe to delete.",
                embedding_model=settings.ragflow_embedding_model,
                chunk_method="naive",
                parser_config={"chunk_token_num": 512, "delimiter": "\n", "layout_recognize": "Plain Text"},
            )
            document_id = await client.upload_document(dataset_id, PROBE_DOCUMENT, PROBE_BODY)
            await client.parse_documents(dataset_id, [document_id])
            assert await _wait_parsed(client, dataset_id, document_id), "probe document did not parse"

            first = await client.upsert_chunk(
                dataset_id,
                document_id,
                content=PROBE_CONTENT,
                identity_prefix=PROBE_IDENTITY,
                image=png,
            )
            assert first.action == "created"
            assert first.image_blocked is False
            assert first.image_id == f"{dataset_id}-{first.chunk_id}"

            row = await client.get_chunk(dataset_id, document_id, first.chunk_id)
            assert row is not None
            assert row["img_id"] == first.image_id
            assert row["doc_type_kwd"] == "image"

            stored = await client.fetch_chunk_image(first.image_id or "")
            assert stored == png, "RAGFlow stored something other than the bytes that were posted"

            # Retrieval is the path the answer service will use, and it has to
            # carry the image handle all the way into Passage.image_id.
            retriever = RAGFlowRetriever(client, dataset_name=PROBE_DATASET, similarity_threshold=0.0)
            from agentpath.domain import Language, PreparedQuery

            prepared = PreparedQuery(
                original="probe console confirm reference button",
                normalized="probe console confirm reference button",
                language=Language.ENGLISH,
                search_text="probe console confirm reference button",
            )
            image_ids: set[str | None] = set()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                result = await retriever.retrieve(prepared, limit=3)
                image_ids = {passage.image_id for passage in result.passages if passage.chunk_id == first.chunk_id}
                if image_ids and None not in image_ids:
                    break
                await asyncio.sleep(0.5)
            assert image_ids == {first.image_id}

            # Second run: the guarantee that matters. Nothing is posted, so the
            # stored screenshot cannot be concatenated onto itself.
            second = await client.upsert_chunk(
                dataset_id,
                document_id,
                content=PROBE_CONTENT,
                identity_prefix=PROBE_IDENTITY,
                image=png,
            )
            assert second.action == "unchanged"
            assert second.chunk_id == first.chunk_id
            assert second.image_blocked is False
            assert await client.fetch_chunk_image(first.image_id or "") == png

            listing = await client.list_chunks(dataset_id, document_id)
            manual = [
                chunk
                for chunk in listing.get("chunks") or []
                if str(chunk.get("content") or "").startswith(PROBE_IDENTITY)
            ]
            assert len(manual) == 1, "the second run duplicated the chunk"
        finally:
            if dataset_id:
                await _delete_dataset(client, dataset_id)
                assert await client.find_dataset_id(PROBE_DATASET) is None


#: Parsing one three-line document takes a couple of seconds warm; the ceiling is
#: for a cold embedder, and a miss fails the test rather than hanging it.
PARSE_DEADLINE_SECONDS = 240.0


async def _wait_parsed(client: RAGFlowClient, dataset_id: str, document_id: str) -> bool:
    deadline = time.monotonic() + PARSE_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        for document in await client.list_documents(dataset_id):
            if document.get("id") == document_id:
                status = str(document.get("run") or "")
                if status in {"DONE", "3"}:
                    return True
                if status in {"FAIL", "4"}:
                    return False
        await asyncio.sleep(2.0)
    return False


async def _delete_dataset(client: RAGFlowClient, dataset_id: str) -> None:
    """Dataset deletion is not part of the client's surface, and stays that way.

    Nothing in AgentPath deletes a dataset; only this test does, on a dataset it
    created itself. Exposing it on the client would put "drop the corpus" one
    typo away from the ingestion path.
    """
    await client._request("DELETE", "/api/v1/datasets", json={"ids": [dataset_id]})

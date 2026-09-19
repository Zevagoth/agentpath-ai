"""Opening the original: the source viewer and the corpus image route.

Built on a bundle written into `tmp_path`, so these run with no fetched corpus and
no network. The bundle is written by the real exporter rather than by hand, which
means the test also covers the ingestion-to-API file contract: if the exporter and
the reader ever disagree about a field name, this fails.

Weighted towards refusal, because the whole point of this route pair is that a
third-party corpus can be opened safely.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agentpath.api.app import create_app
from agentpath.api.deps import ServiceContext
from agentpath.config import Settings
from agentpath.corpus import CorpusCatalogue
from agentpath.query.lexicon import load_lexicon
from agentpath.search_service import SearchService
from agentpath_ingestion.corpora.base import (
    CorpusBundle,
    CorpusDocument,
    CorpusImage,
    CorpusQuestion,
    slice_chunks,
)
from agentpath_ingestion.corpora.export import export_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]

# A 1x1 PNG. Real bytes, so the route serves something a browser would accept.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63f8ffff3f0005fe02fea735d1c2"
    "0000000049454e44ae426082"
)


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    """A complete exported corpus on disk, images included."""
    images = tmp_path / "demo" / "raw" / "images"
    images.mkdir(parents=True)
    (images / "Demo_0.png").write_bytes(PNG)
    (images / "Demo_1.png").write_bytes(PNG)

    text = (
        "# Operating instructions\n\n"
        "Before starting, check the coolant level as shown.\n\n"
        "<PIC>\n\n"
        "Then close the access panel and secure both latches.\n\n"
        "<PIC>\n\n"
        "The machine is now ready for use.\n"
    )
    offsets = [index for index in range(len(text)) if text.startswith("<PIC>", index)]
    document = CorpusDocument(
        document_id="900",
        title="Operating instructions",
        text=text,
        source_id="demo-corpus-900",
        images=(
            CorpusImage(
                image_id="Demo_0",
                document_id="900",
                position_index=0,
                char_offset=offsets[0],
                source_url="https://cdn.example.test/Demo_0.png",
            ),
            CorpusImage(
                image_id="Demo_1",
                document_id="900",
                position_index=1,
                char_offset=offsets[1],
                # No upstream URL, like 299 of MRAMG-Manual's images.
                source_url=None,
            ),
        ),
    )
    bundle = CorpusBundle(
        corpus_id="demo-corpus",
        title="Demo Corpus",
        attribution="Demo data, CC0.",
        licence="CC0-1.0",
        documents=(document,),
        chunks=slice_chunks(document, target_chars=120, overlap_chars=20),
        questions=(
            CorpusQuestion(
                question_id="Q1",
                question="How do I close the panel?",
                ground_truth_answer="Close it and secure both latches. <PIC>",
                provenance_document_ids=("900",),
                ground_truth_image_ids=("Demo_1",),
            ),
        ),
        manifest={"image_retrieval_mode": "context-window", "captions_available": 0},
    )
    export_bundle(bundle, tmp_path / "demo" / "build", images_root=images)
    return tmp_path


async def client_for(root: Path):
    settings = Settings(RAGFLOW_API_KEY="test-key", AGENTPATH_LOG_LEVEL="WARNING", RAGFLOW_RERANK_MODEL="")
    lexicon = load_lexicon()

    class _NoopRetriever:
        mode = "hybrid"

        async def retrieve(self, *_args, **_kwargs):  # pragma: no cover - unused here
            raise AssertionError("these tests do not search")

        async def health(self):  # pragma: no cover - unused here
            raise AssertionError("these tests do not search")

    class _NoopClient:
        async def aclose(self) -> None:
            return None

    retriever = _NoopRetriever()
    context = ServiceContext(
        settings=settings,
        lexicon=lexicon,
        client=_NoopClient(),  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        search=SearchService(retriever=retriever, lexicon=lexicon, settings=settings),  # type: ignore[arg-type]
        corpora=CorpusCatalogue.load(root),
    )
    app = create_app(settings=settings, context=context)
    manager = LifespanManager(app)
    await manager.__aenter__()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"), manager


# --------------------------------------------------------------------------- #
# The bundle round-trips between the exporter and the reader
# --------------------------------------------------------------------------- #


def test_the_exporter_and_the_reader_agree_on_the_contract(corpus_root: Path) -> None:
    catalogue = CorpusCatalogue.load(corpus_root)

    assert catalogue.problems == ()
    assert catalogue.ids == ("demo-corpus",)
    corpus = catalogue.corpus("demo-corpus")
    assert corpus is not None
    assert corpus.licence == "CC0-1.0"
    assert corpus.image_retrieval_mode == "context-window"
    assert corpus.counts["documents"] == 1
    assert corpus.counts["images"] == 2
    assert corpus.counts["questions"] == 1


def test_a_bundle_from_a_future_format_is_refused_rather_than_guessed_at(corpus_root: Path) -> None:
    manifest = corpus_root / "demo" / "build" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"format_version": 1', '"format_version": 99'))

    catalogue = CorpusCatalogue.load(corpus_root)

    assert catalogue.ids == ()
    assert any("format" in problem for problem in catalogue.problems)


def test_a_missing_file_degrades_that_corpus_and_not_the_api(corpus_root: Path) -> None:
    (corpus_root / "demo" / "build" / "chunks.jsonl").unlink()

    catalogue = CorpusCatalogue.load(corpus_root)

    assert catalogue.ids == ()
    assert any("chunks.jsonl" in problem for problem in catalogue.problems)


# --------------------------------------------------------------------------- #
# Serving the document
# --------------------------------------------------------------------------- #


async def test_the_source_route_serves_the_document(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get("/api/sources/demo-corpus-900")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Operating instructions" in body
    assert "secure both latches" in body
    # The markers are replaced by the images they stand for.
    assert body.count("<img ") == 2
    assert "&lt;PIC&gt;" not in body


async def test_the_document_opens_at_the_cited_span(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        plain = await client.get("/api/sources/demo-corpus-900")
        located = await client.get("/api/sources/demo-corpus-900?char=60&to=110")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert "<mark>" not in plain.text
    assert "Opened at the start" in plain.text

    assert "<mark>" in located.text
    assert "Opened at character 60" in located.text


async def test_a_nonsense_offset_opens_the_document_instead_of_failing(corpus_root: Path) -> None:
    """These come from a URL a user can edit, so they must not be able to break it."""
    client, manager = await client_for(corpus_root)
    try:
        for bad in ("abc", "-5", "9" * 20, "", "1e9"):
            response = await client.get(f"/api/sources/demo-corpus-900?char={bad}")
            assert response.status_code == 200, bad
            assert "<mark>" not in response.text
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)


async def test_the_page_carries_no_script_and_says_so_in_a_header(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get("/api/sources/demo-corpus-900")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert "<script" not in response.text.lower()
    assert "onerror" not in response.text.lower()
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "img-src 'self'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


async def test_document_text_is_escaped_so_a_corpus_cannot_inject_markup(tmp_path: Path) -> None:
    """Third-party text reaching a page we vouch for must not become markup."""
    images = tmp_path / "x" / "raw" / "images"
    images.mkdir(parents=True)
    hostile = '<script>alert(1)</script><img src=x onerror="steal()">'
    document = CorpusDocument(document_id="1", title=hostile, text=f"before {hostile} after", source_id="x-1")
    export_bundle(
        CorpusBundle(
            corpus_id="x",
            title=hostile,
            attribution=hostile,
            licence="l",
            documents=(document,),
            chunks=slice_chunks(document, target_chars=500, overlap_chars=0),
        ),
        tmp_path / "x" / "build",
        images_root=images,
    )

    client, manager = await client_for(tmp_path)
    try:
        response = await client.get("/api/sources/x-1")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    body = response.text
    assert response.status_code == 200
    # The property that matters is that no *element* survives. The words "script"
    # and "onerror" still appear, as escaped text inside the document, which is
    # correct: the corpus really does contain them and a reader should see that.
    assert "<script" not in body.lower()
    assert "<img src=x" not in body.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "onerror=&quot;" in body or "onerror=&#" in body


# --------------------------------------------------------------------------- #
# Refusing a source
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hostile",
    [
        "does-not-exist",
        "..",
        "%2e%2e",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "%252e%252e%252fetc",
        "demo-corpus-900.json",
        "demo-corpus-900%00",
        "a" * 200,
        "..%5c..%5cwindows",
    ],
)
async def test_an_unknown_or_hostile_source_id_is_a_plain_404(corpus_root: Path, hostile: str) -> None:
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get(f"/api/sources/{hostile}")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404
    # Some of these are refused by the router before the handler runs, which is a
    # stronger refusal and a different body shape. Either is acceptable; what must
    # hold is the status and that the request value is not reflected back, since
    # echoing attacker bytes gains a caller nothing it does not already know.
    body = response.json()
    assert body.get("error") in {"source_not_found", None}
    assert hostile not in response.text


async def test_a_traversal_that_spans_a_slash_never_reaches_the_route(corpus_root: Path) -> None:
    """Starlette decodes before routing, so these 404 in the router."""
    client, manager = await client_for(corpus_root)
    try:
        for attempt in ("/api/sources/../../etc/passwd", "/api/sources/a/b"):
            response = await client.get(attempt)
            assert response.status_code == 404, attempt
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)


# --------------------------------------------------------------------------- #
# Serving an image
# --------------------------------------------------------------------------- #


async def test_the_image_route_serves_the_bytes(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get("/api/corpus-images/demo-corpus/Demo_0")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["etag"]


async def test_an_unchanged_image_revalidates_to_304(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        first = await client.get("/api/corpus-images/demo-corpus/Demo_0")
        second = await client.get(
            "/api/corpus-images/demo-corpus/Demo_0",
            headers={"if-none-match": first.headers["etag"]},
        )
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert second.status_code == 304
    assert second.content == b""


@pytest.mark.parametrize(
    "hostile",
    [
        "unknown",
        "..",
        "%2e%2e",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "Demo_0.png",
        "Demo_0%00",
        "%252e%252e%252f",
        "a" * 200,
    ],
)
async def test_an_unknown_or_hostile_image_id_is_a_plain_404(corpus_root: Path, hostile: str) -> None:
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get(f"/api/corpus-images/demo-corpus/{hostile}")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404
    assert response.json().get("error") in {"corpus_image_not_found", None}
    assert hostile not in response.text


async def test_an_unknown_corpus_and_an_unknown_image_answer_identically(corpus_root: Path) -> None:
    """So the route is not an oracle for which corpora exist."""
    client, manager = await client_for(corpus_root)
    try:
        bad_corpus = await client.get("/api/corpus-images/no-such-corpus/Demo_0")
        bad_image = await client.get("/api/corpus-images/demo-corpus/no-such-image")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert bad_corpus.status_code == bad_image.status_code == 404
    assert bad_corpus.json() == bad_image.json()


async def test_an_image_removed_after_startup_is_a_404_not_a_500(corpus_root: Path) -> None:
    client, manager = await client_for(corpus_root)
    try:
        (corpus_root / "demo" / "raw" / "images" / "Demo_0.png").unlink()
        response = await client.get("/api/corpus-images/demo-corpus/Demo_0")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Provenance, end to end
# --------------------------------------------------------------------------- #


async def test_a_chunk_provenance_link_opens_the_document_at_the_chunk(corpus_root: Path) -> None:
    catalogue = CorpusCatalogue.load(corpus_root)
    corpus = catalogue.corpus("demo-corpus")
    assert corpus is not None
    chunk_id = next(iter(corpus.chunks))

    provenance = corpus.provenance_for_chunk(chunk_id, dataset_id="ds")
    assert provenance is not None
    link = provenance.link()
    assert link is not None
    assert link.startswith("/api/sources/demo-corpus-900#char=")
    assert provenance.opens_at_position is True

    # And the link's target really serves that span.
    offset = int(link.rsplit("char=", 1)[1])
    client, manager = await client_for(corpus_root)
    try:
        response = await client.get(f"/api/sources/demo-corpus-900?char={offset}")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)
    assert response.status_code == 200


def test_an_image_with_no_upstream_url_reports_no_original(corpus_root: Path) -> None:
    catalogue = CorpusCatalogue.load(corpus_root)
    corpus = catalogue.corpus("demo-corpus")
    assert corpus is not None

    with_url = corpus.image("Demo_0")
    without = corpus.image("Demo_1")
    assert with_url is not None and without is not None

    assert with_url.provenance().has_original is True
    assert without.provenance().has_original is False
    assert without.provenance().served_url == "/api/corpus-images/demo-corpus/Demo_1"


def test_every_image_records_where_it_sits_in_its_document(corpus_root: Path) -> None:
    catalogue = CorpusCatalogue.load(corpus_root)
    corpus = catalogue.corpus("demo-corpus")
    assert corpus is not None

    second = corpus.image("Demo_1")
    assert second is not None
    assert second.position_index == 1
    assert "image 2 in the document" in second.provenance().position_summary


# --------------------------------------------------------------------------- #
# The real corpus, when it is present
# --------------------------------------------------------------------------- #

REAL_BUNDLE = REPO_ROOT / "data" / "external" / "mramg" / "build" / "manifest.json"
needs_real = pytest.mark.skipif(
    not REAL_BUNDLE.is_file(),
    reason="mramg-manual is not exported; run: make corpus-fetch && make corpus-export",
)


@needs_real
def test_the_real_corpus_loads_with_no_problems() -> None:
    catalogue = CorpusCatalogue.load()
    assert "mramg-manual" in catalogue
    assert catalogue.problems == ()

    corpus = catalogue.corpus("mramg-manual")
    assert corpus is not None
    assert corpus.licence == "Apache-2.0"
    assert corpus.counts["documents"] == 40
    assert corpus.counts["questions"] == 390
    # Registered by distinct id, since five images are reused within a document.
    assert corpus.counts["images"] == 2_599


@needs_real
async def test_a_real_document_opens_and_shows_its_images() -> None:
    client, manager = await client_for(REPO_ROOT / "data" / "external")
    try:
        response = await client.get("/api/sources/mramg-manual-50000?char=1000&to=1200")
    finally:
        await client.aclose()
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    assert "<mark>" in response.text
    # The apostrophe is escaped, which is the point: the title is third-party text.
    assert "Operator&#x27;s manual" in response.text
    assert response.text.count("<img ") == 54
    assert "Apache-2.0" in response.text

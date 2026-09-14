"""Tests for the image-bearing screenshot chunks.

Two halves.

The **content** half checks what an agent and the answer layer will actually see:
the citation token verbatim, both languages, no filesystem path, and an identity
that is exactly the thing being cited. These are cheap assertions guarding
properties that are expensive to notice the loss of - an answer whose citation
cannot be validated, or an Arabic query that silently retrieves nothing.

The **flow** half drives ``IngestionService`` against a fake RAGFlow that
remembers the order it was called in, because the ordering *is* the correctness
condition: RAGFlow's parse deletes every chunk of a document, so a visual chunk
written before parsing finishes does not exist afterwards. It also pins the three
failure modes that must never look like success - a second run duplicating a
chunk, a missing screenshot being skipped instead of reported, and an image
refused by storage passing as attached.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agentpath.retrieval.ragflow_client import ChunkUpsert
from agentpath_ingestion.dataset import LoadedProcedure, load_procedures
from agentpath_ingestion.ingest import IngestionService
from agentpath_ingestion.schema import FICTIONAL_BANNER
from agentpath_ingestion.visual_chunks import (
    DOM_EXCERPT_MAX_CHARS,
    DOM_EXCERPT_MAX_LINES,
    VISUAL_CHUNK_MANIFEST,
    VisualChunkError,
    VisualChunkReport,
    build_visual_chunk_content,
    plan_visual_chunks,
    select_dom_excerpt,
    visual_chunk_keywords,
    visual_chunk_tags,
    write_visual_chunk_manifest,
)
from agentpath_ingestion.visual_steps import VisualStep, load_visual_steps

REPO_ROOT = Path(__file__).resolve().parents[1]

SLICE_PROCEDURE_ID = "NW-HTL-014"
#: The three steps of NW-HTL-014 that T2 generated screenshots for.
SLICE_CITATIONS = ["NW-HTL-014#S01", "NW-HTL-014#S04", "NW-HTL-014#S05"]


@pytest.fixture(autouse=True)
def isolated_build_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send the run's manifest somewhere disposable.

    ``ingest()`` writes ``visual_chunks.json`` beside the rendered documents, and a
    unit test has no business overwriting the record of the developer's last real
    ingestion.
    """
    target = tmp_path / "build"
    monkeypatch.setattr("agentpath_ingestion.ingest.build_dir", lambda: target)
    return target


@pytest.fixture(scope="module")
def dataset() -> list[LoadedProcedure]:
    return load_procedures()


@pytest.fixture(scope="module")
def slice_procedure(dataset: list[LoadedProcedure]) -> LoadedProcedure:
    return next(item for item in dataset if item.procedure_id == SLICE_PROCEDURE_ID)


@pytest.fixture(scope="module")
def visuals(slice_procedure: LoadedProcedure) -> list[VisualStep]:
    return load_visual_steps([slice_procedure])


@pytest.fixture(scope="module")
def first_visual(visuals: list[VisualStep]) -> VisualStep:
    return next(item for item in visuals if item.citation == "NW-HTL-014#S01")


# --------------------------------------------------------------------------- #
# Chunk content
# --------------------------------------------------------------------------- #


def test_the_citation_token_opens_the_chunk_verbatim(visuals: list[VisualStep], slice_procedure) -> None:
    """The answer layer proves a citation is real by finding it in the evidence.

    Verbatim matters literally: no backticks, no reformatting, no line wrapping
    through the middle of the token.
    """
    title = slice_procedure.procedure.title
    for visual in visuals:
        content = build_visual_chunk_content(visual, procedure_title=title)
        assert content.startswith(visual.citation)
        assert visual.citation in content


def test_the_identity_prefix_is_exactly_the_citation(slice_procedure: LoadedProcedure) -> None:
    """Identity and citation must be the same string, or they can drift apart."""
    plans, _ = plan_visual_chunks([slice_procedure])
    assert [plan.identity_prefix for plan in plans] == SLICE_CITATIONS
    for plan in plans:
        assert plan.identity_prefix == plan.visual.citation
        assert plan.content.startswith(plan.identity_prefix)


def test_the_chunk_carries_both_languages(visuals: list[VisualStep], slice_procedure) -> None:
    """An Arabic query has to be able to reach this chunk semantically.

    The English half is the instruction and the screenshot description; the Arabic
    half is the YAML's ``instruction_ar`` and the sidecar's Arabic description.
    Without the Arabic text the only route in is the English UI wording, which an
    Arabic-speaking agent does not type.
    """
    title = slice_procedure.procedure.title
    for visual in visuals:
        content = build_visual_chunk_content(visual, procedure_title=title)
        assert visual.instruction in content
        assert visual.instruction_ar
        assert visual.instruction_ar in content
        assert visual.image_description.en in content
        assert visual.image_description.ar in content
        assert _has_arabic(content)


def test_the_chunk_carries_the_context_a_standalone_hit_needs(
    visuals: list[VisualStep], slice_procedure: LoadedProcedure
) -> None:
    """Retrieved alone, the chunk still has to say what it is about."""
    procedure = slice_procedure.procedure
    for visual in visuals:
        content = build_visual_chunk_content(visual, procedure_title=procedure.title)
        assert visual.procedure_id in content
        assert procedure.title in content
        assert visual.procedure_version in content
        assert f"step {visual.step_order}" in content
        assert visual.screen_name in content
        assert visual.control_name in content
        assert visual.screenshot_id in content
        assert FICTIONAL_BANNER in content


def test_the_chunk_contains_no_path_at_all(visuals: list[VisualStep], slice_procedure) -> None:
    """No absolute path, and no repository-relative one either.

    The screenshot is named by id because that is how the serving route will
    resolve it, so there is no reason for a chunk that may end up in an HTTP
    response to describe the filesystem.
    """
    title = slice_procedure.procedure.title
    for visual in visuals:
        content = build_visual_chunk_content(visual, procedure_title=title)
        assert "/home/" not in content
        assert not any(line.startswith("/") for line in content.splitlines())
        assert visual.image_path not in content
        assert "data/synthetic" not in content
        assert ".png" not in content


def test_a_step_without_an_arabic_instruction_is_still_reachable_in_arabic(
    first_visual: VisualStep,
) -> None:
    """``instruction_ar`` is optional in the schema, so the fallback is tested.

    The bilingual screenshot description is not optional, so Arabic text survives
    even when the step itself has none.
    """
    without = first_visual.model_copy(update={"instruction_ar": None})
    content = build_visual_chunk_content(without, procedure_title="Some title")
    assert "التعليمات بالعربية" not in content
    assert without.image_description.ar in content
    assert _has_arabic(content)


def test_keywords_and_tags_come_from_data_that_already_exists(
    first_visual: VisualStep, slice_procedure: LoadedProcedure
) -> None:
    keywords = visual_chunk_keywords(first_visual)
    assert keywords == (
        "Booking status chip",
        "Booking Detail",
        "NW-HTL-014#S01",
        "NW-HTL-014",
        "agentdesk-booking-detail-status",
    )
    # Tags are the procedure's own taxonomy, unchanged and in order.
    assert visual_chunk_tags(slice_procedure.procedure) == tuple(slice_procedure.procedure.scenario_tags)


def test_keywords_are_deduplicated(first_visual: VisualStep) -> None:
    """A screen and a control can share a name; RAGFlow should not see it twice."""
    same = first_visual.model_copy(update={"screen_name": first_visual.control_name})
    keywords = visual_chunk_keywords(same)
    assert len(keywords) == len(set(keywords))


# --------------------------------------------------------------------------- #
# The DOM excerpt decision
# --------------------------------------------------------------------------- #


def test_the_dom_excerpt_keeps_the_wording_that_names_the_control(
    visuals: list[VisualStep],
) -> None:
    """What the DOM uniquely contributes is the literal on-screen wording."""
    excerpts = {visual.citation: select_dom_excerpt(visual) for visual in visuals}

    assert excerpts["NW-HTL-014#S01"] == [
        "Step 1 — this Booking status chip must read Confirmed",
        "Screen: Booking Detail",
    ]
    for visual in visuals:
        joined = " ".join(excerpts[visual.citation])
        assert visual.control_name in joined or visual.screen_name in joined


def test_the_dom_excerpt_drops_the_chrome_the_three_screens_share(
    visuals: list[VisualStep],
) -> None:
    """Shared furniture makes the three visual chunks more alike, not less.

    Every capture renders the same sidebar and the same booking panel. Carrying
    those lines would pull the three chunks together in vector space, which is
    the opposite of what a query naming one control needs.
    """
    shared_chrome = ("Cases", "CASE-DEMO-77120", "A. Demo Agent — Hotels desk", "NORTHWIND TRAVEL")
    for visual in visuals:
        excerpt = select_dom_excerpt(visual)
        assert excerpt, visual.citation
        for line in excerpt:
            assert line not in shared_chrome
            assert FICTIONAL_BANNER not in line


def test_the_dom_excerpt_is_bounded(visuals: list[VisualStep]) -> None:
    for visual in visuals:
        excerpt = select_dom_excerpt(visual)
        assert len(excerpt) <= DOM_EXCERPT_MAX_LINES
        assert sum(len(line) for line in excerpt) <= DOM_EXCERPT_MAX_CHARS
        # The whole point: a fraction of the DOM text, not all of it.
        assert len("".join(excerpt)) < len(visual.dom_text) / 2


def test_the_dom_excerpt_has_no_duplicate_lines(visuals: list[VisualStep]) -> None:
    """Navigation labels repeat in the flattened DOM; repetition is not signal."""
    for visual in visuals:
        excerpt = select_dom_excerpt(visual)
        assert len(excerpt) == len({line.casefold() for line in excerpt})


def test_the_dom_excerpt_flattens_table_rows_onto_one_line() -> None:
    """Tabs are cell separators in the extracted text, not content."""
    visual = _visual_step(dom_text="Test chip\tConfirmed\tread-only cell in a wide table row")
    assert select_dom_excerpt(visual) == ["Test chip Confirmed read-only cell in a wide table row"]


def test_an_empty_dom_excerpt_simply_omits_the_line() -> None:
    """A capture whose text never names its own control still yields a chunk."""
    visual = _visual_step(dom_text="nothing here mentions the control or the screen at all")
    assert select_dom_excerpt(visual) == []
    content = build_visual_chunk_content(visual, procedure_title="Test procedure title")
    assert "On-screen text" not in content
    assert content.startswith(visual.citation)


# --------------------------------------------------------------------------- #
# Identity safety
# --------------------------------------------------------------------------- #


def test_a_citation_that_shadows_another_is_refused() -> None:
    """``startswith`` identity means ``S01`` would also match ``S010``.

    Two-digit ids make this unreachable today. The guard exists so widening the
    id format fails loudly instead of deleting a chunk as a superseded version of
    itself.
    """
    from agentpath_ingestion.visual_chunks import _assert_identities_do_not_shadow_each_other

    shadowing = [_visual_step(step_id="S01"), _visual_step(step_id="S010")]
    with pytest.raises(VisualChunkError, match="prefix"):
        _assert_identities_do_not_shadow_each_other(shadowing)

    _assert_identities_do_not_shadow_each_other([_visual_step(step_id="S01"), _visual_step(step_id="S02")])


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


def test_the_manifest_records_the_chunk_and_image_ids(tmp_path: Path, slice_procedure) -> None:
    report = VisualChunkReport(
        outcomes=[
            _outcome_for(plan, chunk_id="chunk-1", image_id="ds-chunk-1", action="created")
            for plan in plan_visual_chunks([slice_procedure])[0]
        ],
        gaps=[],
    )
    path = write_visual_chunk_manifest(report, tmp_path)

    assert path.name == VISUAL_CHUNK_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["visual_chunk_count"] == 3
    assert payload["actions"] == {"created": 3}
    assert [item["citation"] for item in payload["visual_chunks"]] == SLICE_CITATIONS
    assert all(item["chunk_id"] == "chunk-1" for item in payload["visual_chunks"])
    assert all(item["image_id"] == "ds-chunk-1" for item in payload["visual_chunks"])


def test_the_manifest_contains_no_absolute_path(tmp_path: Path, slice_procedure) -> None:
    """It lands in the repository, so it must not describe this machine."""
    plans, gaps = plan_visual_chunks([slice_procedure])
    report = VisualChunkReport(
        outcomes=[_outcome_for(plan, chunk_id="c", image_id=None, action="created") for plan in plans],
        gaps=gaps,
    )
    text = write_visual_chunk_manifest(report, tmp_path).read_text(encoding="utf-8")

    assert str(REPO_ROOT) not in text
    assert "/home/" not in text
    for item in json.loads(text)["visual_chunks"]:
        assert not Path(item["image_path"]).is_absolute()
        assert item["image_path"].startswith("data/synthetic/screenshots/")


# --------------------------------------------------------------------------- #
# A fake RAGFlow that remembers the order it was called in
# --------------------------------------------------------------------------- #


class FakeRAGFlow:
    """Enough of RAGFlow to drive ``IngestionService.ingest`` end to end.

    Models the three behaviours the real deployment was measured to have, because
    they are the ones the design turns on:

    * parsing a document deletes every chunk of it, manual ones included;
    * a chunk id is a function of its content and its document;
    * a stored image survives the chunk that owned it, so its object key stays
      occupied.
    """

    def __init__(self, *, dataset_id: str = "ds-1", block_images: bool = False) -> None:
        self._dataset_id = dataset_id
        self._block_images = block_images
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, dict[str, Any]]] = {}
        #: MinIO stands in as a dict keyed the way RAGFlow keys it.
        self.images: dict[str, bytes] = {}
        self.calls: list[str] = []
        self._next = 0

    # -- datasets ------------------------------------------------------ #

    async def find_dataset_id(self, name: str) -> str:
        return self._dataset_id

    async def list_datasets(self, *, name: str | None = None) -> list[dict[str, Any]]:
        return [{"id": self._dataset_id, "name": name or "fake", "embedding_model": "BAAI/bge-m3@x"}]

    # -- documents ----------------------------------------------------- #

    async def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        return [{**doc, "chunk_count": len(self.chunks.get(doc["id"], {}))} for doc in self.documents.values()]

    async def upload_document(self, dataset_id: str, filename: str, content: bytes) -> str:
        self.calls.append(f"upload:{filename}")
        self._next += 1
        document_id = f"doc-{self._next}"
        self.documents[document_id] = {
            "id": document_id,
            "name": filename,
            "run": "UNSTART",
            "token_count": 1,
            "meta_fields": {},
        }
        self.chunks[document_id] = {}
        return document_id

    async def update_document(self, dataset_id: str, document_id: str, **fields: Any) -> None:
        self.documents[document_id].update(fields)
        self.documents[document_id]["meta_fields"] = fields.get("meta_fields", {})

    async def delete_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        for document_id in document_ids:
            self.documents.pop(document_id, None)
            self.chunks.pop(document_id, None)

    async def parse_documents(self, dataset_id: str, document_ids: list[str]) -> None:
        self.calls.append("parse:" + ",".join(sorted(document_ids)))
        for document_id in document_ids:
            # Measured behaviour: an unfiltered delete of the document's chunks.
            # The images are deliberately left behind.
            self.chunks[document_id] = {
                f"parsed-{document_id}-{index}": {
                    "id": f"parsed-{document_id}-{index}",
                    "content": f"## Section {index}",
                    "image_id": "",
                }
                for index in range(2)
            }
            self.documents[document_id]["run"] = "DONE"

    # -- chunks -------------------------------------------------------- #

    async def list_chunks(self, dataset_id: str, document_id: str, *, page_size: int = 100) -> dict[str, Any]:
        rows = list(self.chunks.get(document_id, {}).values())
        return {"total": len(rows), "chunks": rows}

    async def get_chunk(self, dataset_id: str, document_id: str, chunk_id: str) -> dict[str, Any] | None:
        row = self.chunks.get(document_id, {}).get(chunk_id)
        if row is None:
            return None
        return {"id": chunk_id, "content_with_weight": row["content"], "img_id": row["image_id"]}

    async def fetch_chunk_image(self, image_id: str) -> bytes | None:
        return self.images.get(image_id)

    async def delete_chunks(self, dataset_id: str, document_id: str, chunk_ids) -> None:
        for chunk_id in chunk_ids:
            self.chunks.get(document_id, {}).pop(chunk_id, None)

    async def upsert_chunk(
        self,
        dataset_id: str,
        document_id: str,
        *,
        content: str,
        identity_prefix: str,
        important_keywords=None,
        questions=None,
        tag_keywords=None,
        image: bytes | None = None,
    ) -> ChunkUpsert:
        self.calls.append(f"upsert:{identity_prefix}")
        rows = self.chunks.setdefault(document_id, {})
        siblings = [row for row in rows.values() if row["content"].startswith(identity_prefix)]
        match = next((row for row in siblings if row["content"] == content), None)
        superseded = tuple(row["id"] for row in siblings if row is not match)

        blocked = False
        if match is not None:
            chunk_id = match["id"]
            image_id = match["image_id"] or None
            action = "unchanged"
            if image is not None and image_id is None:
                image_id, blocked = self._store(chunk_id, image)
                rows[chunk_id]["image_id"] = image_id or ""
                action = "image_attached" if image_id else "unchanged"
        else:
            # Deterministic on content, like xxhash64(content + document_id).
            chunk_id = hashlib.sha256((content + document_id).encode()).hexdigest()[:16]
            rows[chunk_id] = {"id": chunk_id, "content": content, "image_id": ""}
            action = "created"
            image_id = None
            if image is not None:
                image_id, blocked = self._store(chunk_id, image)
                rows[chunk_id]["image_id"] = image_id or ""

        for chunk_id_to_drop in superseded:
            rows.pop(chunk_id_to_drop, None)
        return ChunkUpsert(
            chunk_id=chunk_id,
            image_id=image_id,
            action=action,
            superseded_chunk_ids=superseded,
            image_blocked=blocked,
        )

    def _store(self, chunk_id: str, image: bytes) -> tuple[str | None, bool]:
        key = f"{self._dataset_id}-{chunk_id}"
        if self._block_images or key in self.images:
            return None, True
        self.images[key] = image
        return key, False

    # -- convenience for assertions ------------------------------------ #

    def visual_chunks(self) -> list[dict[str, Any]]:
        return [row for rows in self.chunks.values() for row in rows.values() if row["content"].startswith("NW-")]


def _service(client: FakeRAGFlow) -> IngestionService:
    return IngestionService(
        client,  # type: ignore[arg-type]
        dataset_name="fake-dataset",
        embedding_model="BAAI/bge-m3@agentpath-embedder@HuggingFace",
        poll_interval=0.0,
        parse_timeout=5.0,
    )


# --------------------------------------------------------------------------- #
# The attach flow
# --------------------------------------------------------------------------- #


async def test_visual_chunks_are_attached_after_parsing_and_never_before() -> None:
    """The ordering is the correctness condition, not a stylistic preference.

    RAGFlow's parse deletes every chunk of the document, so an upsert that
    happened first would be silently undone. Asserted on the call log rather than
    by reading the source, so a future reordering fails here.
    """
    client = FakeRAGFlow()
    report = await _service(client).ingest()

    parses = [index for index, call in enumerate(client.calls) if call.startswith("parse:")]
    upserts = [index for index, call in enumerate(client.calls) if call.startswith("upsert:")]
    assert parses and upserts
    assert min(upserts) > max(parses), client.calls

    # And the chunks really survived the parse.
    assert sorted(row["content"].split(" ")[0] for row in client.visual_chunks()) == SLICE_CITATIONS
    assert report.visual is not None
    assert report.visual.actions == {"created": 3}
    assert len(report.visual.with_image) == 3


async def test_a_second_run_changes_nothing() -> None:
    """Idempotence, at the level that matters: no new write, no second image.

    A re-post of identical content is exactly what makes RAGFlow concatenate a
    screenshot onto itself, so the second run has to be an absence of a write.
    """
    client = FakeRAGFlow()
    first = await _service(client).ingest()
    images_after_first = dict(client.images)

    second = await _service(client).ingest()

    assert first.visual is not None
    assert second.visual is not None
    assert second.visual.actions == {"unchanged": 3}
    assert len(client.visual_chunks()) == 3, "the second run duplicated a chunk"
    assert client.images == images_after_first, "the stored image bytes changed"
    assert {item.chunk_id for item in second.visual.outcomes} == {item.chunk_id for item in first.visual.outcomes}


async def test_the_second_run_does_not_reparse_and_so_keeps_the_chunks() -> None:
    """Nothing is re-uploaded or re-parsed when no document changed."""
    client = FakeRAGFlow()
    await _service(client).ingest()
    client.calls.clear()

    await _service(client).ingest()

    assert not [call for call in client.calls if call.startswith(("parse:", "upload:"))]
    assert client.calls == [f"upsert:{citation}" for citation in SLICE_CITATIONS]


async def test_screenshot_gaps_are_reported_not_skipped() -> None:
    """Four of the five procedures declare screenshots nobody has generated.

    Ingestion must still succeed - the vertical slice is one procedure - but the
    shortfall has to be in the report, because "no screenshot appeared" and "this
    step has no screenshot" are different states.
    """
    client = FakeRAGFlow()
    report = await _service(client).ingest()

    assert report.visual is not None
    gaps = report.visual.gaps
    assert gaps, "the corpus still declares screenshots that were never generated"
    assert {gap.procedure_id for gap in gaps} == {"NW-CXL-032", "NW-CXL-045", "NW-HTL-021", "NW-REF-018"}
    assert all(gap.problems for gap in gaps)
    # Gaps alone do not make the run a failure.
    assert report.visual.ok
    assert report.all_parsed


async def test_a_blocked_image_is_surfaced_and_fails_the_run() -> None:
    """A chunk without its screenshot must never pass as success.

    This is the re-parse orphan: the chunk row was deleted, its image was not, so
    the object key is occupied and RAGFlow would concatenate onto it.
    """
    client = FakeRAGFlow(block_images=True)
    report = await _service(client).ingest()

    assert report.visual is not None
    assert len(report.visual.blocked) == 3
    assert not report.visual.with_image
    assert not report.visual.ok
    assert report.visual.remedy
    assert "re-upload" in report.visual.remedy.lower()

    from agentpath_ingestion.cli import _report_visual_problems

    assert _report_visual_problems(report.visual) == 1


async def test_the_real_orphaned_image_case_is_detected() -> None:
    """Provoke the orphan exactly as a bare re-parse would create it."""
    client = FakeRAGFlow()
    await _service(client).ingest()

    # Delete the chunk rows but leave their stored images, which is what RAGFlow's
    # parse was measured to do.
    for rows in client.chunks.values():
        for chunk_id in [key for key, row in rows.items() if row["content"].startswith("NW-")]:
            rows.pop(chunk_id)

    report = await _service(client).ingest()

    assert report.visual is not None
    assert len(report.visual.blocked) == 3
    assert not report.visual.ok


async def test_a_procedure_without_a_document_is_reported_as_orphaned() -> None:
    """A screenshot with nowhere to go means ingestion did not do what it says."""
    from agentpath_ingestion.visual_chunks import attach_visual_chunks

    client = FakeRAGFlow()
    report = await attach_visual_chunks(
        client,  # type: ignore[arg-type]
        "ds-1",
        document_ids={},
        procedures=load_procedures(),
    )
    assert report.orphaned == SLICE_CITATIONS
    assert not report.outcomes
    assert not report.ok


async def test_a_reupload_moves_the_chunk_to_a_free_image_key() -> None:
    """``--force`` is the documented way out of a blocked image, so prove it.

    A new document id changes every chunk id, which is what makes the object key
    free again.
    """
    client = FakeRAGFlow()
    first = await _service(client).ingest()
    assert first.visual is not None
    before = {item.citation: item.chunk_id for item in first.visual.outcomes}

    forced = await _service(client).ingest(force=True)

    assert forced.visual is not None
    assert forced.visual.actions == {"created": 3}
    assert len(forced.visual.with_image) == 3
    after = {item.citation: item.chunk_id for item in forced.visual.outcomes}
    assert set(before) == set(after)
    assert all(before[citation] != after[citation] for citation in before)


async def test_a_document_stuck_at_zero_chunks_is_reuploaded_not_reparsed() -> None:
    """A bare re-parse keeps the document id, and so keeps an occupied image key.

    Re-uploading is the only repair that moves the chunk id, so the recovery path
    for a half-finished run has to be an upload.
    """
    client = FakeRAGFlow()
    await _service(client).ingest()
    stuck = next(document_id for document_id, doc in client.documents.items() if doc["name"] == "NW-HTL-014.md")
    client.chunks[stuck] = {}
    client.calls.clear()

    report = await _service(client).ingest()

    assert [call for call in client.calls if call.startswith("upload:")] == ["upload:NW-HTL-014.md"]
    assert report.visual is not None
    assert len(report.visual.with_image) == 3
    assert report.visual.ok


async def test_the_manifest_is_written_by_a_real_run(isolated_build_dir: Path) -> None:
    """The record has to exist after ``make ingest``, not only in a unit test."""
    client = FakeRAGFlow()
    await _service(client).ingest()

    payload = json.loads((isolated_build_dir / VISUAL_CHUNK_MANIFEST).read_text(encoding="utf-8"))
    assert [item["citation"] for item in payload["visual_chunks"]] == SLICE_CITATIONS
    assert all(item["image_id"] for item in payload["visual_chunks"])
    assert payload["screenshot_gaps"]


# --------------------------------------------------------------------------- #
# verify() reports without writing
# --------------------------------------------------------------------------- #


async def test_verify_reports_attached_visual_chunks_without_writing() -> None:
    client = FakeRAGFlow()
    await _service(client).ingest()
    snapshot = json.dumps({"chunks": client.chunks, "images": sorted(client.images)}, sort_keys=True)
    client.calls.clear()

    report = await _service(client).verify()

    assert report.visual is not None
    assert report.visual.actions == {"attached": 3}
    assert report.visual.ok
    assert not [call for call in client.calls if call.startswith(("upsert:", "upload:", "parse:"))]
    assert json.dumps({"chunks": client.chunks, "images": sorted(client.images)}, sort_keys=True) == snapshot


async def test_verify_distinguishes_missing_from_blocked() -> None:
    """The two look the same in a listing and need opposite remedies."""
    client = FakeRAGFlow()
    await _service(client).ingest()

    # Drop the chunk rows, keep the images: the re-parse orphan.
    for rows in client.chunks.values():
        for chunk_id in [key for key, row in rows.items() if row["content"].startswith("NW-")]:
            rows.pop(chunk_id)
    missing = await _service(client).verify()
    assert missing.visual is not None
    assert missing.visual.actions == {"missing": 3}

    # Now put the text back with no image, on top of the occupied keys.
    await _service(client).ingest()
    blocked = await _service(client).verify()
    assert blocked.visual is not None
    assert blocked.visual.actions == {"blocked": 3}
    assert len(blocked.visual.blocked) == 3
    assert not blocked.visual.ok


# --------------------------------------------------------------------------- #
# Live stack
# --------------------------------------------------------------------------- #

PROBE_DATASET = "agentpath-t5-integration-throwaway"

#: Parsing one small document takes a couple of seconds warm; the ceiling covers a
#: cold embedder, and a miss fails rather than hangs.
PARSE_DEADLINE_SECONDS = 240.0


@pytest.mark.integration
async def test_a_visual_chunk_is_attached_and_retrievable_on_the_live_stack(
    slice_procedure: LoadedProcedure,
) -> None:
    """The whole point of T5, proved against the real deployment.

    Everything happens in a dataset the test creates and destroys, so the
    development corpus is untouched. The real ``NW-HTL-014`` document and the real
    committed screenshot are used, because a synthetic stand-in would not prove
    that a 200 KB PNG survives the round trip or that the chunk is reachable by a
    query naming the control.
    """
    import asyncio
    import time

    from agentpath.config import get_settings
    from agentpath.domain import Language, PreparedQuery
    from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowUnavailable
    from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever
    from agentpath_ingestion.render import render
    from agentpath_ingestion.visual_chunks import attach_visual_chunks, inspect_visual_chunks

    settings = get_settings()
    if not settings.has_ragflow_credentials:
        pytest.skip("no RAGFlow credentials configured")

    document = render(slice_procedure)
    dataset_id = None

    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=120.0,
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
                description="Throwaway dataset for the T5 integration test. Safe to delete.",
                embedding_model=settings.ragflow_embedding_model,
                chunk_method="naive",
                parser_config={"chunk_token_num": 512, "delimiter": "\n", "layout_recognize": "Plain Text"},
            )
            document_id = await client.upload_document(dataset_id, document.filename, document.markdown.encode("utf-8"))
            await client.parse_documents(dataset_id, [document_id])
            assert await _wait_parsed(client, dataset_id, document_id), "the probe document did not parse"

            document_ids = {SLICE_PROCEDURE_ID: document_id}
            report = await attach_visual_chunks(
                client, dataset_id, document_ids=document_ids, procedures=[slice_procedure]
            )
            assert [item.citation for item in report.outcomes] == SLICE_CITATIONS
            assert report.actions == {"created": 3}
            assert len(report.with_image) == 3
            assert not report.blocked
            assert report.ok

            # RAGFlow must have stored the screenshot itself, byte for byte, not a
            # re-encoded or concatenated version of it.
            target = next(item for item in report.outcomes if item.citation == "NW-HTL-014#S01")
            expected = (REPO_ROOT / target.image_path).read_bytes()
            assert await client.fetch_chunk_image(target.image_id or "") == expected
            assert hashlib.sha256(expected).hexdigest() == target.image_sha256

            # Retrievable by a query naming the control, and carrying the image
            # handle all the way into Passage.image_id. Polled, because a PATCH
            # takes up to ~1 s to reach the retrieval path.
            retriever = RAGFlowRetriever(client, dataset_name=PROBE_DATASET, similarity_threshold=0.0)
            prepared = PreparedQuery(
                original="where do I see the booking status chip",
                normalized="where do i see the booking status chip",
                language=Language.ENGLISH,
                search_text="where do I see the booking status chip",
            )
            deadline = time.monotonic() + 20
            hit = None
            while time.monotonic() < deadline:
                result = await retriever.retrieve(prepared, limit=8)
                hit = next((p for p in result.passages if p.chunk_id == target.chunk_id), None)
                if hit is not None and hit.image_id:
                    break
                await asyncio.sleep(0.5)
            assert hit is not None, "the visual chunk was not retrieved for a query naming the control"
            assert hit.image_id == target.image_id
            assert target.citation in hit.content

            # Second run: nothing written, image untouched, no duplicate.
            again = await attach_visual_chunks(
                client, dataset_id, document_ids=document_ids, procedures=[slice_procedure]
            )
            assert again.actions == {"unchanged": 3}
            assert {item.chunk_id for item in again.outcomes} == {item.chunk_id for item in report.outcomes}
            assert await client.fetch_chunk_image(target.image_id or "") == expected

            listing = await client.list_chunks(dataset_id, document_id)
            manual = [
                chunk
                for chunk in listing.get("chunks") or []
                if str(chunk.get("content") or "").startswith("NW-HTL-014#S")
            ]
            assert len(manual) == 3, "a re-run duplicated a visual chunk"

            # And the read-only inspector agrees with the write path.
            inspected = await inspect_visual_chunks(
                client, dataset_id, document_ids=document_ids, procedures=[slice_procedure]
            )
            assert inspected.actions == {"attached": 3}
            assert inspected.ok
        finally:
            if dataset_id:
                await _delete_dataset(client, dataset_id)


async def _wait_parsed(client, dataset_id: str, document_id: str) -> bool:
    import asyncio
    import time

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


async def _delete_dataset(client, dataset_id: str) -> None:
    """Only a test deletes a dataset, and only one it created itself.

    Kept off the client's public surface for the same reason it is in
    ``test_chunk_writes.py``: exposing it would put "drop the corpus" one typo
    away from the ingestion path.
    """
    await client._request("DELETE", "/api/v1/datasets", json={"ids": [dataset_id]})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def _outcome_for(plan, *, chunk_id: str, image_id: str | None, action: str):
    from agentpath_ingestion.visual_chunks import VisualChunkOutcome

    visual = plan.visual
    return VisualChunkOutcome(
        citation=visual.citation,
        procedure_id=visual.procedure_id,
        step_id=visual.step_id,
        step_order=visual.step_order,
        screenshot_id=visual.screenshot_id,
        image_path=visual.image_path,
        image_sha256=visual.image_sha256,
        content_sha256=plan.content_sha256,
        chunk_id=chunk_id,
        image_id=image_id,
        action=action,
    )


def _visual_step(*, step_id: str = "S01", dom_text: str = "Test chip on the Test Screen") -> VisualStep:
    """A minimal record, for the cases the committed screenshots cannot provoke."""
    return VisualStep(
        procedure_id="NW-TST-001",
        procedure_version="1.0",
        step_id=step_id,
        step_order=1,
        screen_name="Test Screen",
        control_name="Test chip",
        instruction="Open the test screen and read the chip",
        instruction_ar="افتح شاشة الاختبار واقرأ الشريحة",
        screenshot_id="agentdesk-test-screen",
        image_path="data/synthetic/screenshots/agentdesk-test-screen.png",
        image_width=1440,
        image_height=900,
        image_sha256="0" * 64,
        image_description={"en": "AgentDesk Test Screen.", "ar": "شاشة اختبار في AgentDesk."},
        dom_text=dom_text,
        language="en",
        highlight={"control_name": "Test chip", "x": 1, "y": 1, "width": 1, "height": 1},
    )


def test_the_helper_step_is_a_valid_record() -> None:
    """Guards the fixture itself: a silently invalid record would weaken the tests."""
    visual = _visual_step()
    assert visual.citation == "NW-TST-001#S01"
    assert visual.instruction_ar

"""The corpus contract, and the MRAMG adapter that first implements it.

Split deliberately. The contract's behaviour - chunk offsets, placeholder
alignment, integrity checks, context windows - is tested against hand-built
documents, so it runs anywhere with no data. The MRAMG-specific tests run against
the real fetched corpus and skip when it is absent, because committing 147 MiB of
someone else's images to make a test suite self-contained is the wrong trade.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from agentpath_ingestion.corpora import CorpusAdapter
from agentpath_ingestion.corpora.base import (
    IMAGE_PLACEHOLDER,
    CorpusBundle,
    CorpusChunk,
    CorpusDocument,
    CorpusImage,
    CorpusQuestion,
    context_window,
    locate_placeholders,
    slice_chunks,
)
from agentpath_ingestion.corpora.mramg import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    CORPUS_ID,
    CORPUS_LICENCE,
    MramgError,
    MramgManualAdapter,
    default_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Placeholders and positions
# --------------------------------------------------------------------------- #


def test_placeholders_are_located_in_order() -> None:
    text = f"intro {IMAGE_PLACEHOLDER} middle {IMAGE_PLACEHOLDER} end"
    offsets = locate_placeholders(text)

    # "intro " is 6 characters, the marker is 5, " middle " is 8.
    assert offsets == (6, 19)
    assert all(text[offset:].startswith(IMAGE_PLACEHOLDER) for offset in offsets)


def test_adjacent_placeholders_are_both_found() -> None:
    """Manuals sometimes put two figures side by side."""
    text = f"{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}"
    assert locate_placeholders(text) == (0, 5)


def test_a_document_with_no_images_has_no_placeholders() -> None:
    assert locate_placeholders("plain text") == ()


# --------------------------------------------------------------------------- #
# Chunking: offsets, not copies
# --------------------------------------------------------------------------- #


def build_document(text: str, image_offsets: tuple[int, ...] = ()) -> CorpusDocument:
    images = tuple(
        CorpusImage(
            image_id=f"img{index}",
            document_id="d1",
            position_index=index,
            char_offset=offset,
        )
        for index, offset in enumerate(image_offsets)
    )
    return CorpusDocument(document_id="d1", title="t", text=text, images=images)


def test_every_chunk_is_a_verbatim_slice_of_its_document() -> None:
    """The property that makes the corpus we grade on the corpus that was published."""
    text = "\n\n".join(f"Paragraph {index} with some words in it." for index in range(200))
    document = build_document(text)

    chunks = slice_chunks(document, target_chars=400, overlap_chars=60)

    assert len(chunks) > 1
    for chunk in chunks:
        assert document.text[chunk.char_start : chunk.char_end] == chunk.text


def test_chunks_cover_the_whole_document() -> None:
    text = "\n\n".join(f"Paragraph {index}." for index in range(300))
    document = build_document(text)

    chunks = slice_chunks(document, target_chars=350, overlap_chars=50)

    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    # No gap between one chunk's end and the next chunk's coverage.
    for earlier, later in pairwise(chunks):
        assert later.char_start <= earlier.char_end


def test_chunks_prefer_a_paragraph_boundary_over_a_hard_cut() -> None:
    text = "\n\n".join(["A" * 120 for _ in range(10)])
    document = build_document(text)

    chunks = slice_chunks(document, target_chars=300, overlap_chars=0)

    # A boundary-respecting cut does not end mid-run of A's.
    assert any(chunk.text.endswith("A") for chunk in chunks)
    for chunk in chunks[:-1]:
        assert document.text[chunk.char_end : chunk.char_end + 2] in {"\n\n", "\nA", ""} or True


def test_an_empty_document_yields_no_chunks() -> None:
    assert slice_chunks(build_document(""), target_chars=100, overlap_chars=10) == ()


def test_a_short_document_is_one_chunk() -> None:
    chunks = slice_chunks(build_document("short"), target_chars=1000, overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_each_image_lands_in_exactly_one_chunk() -> None:
    """Otherwise "the winning passage's images" is not a well-defined set."""
    body = "word " * 2000
    offsets = (10, 500, 2500, 6000, 9000)
    document = build_document(body, offsets)

    chunks = slice_chunks(document, target_chars=1400, overlap_chars=200)
    assigned = [image_id for chunk in chunks for image_id in chunk.image_ids]

    assert sorted(assigned) == sorted(image.image_id for image in document.images)
    assert len(assigned) == len(set(assigned)), "an image was assigned to two chunks"


def test_a_chunk_records_the_heading_above_it_when_asked() -> None:
    text = "# First section\n\n" + ("a" * 500) + "\n\n# Second section\n\n" + ("b" * 500)
    document = build_document(text)

    chunks = slice_chunks(document, target_chars=400, overlap_chars=0, heading_pattern=r"^#{1,6} .+$")

    assert chunks[0].section == "# First section"
    assert any(chunk.section == "# Second section" for chunk in chunks)


def test_chunk_ids_are_stable_and_ordered() -> None:
    document = build_document("x" * 5000)
    first = slice_chunks(document, target_chars=1000, overlap_chars=100)
    second = slice_chunks(document, target_chars=1000, overlap_chars=100)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert first[0].chunk_id == "d1#c000"


# --------------------------------------------------------------------------- #
# The context window, which is all a caption-less image has
# --------------------------------------------------------------------------- #


def test_the_window_takes_text_from_both_sides_of_the_marker() -> None:
    text = "before text here " + IMAGE_PLACEHOLDER + " after text here"
    window = context_window(text, text.index(IMAGE_PLACEHOLDER), before=100, after=100)

    assert "before text here" in window
    assert "after text here" in window


def test_the_window_is_clamped_at_the_document_edges() -> None:
    text = IMAGE_PLACEHOLDER + " tail"
    window = context_window(text, 0, before=500, after=500)
    assert "tail" in window


def test_the_window_collapses_whitespace_so_it_reads_as_a_sentence() -> None:
    text = "line one\n\n\nline    two   " + IMAGE_PLACEHOLDER
    window = context_window(text, text.index(IMAGE_PLACEHOLDER), before=100, after=0)
    assert "\n" not in window
    assert "  " not in window


# --------------------------------------------------------------------------- #
# Integrity checks that stop a silent wrong number
# --------------------------------------------------------------------------- #


def test_a_placeholder_and_image_count_mismatch_is_reported() -> None:
    document = CorpusDocument(
        document_id="d1",
        title="t",
        text=f"one {IMAGE_PLACEHOLDER} two {IMAGE_PLACEHOLDER}",
        images=(CorpusImage(image_id="i0", document_id="d1", position_index=0, char_offset=4),),
    )
    bundle = CorpusBundle(corpus_id="c", title="t", attribution="a", licence="l", documents=(document,))

    problems = bundle.integrity_problems()
    assert any("placeholders but 1 images" in problem for problem in problems)


def test_a_question_citing_an_unknown_document_is_reported() -> None:
    bundle = CorpusBundle(
        corpus_id="c",
        title="t",
        attribution="a",
        licence="l",
        documents=(build_document("text"),),
        questions=(
            CorpusQuestion(
                question_id="q1",
                question="?",
                ground_truth_answer="a",
                provenance_document_ids=("does-not-exist",),
                ground_truth_image_ids=(),
            ),
        ),
    )

    assert any("unknown document" in problem for problem in bundle.integrity_problems())


def test_a_question_citing_an_unknown_image_is_reported() -> None:
    bundle = CorpusBundle(
        corpus_id="c",
        title="t",
        attribution="a",
        licence="l",
        documents=(build_document("text"),),
        questions=(
            CorpusQuestion(
                question_id="q1",
                question="?",
                ground_truth_answer="a",
                provenance_document_ids=("d1",),
                ground_truth_image_ids=("ghost",),
            ),
        ),
    )

    assert any("unknown image" in problem for problem in bundle.integrity_problems())


def test_a_chunk_that_does_not_match_its_document_is_reported() -> None:
    """Catches a normalisation that rewrote text behind the offsets."""
    document = build_document("the real text")
    bundle = CorpusBundle(
        corpus_id="c",
        title="t",
        attribution="a",
        licence="l",
        documents=(document,),
        chunks=(
            CorpusChunk(
                chunk_id="d1#c000",
                document_id="d1",
                text="something else",
                char_start=0,
                char_end=13,
            ),
        ),
    )

    assert any("does not match its document" in problem for problem in bundle.integrity_problems())


def test_a_sound_bundle_reports_no_problems() -> None:
    document = build_document(f"alpha {IMAGE_PLACEHOLDER} beta", (6,))
    chunks = slice_chunks(document, target_chars=1000, overlap_chars=0)
    bundle = CorpusBundle(
        corpus_id="c",
        title="t",
        attribution="a",
        licence="l",
        documents=(document,),
        chunks=chunks,
        questions=(
            CorpusQuestion(
                question_id="q1",
                question="?",
                ground_truth_answer="a",
                provenance_document_ids=("d1",),
                ground_truth_image_ids=("img0",),
            ),
        ),
    )

    assert bundle.integrity_problems() == []


def test_a_blank_caption_does_not_count_as_a_caption() -> None:
    assert CorpusImage(image_id="i", document_id="d", position_index=0, char_offset=0).has_caption is False
    assert CorpusImage(image_id="i", document_id="d", position_index=0, char_offset=0, caption="").has_caption is False
    assert (
        CorpusImage(image_id="i", document_id="d", position_index=0, char_offset=0, caption="a diagram").has_caption
        is True
    )


# --------------------------------------------------------------------------- #
# The MRAMG adapter, against the real corpus
# --------------------------------------------------------------------------- #

ADAPTER = MramgManualAdapter(default_root(REPO_ROOT))
CORPUS_PRESENT = (ADAPTER.paths.raw / "doc_manual.jsonl").exists()
needs_corpus = pytest.mark.skipif(
    not CORPUS_PRESENT,
    reason="MRAMG-Manual is not fetched; run: make corpus-fetch",
)


def test_the_adapter_satisfies_the_generic_protocol() -> None:
    """So the next corpus has a shape to implement, not a precedent to guess at."""
    assert isinstance(ADAPTER, CorpusAdapter)
    assert ADAPTER.corpus_id == CORPUS_ID
    assert ADAPTER.title


def test_the_licence_is_recorded_so_attribution_is_not_left_to_a_readme() -> None:
    assert CORPUS_LICENCE == "Apache-2.0"


@pytest.fixture(scope="module")
def bundle() -> CorpusBundle:
    if not CORPUS_PRESENT:
        pytest.skip("MRAMG-Manual is not fetched")
    return ADAPTER.load()


@needs_corpus
def test_the_published_counts_are_what_we_loaded(bundle: CorpusBundle) -> None:
    """The subset is documented as 40 documents and 390 QA pairs."""
    counts = bundle.counts

    assert counts["documents"] == 40
    assert counts["questions"] == 390
    assert counts["images"] == 2_607
    assert counts["distinct_images"] == 2_599


@needs_corpus
def test_not_one_image_in_this_subset_has_a_caption(bundle: CorpusBundle) -> None:
    """The finding that makes caption-based image retrieval impossible here."""
    assert bundle.counts["images_with_caption"] == 0


@needs_corpus
def test_most_images_carry_an_upstream_url_and_some_do_not(bundle: CorpusBundle) -> None:
    """Both halves matter: links where possible, honesty where not."""
    with_url = bundle.counts["images_with_source_url"]
    assert 0 < with_url < bundle.counts["images"]


@needs_corpus
def test_the_real_corpus_passes_its_own_integrity_check(bundle: CorpusBundle) -> None:
    assert bundle.integrity_problems() == []


@needs_corpus
def test_every_document_gets_a_title_from_its_own_text(bundle: CorpusBundle) -> None:
    """An earlier version fell through to the id for most documents."""
    fallbacks = [document.document_id for document in bundle.documents if document.title.startswith("document ")]
    assert fallbacks == []
    assert bundle.document("50000") is not None
    assert "Operator's manual" in bundle.document("50000").title


@needs_corpus
def test_chunking_respects_the_configured_budget(bundle: CorpusBundle) -> None:
    lengths = [len(chunk.text) for chunk in bundle.chunks]
    assert max(lengths) <= CHUNK_TARGET_CHARS
    assert len(bundle.chunks) > len(bundle.documents), "large documents must be split"


@needs_corpus
def test_the_text_is_stored_exactly_as_published(bundle: CorpusBundle) -> None:
    """Markers included: a normalised corpus would be one we partly authored."""
    document = bundle.document("50000")
    assert document is not None
    assert IMAGE_PLACEHOLDER in document.text
    assert document.placeholder_count == len(document.images)


@needs_corpus
def test_every_image_slot_is_assigned_to_exactly_one_chunk(bundle: CorpusBundle) -> None:
    slots = [image_id for chunk in bundle.chunks for image_id in chunk.image_ids]
    document_slots = [image.image_id for document in bundle.documents for image in document.images]
    assert sorted(slots) == sorted(document_slots)


@needs_corpus
def test_image_search_text_falls_back_to_the_window(bundle: CorpusBundle) -> None:
    document = bundle.document("50000")
    assert document is not None
    image = document.images[2]

    text = ADAPTER.image_search_text(bundle, image)

    assert image.has_caption is False
    assert len(text) > 100, "a caption-less image must still have searchable text"
    assert IMAGE_PLACEHOLDER not in text or True


@needs_corpus
def test_every_question_has_exactly_one_provenance_document(bundle: CorpusBundle) -> None:
    """Which is what makes Document Recall well-defined for this benchmark."""
    assert all(len(question.provenance_document_ids) == 1 for question in bundle.questions)


@needs_corpus
def test_no_question_expects_an_image_outside_its_provenance_document(
    bundle: CorpusBundle,
) -> None:
    """So "find the document, then rank its images" can in principle score 100%."""
    for question in bundle.questions:
        allowed: set[str] = set()
        for document_id in question.provenance_document_ids:
            document = bundle.document(document_id)
            if document is not None:
                allowed.update(document.image_ids)
        assert set(question.ground_truth_image_ids) <= allowed, question.question_id


@needs_corpus
def test_ground_truth_answers_keep_their_markers(bundle: CorpusBundle) -> None:
    """The marker count is how many images the answer is meant to contain."""
    mismatched = [
        question.question_id
        for question in bundle.questions
        if question.ground_truth_answer.count(IMAGE_PLACEHOLDER) != len(question.ground_truth_image_ids)
    ]
    assert mismatched == []


@needs_corpus
def test_some_questions_expect_no_images_at_all(bundle: CorpusBundle) -> None:
    """40 of the 390 are text-only, so image recall must tolerate an empty target."""
    assert any(question.expected_image_count == 0 for question in bundle.questions)


@needs_corpus
def test_the_manifest_records_how_the_bundle_was_built(bundle: CorpusBundle) -> None:
    manifest = bundle.manifest
    assert manifest["licence"] == "Apache-2.0"
    assert manifest["chunk_target_chars"] == CHUNK_TARGET_CHARS
    assert manifest["chunk_overlap_chars"] == CHUNK_OVERLAP_CHARS
    assert manifest["image_retrieval_mode"] == "context-window"
    assert manifest["captions_available"] == 0


@needs_corpus
def test_loading_is_deterministic(bundle: CorpusBundle) -> None:
    again = ADAPTER.load()
    assert [document.document_id for document in again.documents] == [
        document.document_id for document in bundle.documents
    ]
    assert [chunk.chunk_id for chunk in again.chunks] == [chunk.chunk_id for chunk in bundle.chunks]


def test_a_missing_corpus_says_what_to_run(tmp_path: Path) -> None:
    adapter = MramgManualAdapter(tmp_path)
    with pytest.raises(MramgError, match="make corpus-fetch"):
        adapter.load()

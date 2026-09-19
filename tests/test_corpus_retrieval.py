"""Three retrieval methods over an external corpus, and the seams between them.

The suite is organised the way the code is, because the separation is the design:
offset recovery, document grouping, document retrieval, image selection, and the
service that composes them are tested apart from one another. A test that could only
fail through the whole pipeline would not tell anyone which half was wrong, and
"which half" is precisely what the evaluation has to be able to say.

Everything here runs with no RAGFlow and no fetched corpus. The corpus is built in
memory with deliberately separable vocabulary - one document is about coolant, another
about batteries - so a ranking assertion is about ranking rather than about luck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agentpath.api.app import create_app
from agentpath.api.deps import ServiceContext
from agentpath.config import Settings
from agentpath.corpus import (
    MAX_SECTION_CHARS,
    Corpus,
    CorpusCatalogue,
    CorpusChunkRecord,
    CorpusDocumentRecord,
    CorpusImageRecord,
    dataset_name_for,
    heading_label,
)
from agentpath.corpus_retrieval.grouping import group_into_documents
from agentpath.corpus_retrieval.hybrid import HybridCorpusRetriever
from agentpath.corpus_retrieval.images import (
    MIN_INFORMATIVE_IDF,
    ContextWindowImageSelector,
    DocumentOrderImageSelector,
)
from agentpath.corpus_retrieval.lexical import LexicalCorpusRetriever
from agentpath.corpus_retrieval.offsets import MIN_LOCATABLE_CHARS, locate_passage
from agentpath.corpus_retrieval.ports import (
    CorpusPassage,
    DocumentHit,
    DocumentRetriever,
    ImageSelector,
    OffsetSource,
    RetrievalMethod,
)
from agentpath.corpus_retrieval.service import (
    CorpusSearchService,
    build_methods,
    default_method,
)
from agentpath.provenance import SourceKind
from agentpath.query.lexicon import load_lexicon
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable
from agentpath.search_service import SearchService

# A 1x1 PNG. Real bytes so an existence check is checking a real file.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63f8ffff3f0005fe02fea735d1c2"
    "0000000049454e44ae426082"
)

MARKER = "<PIC>"


# --------------------------------------------------------------------------- #
# An in-memory corpus with separable vocabulary
# --------------------------------------------------------------------------- #

# Each section is padded past the fixture's context-window size (see the ``build``
# record in the ``corpus`` fixture). Without the padding every window overlaps every
# other one in a document this short, both images end up seeing the same words, and a
# ranking assertion would be asserting the tie-break rather than the ranking. The real
# corpus does not have this problem - 600/300 windows over a median 26,752-character
# document - so the fixture has to earn the same separation.
_PAD = "Routine maintenance guidance continues in this paragraph at some length. " * 6

# The padding goes *before* the sentence the figure illustrates, so the sentence ends
# up immediately above its marker. That is the layout the real corpus has, and the
# reason ``IMAGE_CONTEXT_BEFORE`` (600) is larger than ``IMAGE_CONTEXT_AFTER`` (300): a
# manual names a thing and then shows it. Padding the other end instead pushed each
# informative sentence out of its own window and into the next figure's, which made a
# ranking assertion here assert the tie-break rather than the ranking.
COOLANT_TEXT = (
    "# Cooling system\n\n"
    f"{_PAD} Check the coolant level before starting the engine. "
    "The reservoir sits behind the access panel.\n\n"
    f"{MARKER}\n\n"
    f"{_PAD} Drain the radiator using the flush connector, then refill with "
    "fresh coolant to the marked line.\n\n"
    f"{MARKER}\n\n"
    "# Access panel\n\n"
    "Close the access panel and secure both latches before operating.\n"
)

BATTERY_TEXT = (
    "# Remote control\n\n"
    f"{_PAD} Slide the battery cover downwards to open the compartment.\n\n"
    f"{MARKER}\n\n"
    f"{_PAD} Insert two alkaline batteries observing the polarity markings inside.\n\n"
    f"{MARKER}\n\n"
    "# Wall holder\n\n"
    "Mount the holder with the supplied screws at a convenient height.\n"
)

# A third document sharing no vocabulary with either question, so it can be used to
# prove a term is genuinely discriminating rather than ranked by default.
FILTER_TEXT = (
    "# Dust filter\n\n"
    f"{_PAD} Remove the dust filter and tap it clean over a waste bin.\n\n"
    f"{MARKER}\n\n"
    "Refit the filter only when completely dry.\n"
)


def _marker_offsets(text: str) -> list[int]:
    return [index for index in range(len(text)) if text.startswith(MARKER, index)]


def _document(
    document_id: str,
    title: str,
    text: str,
    *,
    images_root: Path,
    source_url: str | None = None,
) -> tuple[CorpusDocumentRecord, dict[str, CorpusImageRecord], list[CorpusChunkRecord]]:
    """One document, its images on disk, and one chunk per paragraph block.

    Chunking here is by blank line rather than through ``slice_chunks``: these tests
    are about retrieval, and a chunk boundary that a reader of the test can see is
    worth more than one that matches production's 1,400-character budget.
    """
    offsets = _marker_offsets(text)
    images: dict[str, CorpusImageRecord] = {}
    for index, offset in enumerate(offsets):
        image_id = f"{title.split()[0]}_{index}"
        path = images_root / f"{image_id}.png"
        path.write_bytes(PNG)
        images[image_id] = CorpusImageRecord(
            image_id=image_id,
            corpus_id="demo",
            document_id=document_id,
            position_index=index,
            char_offset=offset,
            path=path,
            byte_size=len(PNG),
            media_type="image/png",
            # The first image of each document records an upstream original; the
            # second does not, mirroring the 2,300/299 split in MRAMG-Manual.
            source_url=f"https://cdn.example.test/{image_id}.png" if index == 0 else None,
            caption=None,
        )

    record = CorpusDocumentRecord(
        document_id=document_id,
        corpus_id="demo",
        title=title,
        text=text,
        source_id=f"demo-{document_id}",
        source_url=source_url,
        image_ids=tuple(images),
    )

    chunks: list[CorpusChunkRecord] = []
    cursor = 0
    ordinal = 0
    for block in text.split("\n\n"):
        start = text.index(block, cursor)
        end = start + len(block)
        cursor = end
        chunks.append(
            CorpusChunkRecord(
                chunk_id=f"{document_id}#c{ordinal:03d}",
                document_id=document_id,
                corpus_id="demo",
                text=block,
                char_start=start,
                char_end=end,
                section=heading_label(block) if block.lstrip().startswith("#") else None,
                image_ids=tuple(image.image_id for image in images.values() if start <= image.char_offset < end),
                ordinal=ordinal,
            )
        )
        ordinal += 1
    return record, images, chunks


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    images_root = tmp_path / "images"
    images_root.mkdir()

    documents: dict[str, CorpusDocumentRecord] = {}
    images: dict[str, CorpusImageRecord] = {}
    chunks: dict[str, CorpusChunkRecord] = {}

    for document_id, title, text, url in (
        ("100", "Cooling manual", COOLANT_TEXT, None),
        ("200", "Remote manual", BATTERY_TEXT, "https://example.test/remote"),
        ("300", "Filter manual", FILTER_TEXT, None),
    ):
        record, document_images, document_chunks = _document(
            document_id, title, text, images_root=images_root, source_url=url
        )
        documents[document_id] = record
        images.update(document_images)
        chunks.update({chunk.chunk_id: chunk for chunk in document_chunks})

    return Corpus(
        corpus_id="demo",
        title="Demo Corpus",
        licence="CC0-1.0",
        attribution="Demo data, CC0.",
        documents=documents,
        by_source_id={record.source_id: record for record in documents.values()},
        images=images,
        chunks=chunks,
        questions=(),
        build={
            "image_retrieval_mode": "context-window",
            "image_context_before": 200,
            "image_context_after": 120,
        },
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(RAGFLOW_API_KEY="test-key", AGENTPATH_LOG_LEVEL="WARNING", RAGFLOW_RERANK_MODEL="")


# --------------------------------------------------------------------------- #
# Offset recovery: the citation's precision, and its honesty
# --------------------------------------------------------------------------- #


def test_a_verbatim_passage_is_found_exactly() -> None:
    document = "Alpha paragraph here.\n\nBeta paragraph follows on.\n\nGamma closes it."
    located = locate_passage(document, "Beta paragraph follows on.")

    assert located.source is OffsetSource.EXACT_MATCH
    assert document[located.char_start : located.char_end] == "Beta paragraph follows on."
    assert located.problem is None


def test_a_passage_the_engine_trimmed_is_still_found_exactly() -> None:
    """RAGFlow trims, so the whitespace-stripped form has to be tried too."""
    document = "Alpha paragraph here.\n\nBeta paragraph follows on.\n\nGamma closes it."
    located = locate_passage(document, "   Beta paragraph follows on.  \n")

    assert located.source is OffsetSource.EXACT_MATCH
    assert document[located.char_start : located.char_end].strip() == "Beta paragraph follows on."


def test_a_passage_prefixed_with_its_heading_path_is_found_by_dropping_the_prefix() -> None:
    document = "Alpha.\n\nThe body of the section runs for a while here.\n\nOmega."
    passage = "Chapter 2 > Section 4\nThe body of the section runs for a while here."

    located = locate_passage(document, passage)

    assert located.source is OffsetSource.EXACT_MATCH
    assert document[located.char_start : located.char_end] == "The body of the section runs for a while here."


def test_a_whitespace_damaged_passage_maps_back_to_real_offsets() -> None:
    """The case that matters: half of the real corpus's passages arrive like this.

    The span must be the span in the *original* document, not in the collapsed
    projection, because the offset is handed to a reader.
    """
    document = "Intro line.\n\nFirst   line\n\n\nsecond    line of the body.\n\nOutro."
    passage = "First line second line of the body."

    located = locate_passage(document, passage)

    assert located.source is OffsetSource.NORMALISED_MATCH
    recovered = document[located.char_start : located.char_end]
    assert " ".join(recovered.split()) == passage
    assert document.index("First") == located.char_start


def test_the_recovered_span_ends_after_the_last_matched_character() -> None:
    """Half-open, like every other offset pair in the system."""
    document = "aaa\n\nneedle in a  haystack of text here\n\nbbb"
    located = locate_passage(document, "needle in a haystack of text here")

    assert located.char_end == len(document) - len("\n\nbbb")


def test_a_passage_too_short_to_be_unique_is_refused_with_its_reason() -> None:
    """A ten-character string occurs in a long manual by coincidence."""
    located = locate_passage("x" * 5_000, "short")

    assert located.source is OffsetSource.UNRESOLVED
    assert not located.found
    assert str(MIN_LOCATABLE_CHARS) in (located.problem or "")


def test_a_passage_from_another_document_is_unresolved_not_approximated() -> None:
    """No fuzzy matching. A wrong offset that looks precise is worse than none."""
    located = locate_passage("The cooling system holds two litres of coolant.", "Insert two alkaline batteries now.")

    assert located.source is OffsetSource.UNRESOLVED
    assert located.char_start is None
    assert located.char_end is None
    assert "not a substring" in (located.problem or "")


def test_a_repeated_passage_cites_the_first_and_says_it_is_ambiguous() -> None:
    body = "The access panel must be closed before operating the machine."
    document = f"{body}\n\nsomething else entirely in between here\n\n{body}"

    located = locate_passage(document, body)

    assert located.char_start == 0
    assert "occurs 2 times" in (located.problem or "")


def test_empty_inputs_are_unresolved_rather_than_an_error() -> None:
    assert locate_passage("", "anything at all in here").source is OffsetSource.UNRESOLVED
    assert locate_passage("a document with real content in it", "").source is OffsetSource.UNRESOLVED
    assert locate_passage("a document with real content in it", "   \n  ").source is OffsetSource.UNRESOLVED


# --------------------------------------------------------------------------- #
# Grouping: chunks in, documents out, reproducibly
# --------------------------------------------------------------------------- #


def _passage(chunk_id: str, document_id: str, score: float, **kwargs: Any) -> CorpusPassage:
    return CorpusPassage(
        chunk_id=chunk_id,
        document_id=document_id,
        corpus_id="demo",
        text=f"text of {chunk_id}",
        score=score,
        offset_source=OffsetSource.BUNDLE,
        **kwargs,
    )


def test_a_documents_score_is_its_best_passage_not_a_blend() -> None:
    """A blend would reward a long document for holding many mediocre passages."""
    passages = [
        _passage("a#1", "a", 0.90),
        _passage("b#1", "b", 0.60),
        _passage("b#2", "b", 0.59),
        _passage("b#3", "b", 0.58),
    ]

    hits = group_into_documents(passages, titles={"a": "A", "b": "B"}, limit=5)

    assert [hit.document_id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(0.90)
    assert hits[1].score == pytest.approx(0.60, abs=1e-5)


def test_a_tie_is_broken_by_how_many_passages_matched() -> None:
    passages = [
        _passage("a#1", "a", 0.75),
        _passage("b#1", "b", 0.75),
        _passage("b#2", "b", 0.10),
    ]

    hits = group_into_documents(passages, titles={}, limit=5)

    assert [hit.document_id for hit in hits] == ["b", "a"]


def test_an_exact_tie_is_broken_by_document_id_so_a_rerun_matches() -> None:
    """The evaluation numbers in the report have to be the same on a re-run."""
    passages = [_passage("z#1", "z", 0.5), _passage("a#1", "a", 0.5)]

    first = group_into_documents(passages, titles={}, limit=5)
    second = group_into_documents(list(reversed(passages)), titles={}, limit=5)

    assert [hit.document_id for hit in first] == ["a", "z"]
    assert [hit.document_id for hit in first] == [hit.document_id for hit in second]


def test_ranks_are_stamped_from_one_and_the_limit_is_respected() -> None:
    passages = [_passage(f"{name}#1", name, score) for name, score in (("a", 0.9), ("b", 0.8), ("c", 0.7))]

    hits = group_into_documents(passages, titles={}, limit=2)

    assert [hit.rank for hit in hits] == [1, 2]
    assert len(hits) == 2


def test_a_document_with_no_known_title_falls_back_to_its_id() -> None:
    hits = group_into_documents([_passage("a#1", "a", 0.5)], titles={}, limit=1)
    assert hits[0].title == "a"


def test_candidate_images_are_deduplicated_across_a_documents_passages() -> None:
    hit = DocumentHit(
        document_id="a",
        title="A",
        score=1.0,
        passages=(
            _passage("a#1", "a", 0.9, image_ids=("i1", "i2")),
            _passage("a#2", "a", 0.8, image_ids=("i2", "i3")),
        ),
    )

    assert hit.candidate_image_ids == ("i1", "i2", "i3")


# --------------------------------------------------------------------------- #
# The lexical baseline
# --------------------------------------------------------------------------- #


async def test_the_baseline_ranks_the_document_that_shares_the_rare_terms(corpus: Corpus) -> None:
    retriever = LexicalCorpusRetriever(corpus)

    documents, diagnostics = await retriever.retrieve("insert alkaline batteries polarity", limit=3)

    assert documents[0].document_id == "200"
    assert diagnostics["method"] == "lexical"
    assert "alkaline" in diagnostics["terms_in_corpus"]


async def test_the_baseline_reports_terms_the_corpus_has_never_seen(corpus: Corpus) -> None:
    """A question that scores badly because none of its words exist is a different
    failure from one that was outranked, and the diagnostics have to say which."""
    _documents, diagnostics = await retriever_diagnostics(corpus, "coolant thermonuclear flibbertigibbet")

    assert diagnostics["terms_absent_from_corpus"] == ["thermonuclear", "flibbertigibbet"]
    assert "coolant" in diagnostics["terms_in_corpus"]


async def retriever_diagnostics(corpus: Corpus, question: str) -> tuple[tuple[DocumentHit, ...], dict[str, Any]]:
    return await LexicalCorpusRetriever(corpus).retrieve(question, limit=3)


async def test_the_baselines_offsets_come_from_the_bundle_and_need_no_recovery(corpus: Corpus) -> None:
    documents, _ = await retriever_diagnostics(corpus, "drain the radiator flush connector")

    passage = documents[0].best_passage
    assert passage.offset_source is OffsetSource.BUNDLE
    assert passage.has_offsets
    document = corpus.document(passage.document_id)
    assert document is not None
    assert document.text[passage.char_start : passage.char_end] == passage.text


async def test_the_baseline_reports_no_semantic_score_rather_than_zero(corpus: Corpus) -> None:
    """Zero would read as "the vector search found nothing" instead of "there was none"."""
    documents, _ = await retriever_diagnostics(corpus, "coolant reservoir access panel")

    passage = documents[0].best_passage
    assert passage.semantic_score is None
    assert passage.lexical_score == pytest.approx(passage.score)


async def test_an_empty_question_returns_nothing_and_says_why(corpus: Corpus) -> None:
    documents, diagnostics = await retriever_diagnostics(corpus, "   ")

    assert documents == ()
    assert diagnostics["reason"] == "empty query"


async def test_a_question_matching_nothing_returns_no_documents(corpus: Corpus) -> None:
    """An empty result is a legitimate answer, not an error."""
    documents, _ = await retriever_diagnostics(corpus, "zzzz qqqq xxxx")

    assert documents == ()


def test_the_baseline_names_its_method_and_its_corpus(corpus: Corpus) -> None:
    retriever = LexicalCorpusRetriever(corpus)

    assert retriever.method is RetrievalMethod.LEXICAL
    assert retriever.corpus_id == "demo"
    assert isinstance(retriever, DocumentRetriever)


async def test_the_baseline_health_names_what_it_searched(corpus: Corpus) -> None:
    assert "BM25" in await LexicalCorpusRetriever(corpus).health()


# --------------------------------------------------------------------------- #
# Image selection
# --------------------------------------------------------------------------- #


def _hits_for(corpus: Corpus, *document_ids: str) -> list[DocumentHit]:
    """Document hits whose passages carry every image of each document."""
    hits: list[DocumentHit] = []
    for rank, document_id in enumerate(document_ids, start=1):
        image_ids = tuple(record.image_id for record in corpus.images_for(document_id))
        hits.append(
            DocumentHit(
                document_id=document_id,
                title=document_id,
                score=1.0 / rank,
                passages=(_passage(f"{document_id}#1", document_id, 1.0 / rank, image_ids=image_ids),),
                rank=rank,
            )
        )
    return hits


def test_the_document_order_baseline_ignores_the_question(corpus: Corpus) -> None:
    """A baseline that peeked at the question would not be the baseline being argued
    against."""
    selector = DocumentOrderImageSelector(corpus)
    documents = _hits_for(corpus, "200")

    first, _ = selector.select("insert batteries", documents, limit=5)
    second, diagnostics = selector.select("something completely unrelated", documents, limit=5)

    assert [hit.image_id for hit in first] == [hit.image_id for hit in second]
    assert diagnostics["reads"] == "nothing; position only"


def test_the_document_order_baseline_takes_the_best_documents_images_first(corpus: Corpus) -> None:
    selector = DocumentOrderImageSelector(corpus)

    hits, _ = selector.select("anything", _hits_for(corpus, "200", "100"), limit=3)

    assert [hit.image_id for hit in hits] == ["Remote_0", "Remote_1", "Cooling_0"]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_the_document_order_baseline_does_not_repeat_an_image(corpus: Corpus) -> None:
    selector = DocumentOrderImageSelector(corpus)

    hits, _ = selector.select("anything", _hits_for(corpus, "200", "200"), limit=10)

    assert len({hit.image_id for hit in hits}) == len(hits)


def test_context_window_ranking_puts_the_relevant_image_first(corpus: Corpus) -> None:
    """The finding the three-method split exists to isolate.

    Both selectors see the same winning document; only this one reads the text around
    each marker, and only this one can therefore prefer the battery figure over the
    cover figure.
    """
    documents = _hits_for(corpus, "200")

    ordered, _ = DocumentOrderImageSelector(corpus).select("insert alkaline batteries", documents, limit=2)
    ranked, diagnostics = ContextWindowImageSelector(corpus).select("insert alkaline batteries", documents, limit=2)

    assert ordered[0].image_id == "Remote_0"
    assert ranked[0].image_id == "Remote_1"
    assert diagnostics["strategy"] == "context-window"


def test_context_window_ranking_explains_itself_with_the_words_that_scored(corpus: Corpus) -> None:
    hits, _ = ContextWindowImageSelector(corpus).select("alkaline polarity", _hits_for(corpus, "200"), limit=1)

    assert "alkaline" in hits[0].matched_terms
    assert "alkaline" in hits[0].reason
    assert "alkaline" in hits[0].evidence.lower()


def test_a_term_common_to_most_windows_is_kept_out_of_the_explanation(corpus: Corpus) -> None:
    """Stopwords contributed rounding error to the score and dominated the reason.

    Measured before the filter: the top image for a battery question reported nine
    shared terms, four of which were "the", "of", "in" and "to".
    """
    selector = ContextWindowImageSelector(corpus)

    hits, diagnostics = selector.select("the of and insert alkaline", _hits_for(corpus, "200"), limit=1)

    assert hits
    assert "alkaline" in hits[0].matched_terms
    assert "the" not in hits[0].matched_terms
    assert "the" not in diagnostics["informative_terms"]


def test_a_question_of_only_common_words_selects_no_image_and_reports_it(corpus: Corpus) -> None:
    """Selecting nothing is correct here, and must not be backfilled with the
    document's images: 40 of MRAMG-Manual's 390 questions expect no image at all."""
    selector = ContextWindowImageSelector(corpus)

    hits, diagnostics = selector.select("the the the", _hits_for(corpus, "200"), limit=5)

    assert hits == ()
    assert diagnostics["candidates"] > 0
    assert diagnostics["selected_none_despite_candidates"] is True


def test_context_window_ranking_only_considers_images_the_passages_reached(corpus: Corpus) -> None:
    """On a corpus averaging 54 images per document, widening to the whole document
    would let the ranking pick from a section the question never touched."""
    narrow = DocumentHit(
        document_id="100",
        title="Cooling manual",
        score=1.0,
        passages=(_passage("100#1", "100", 1.0, image_ids=("Cooling_0",)),),
        rank=1,
    )

    hits, diagnostics = ContextWindowImageSelector(corpus).select("coolant radiator flush", [narrow], limit=5)

    assert {hit.image_id for hit in hits} <= {"Cooling_0"}
    assert diagnostics["candidates"] == 1


def test_an_image_from_a_better_ranked_document_wins_an_otherwise_equal_match(corpus: Corpus) -> None:
    selector = ContextWindowImageSelector(corpus)
    question = "coolant radiator flush connector"

    first, _ = selector.select(question, _hits_for(corpus, "100", "300"), limit=1)
    flipped, _ = selector.select(question, _hits_for(corpus, "300", "100"), limit=1)

    assert first[0].document_id == "100"
    assert first[0].score > flipped[0].score or flipped[0].document_id == "100"


def test_both_selectors_satisfy_the_protocol(corpus: Corpus) -> None:
    assert isinstance(DocumentOrderImageSelector(corpus), ImageSelector)
    assert isinstance(ContextWindowImageSelector(corpus), ImageSelector)


def test_the_informative_threshold_is_the_idf_of_a_term_in_half_the_corpus() -> None:
    """Not tuned. It is the point at which a term stops distinguishing anything."""
    assert MIN_INFORMATIVE_IDF == pytest.approx(0.6931471805599453)


# --------------------------------------------------------------------------- #
# The hybrid retriever, against a fake RAGFlow
# --------------------------------------------------------------------------- #


class FakeRAGFlow:
    """Just enough of the client for the hybrid path, recording what was asked."""

    def __init__(self, *, chunks: list[dict[str, Any]] | None = None, dataset_id: str | None = "ds-1") -> None:
        self._chunks = chunks if chunks is not None else []
        self._dataset_id = dataset_id
        self.requests: list[dict[str, Any]] = []

    async def find_dataset_id(self, name: str) -> str | None:
        self.requests.append({"find_dataset_id": name})
        return self._dataset_id

    async def retrieve(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {"chunks": self._chunks, "total": len(self._chunks)}


def _ragflow_chunk(document_id: str, text: str, similarity: float, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"rf-{document_id}-{abs(hash(text)) % 10_000}",
        "content": text,
        "similarity": similarity,
        "term_similarity": similarity * 0.4,
        "vector_similarity": similarity * 0.6,
        "document_metadata": {"document_id": document_id, "corpus_id": "demo"},
        **extra,
    }


def _hybrid(corpus: Corpus, client: FakeRAGFlow) -> HybridCorpusRetriever:
    return HybridCorpusRetriever(
        client,  # type: ignore[arg-type]
        corpus,
        dataset_name=dataset_name_for(corpus.corpus_id),
    )


async def test_the_hybrid_retriever_asks_exactly_one_dataset(corpus: Corpus) -> None:
    """The isolation guarantee, as the only value this field can hold."""
    client = FakeRAGFlow(chunks=[_ragflow_chunk("100", "Check the coolant level before starting the engine.", 0.7)])

    await _hybrid(corpus, client).retrieve("coolant", limit=3)

    retrieval = next(request for request in client.requests if "question" in request)
    assert retrieval["dataset_ids"] == ["ds-1"]
    assert len(retrieval["dataset_ids"]) == 1


async def test_the_hybrid_retriever_recovers_offsets_for_every_passage(corpus: Corpus) -> None:
    text = "Insert two alkaline batteries observing the polarity markings inside."
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", text, 0.8)])

    documents, diagnostics = await _hybrid(corpus, client).retrieve("batteries", limit=3)

    passage = documents[0].best_passage
    assert passage.offset_source is OffsetSource.EXACT_MATCH
    document = corpus.document("200")
    assert document is not None
    assert document.text[passage.char_start : passage.char_end] == text
    assert diagnostics["offset_sources"] == {"exact_match": 1}


async def test_a_whitespace_damaged_chunk_is_still_citable(corpus: Corpus) -> None:
    """Half the real corpus's passages come back like this.

    The damaged text is derived from the document rather than written out, so it stays a
    genuine span of it: a hand-written version would silently become a
    not-a-substring test the first time the fixture changed.
    """
    document = corpus.document("200")
    assert document is not None
    span = document.text[300:700]
    damaged = " ".join(span.split())
    assert damaged != span, "the span must actually contain collapsible whitespace"

    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", damaged, 0.8)])

    documents, diagnostics = await _hybrid(corpus, client).retrieve("battery cover", limit=3)

    passage = documents[0].best_passage
    assert passage.offset_source is OffsetSource.NORMALISED_MATCH
    assert passage.has_offsets
    assert diagnostics["offset_sources"] == {"normalised_match": 1}
    # The recovered span is the real one, not the projected one.
    assert " ".join(document.text[passage.char_start : passage.char_end].split()) == damaged


async def test_an_unlocatable_chunk_is_reported_and_carries_no_offsets(corpus: Corpus) -> None:
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", "Text that this document does not contain at all.", 0.8)])

    documents, diagnostics = await _hybrid(corpus, client).retrieve("anything", limit=3)

    passage = documents[0].best_passage
    assert passage.offset_source is OffsetSource.UNRESOLVED
    assert not passage.has_offsets
    assert diagnostics["problems"]
    assert "document 200" in diagnostics["problems"][0]


async def test_an_unlocatable_chunk_offers_no_images_rather_than_the_whole_document(corpus: Corpus) -> None:
    """Showing every image of a manual because one passage could not be placed is the
    failure Part 5 of the brief rules out."""
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", "Text this document does not contain anywhere.", 0.8)])

    documents, _ = await _hybrid(corpus, client).retrieve("anything", limit=3)

    assert documents[0].best_passage.image_ids == ()
    assert documents[0].candidate_image_ids == ()


async def test_images_come_only_from_inside_the_located_span(corpus: Corpus) -> None:
    text = "Insert two alkaline batteries observing the polarity markings inside."
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", text, 0.8)])

    documents, _ = await _hybrid(corpus, client).retrieve("batteries", limit=3)

    passage = documents[0].best_passage
    for image_id in passage.image_ids:
        record = corpus.image(image_id)
        assert record is not None
        assert passage.char_start <= record.char_offset < passage.char_end


async def test_a_chunk_from_a_document_the_corpus_does_not_have_is_dropped_and_counted(corpus: Corpus) -> None:
    """The only way this happens is a dataset holding something the bundle does not,
    which is the isolation guarantee failing. It must be visible in the result."""
    client = FakeRAGFlow(
        chunks=[
            _ragflow_chunk("999", "A passage from a document that is not in this corpus at all.", 0.9),
            _ragflow_chunk("100", "Check the coolant level before starting the engine.", 0.7),
        ]
    )

    documents, diagnostics = await _hybrid(corpus, client).retrieve("coolant", limit=3)

    assert [hit.document_id for hit in documents] == ["100"]
    assert diagnostics["chunks_from_unknown_documents"] == 1
    assert any("999" in problem for problem in diagnostics["problems"])


async def test_a_document_id_is_recovered_from_the_filename_when_metadata_is_absent(corpus: Corpus) -> None:
    chunk = _ragflow_chunk("100", "Check the coolant level before starting the engine.", 0.7)
    chunk["document_metadata"] = {}
    chunk["document_keyword"] = "demo-100.md"
    client = FakeRAGFlow(chunks=[chunk])

    documents, _ = await _hybrid(corpus, client).retrieve("coolant", limit=3)

    assert [hit.document_id for hit in documents] == ["100"]


async def test_a_section_label_from_a_flattened_chunk_is_trimmed(corpus: Corpus) -> None:
    """The bug observed live on document 50024, whose citation position rendered as a
    page of installation steps.

    The chunk here is a flattened heading followed by its section, which is what these
    documents actually look like: one "line" holding both.
    """
    flattened = "# Remote control  Slide the battery cover downwards to open the compartment, " + ("then " * 60)
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", flattened, 0.8)])

    documents, _ = await _hybrid(corpus, client).retrieve("battery", limit=3)

    section = documents[0].best_passage.section
    assert section == "Remote control"
    assert len(section) <= MAX_SECTION_CHARS


async def test_an_empty_question_short_circuits_without_calling_ragflow(corpus: Corpus) -> None:
    client = FakeRAGFlow()

    documents, diagnostics = await _hybrid(corpus, client).retrieve("  ", limit=3)

    assert documents == ()
    assert diagnostics["reason"] == "empty query"
    assert client.requests == []


async def test_a_missing_dataset_is_an_error_with_the_command_that_fixes_it(corpus: Corpus) -> None:
    """ "Nothing matched" and "the corpus was never indexed" must not look the same."""
    client = FakeRAGFlow(dataset_id=None)

    with pytest.raises(RetrievalError, match="make corpus-ingest"):
        await _hybrid(corpus, client).retrieve("coolant", limit=3)


async def test_the_dataset_id_is_resolved_once_and_cached(corpus: Corpus) -> None:
    client = FakeRAGFlow(chunks=[])
    retriever = _hybrid(corpus, client)

    await retriever.retrieve("coolant", limit=3)
    await retriever.retrieve("batteries", limit=3)

    assert sum(1 for request in client.requests if "find_dataset_id" in request) == 1


async def test_the_hybrid_retriever_names_the_dataset_it_would_use(corpus: Corpus) -> None:
    retriever = _hybrid(corpus, FakeRAGFlow())

    assert retriever.dataset_name == "agentpath-demo"
    assert retriever.method is RetrievalMethod.HYBRID
    assert retriever.corpus_id == "demo"
    assert "agentpath-demo" in await retriever.health()


async def test_an_unreachable_engine_is_reported_as_unavailable(corpus: Corpus) -> None:
    from agentpath.retrieval.ragflow_client import RAGFlowUnavailable

    class _Down(FakeRAGFlow):
        async def find_dataset_id(self, name: str) -> str | None:
            raise RAGFlowUnavailable("connection refused")

    with pytest.raises(RetrievalUnavailable):
        await _hybrid(corpus, _Down()).retrieve("coolant", limit=3)


# --------------------------------------------------------------------------- #
# The service, and the registry of three methods
# --------------------------------------------------------------------------- #


def test_without_ragflow_only_the_baseline_exists(corpus: Corpus, settings: Settings) -> None:
    """A deployment with no RAGFlow gets a working baseline and an honestly absent
    hybrid, not a method that raises on first use."""
    methods = build_methods(corpus, settings=settings, client=None)

    assert list(methods) == [RetrievalMethod.LEXICAL]
    assert default_method(methods) is RetrievalMethod.LEXICAL


def test_with_ragflow_all_three_exist_and_the_default_is_multimodal(corpus: Corpus, settings: Settings) -> None:
    methods = build_methods(corpus, settings=settings, client=FakeRAGFlow())  # type: ignore[arg-type]

    assert set(methods) == {RetrievalMethod.LEXICAL, RetrievalMethod.HYBRID, RetrievalMethod.MULTIMODAL}
    assert default_method(methods) is RetrievalMethod.MULTIMODAL


def test_hybrid_and_multimodal_differ_only_in_image_selection(corpus: Corpus, settings: Settings) -> None:
    """One retriever instance is a stronger guarantee of this than two configured
    identically, so the test asserts the sharing rather than the configuration."""
    methods = build_methods(corpus, settings=settings, client=FakeRAGFlow())  # type: ignore[arg-type]

    hybrid = methods[RetrievalMethod.HYBRID]
    multimodal = methods[RetrievalMethod.MULTIMODAL]

    assert hybrid._retriever is multimodal._retriever
    assert hybrid.image_strategy == "document-order"
    assert multimodal.image_strategy == "context-window"


def test_lexical_and_hybrid_share_the_image_selector(corpus: Corpus, settings: Settings) -> None:
    methods = build_methods(corpus, settings=settings, client=FakeRAGFlow())  # type: ignore[arg-type]

    assert methods[RetrievalMethod.LEXICAL]._images is methods[RetrievalMethod.HYBRID]._images


def test_a_corpus_with_no_method_at_all_cannot_supply_a_default() -> None:
    with pytest.raises(ValueError, match="no retrieval method"):
        default_method({})


async def test_the_service_times_retrieval_and_image_selection_apart(corpus: Corpus, settings: Settings) -> None:
    """Folding them together would let one hide the other."""
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]

    result = await service.search("insert alkaline batteries", limit=2, image_limit=2)

    assert result.retrieval_ms > 0
    assert result.total_ms >= result.retrieval_ms
    assert "image_selection" in str(result.diagnostics)


async def test_the_service_echoes_the_corpus_and_method_that_produced_the_result(
    corpus: Corpus, settings: Settings
) -> None:
    """A results table whose rows cannot prove which method wrote them is not evidence."""
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]

    result = await service.search("coolant radiator", limit=1)

    assert result.corpus_id == "demo"
    assert result.method is RetrievalMethod.LEXICAL
    assert result.question == "coolant radiator"
    assert result.diagnostics["image_retrieval_mode"] == "context-window"


async def test_a_citation_opens_the_document_at_the_best_passage(corpus: Corpus, settings: Settings) -> None:
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]
    result = await service.search("drain the radiator flush connector", limit=1)

    citations = service.citations(result)

    assert len(citations) == 1
    trail = citations[0]
    assert trail.corpus_id == "demo"
    assert trail.document_id == "100"
    assert trail.kind is SourceKind.LOCAL_DOCUMENT
    assert trail.link() == f"/api/sources/demo-100#char={trail.locator.char_start}"


async def test_a_citation_for_a_document_with_an_upstream_url_prefers_it(corpus: Corpus, settings: Settings) -> None:
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]
    result = await service.search("insert alkaline batteries polarity", limit=1)

    trail = service.citations(result)[0]

    assert trail.kind is SourceKind.EXTERNAL_URL
    assert trail.link() is not None
    assert trail.link().startswith("https://example.test/remote")


async def test_a_citation_without_recoverable_offsets_carries_no_locator(corpus: Corpus) -> None:
    """The document opens at the top and the page says so. A fabricated offset would
    look precise and be wrong.

    Document 100 is used because it records no upstream URL, so the link is this API's
    own viewer and the absence of a ``#char=`` fragment is visible in it.
    """
    client = FakeRAGFlow(chunks=[_ragflow_chunk("100", "Text this document does not contain anywhere.", 0.8)])
    service = CorpusSearchService(
        corpus=corpus,
        retriever=_hybrid(corpus, client),
        images=DocumentOrderImageSelector(corpus),
        method=RetrievalMethod.HYBRID,
    )

    result = await service.search("anything", limit=1, image_limit=0)
    trail = service.citations(result)[0]

    assert trail.locator.char_start is None
    assert trail.locator.is_precise is False
    assert trail.link() == "/api/sources/demo-100"
    assert "#char=" not in trail.link()


async def test_a_citation_carries_only_the_selected_images_of_its_own_document(
    corpus: Corpus, settings: Settings
) -> None:
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]

    result = await service.search("insert alkaline batteries", limit=3, image_limit=5)

    for hit in result.documents:
        trail = service.citation_for(hit, result)
        assert trail is not None
        for image in trail.images:
            assert image.document_id == hit.document_id
            assert image.image_id in result.image_ids


async def test_a_selected_image_whose_file_vanished_is_reported_as_a_problem(
    corpus: Corpus, settings: Settings
) -> None:
    """The evaluation reports a broken-image rate, and a rate computed from the
    startup registry would only ever report zero."""
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]
    record = corpus.image("Remote_1")
    assert record is not None
    record.path.unlink()

    result = await service.search("insert alkaline batteries polarity", limit=1, image_limit=5)

    assert any("Remote_1" in problem for problem in result.problems)


async def test_zero_image_limit_returns_no_images(corpus: Corpus, settings: Settings) -> None:
    service = build_methods(corpus, settings=settings, client=None)[RetrievalMethod.LEXICAL]

    result = await service.search("insert alkaline batteries", limit=2, image_limit=0)

    assert result.images == ()
    assert result.image_ids == ()


# --------------------------------------------------------------------------- #
# The route: one corpus per request, and no way to ask for two
# --------------------------------------------------------------------------- #


async def api_for(corpus: Corpus, settings: Settings, *, with_ragflow: bool = False):
    lexicon = load_lexicon()

    class _NoopRetriever:
        mode = "hybrid"

        async def retrieve(self, *_args, **_kwargs):  # pragma: no cover - unused
            raise AssertionError("these tests do not use the procedures path")

        async def health(self):  # pragma: no cover - unused
            raise AssertionError("these tests do not use the procedures path")

    class _NoopClient:
        async def aclose(self) -> None:
            return None

    retriever = _NoopRetriever()
    client = FakeRAGFlow(chunks=[_ragflow_chunk("200", "Insert two alkaline batteries observing", 0.8)])
    context = ServiceContext(
        settings=settings,
        lexicon=lexicon,
        client=_NoopClient(),  # type: ignore[arg-type]
        retriever=retriever,  # type: ignore[arg-type]
        search=SearchService(retriever=retriever, lexicon=lexicon, settings=settings),  # type: ignore[arg-type]
        corpora=CorpusCatalogue(corpora={corpus.corpus_id: corpus}),
        corpus_methods={
            corpus.corpus_id: build_methods(
                corpus,
                settings=settings,
                client=client if with_ragflow else None,  # type: ignore[arg-type]
            )
        },
    )
    app = create_app(settings=settings, context=context)
    manager = LifespanManager(app)
    await manager.__aenter__()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"), manager


async def test_the_corpus_list_carries_the_licence_and_the_image_mode(corpus: Corpus, settings: Settings) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.get("/api/corpora")
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    body = response.json()
    entry = body["corpora"][0]
    assert entry["corpus_id"] == "demo"
    assert entry["licence"] == "CC0-1.0"
    assert entry["attribution"]
    assert entry["image_retrieval_mode"] == "context-window"
    assert entry["methods"] == ["lexical"]
    assert entry["default_method"] == "lexical"
    assert body["procedures_corpus_id"] == "synthetic-procedures"


async def test_the_corpus_list_offers_every_method_when_ragflow_is_present(corpus: Corpus, settings: Settings) -> None:
    client, manager = await api_for(corpus, settings, with_ragflow=True)
    try:
        body = (await client.get("/api/corpora")).json()
    finally:
        await manager.__aexit__(None, None, None)

    assert body["corpora"][0]["methods"] == ["lexical", "hybrid", "multimodal"]
    assert body["corpora"][0]["default_method"] == "multimodal"


async def test_a_search_echoes_the_corpus_and_method_and_cites_every_document(
    corpus: Corpus, settings: Settings
) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.post(
            "/api/corpus-search",
            json={"corpus_id": "demo", "question": "insert alkaline batteries", "limit": 2, "image_limit": 3},
        )
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 200
    body = response.json()
    assert body["corpus_id"] == "demo"
    assert body["method"] == "lexical"
    assert body["licence"] == "CC0-1.0"
    assert body["documents"]
    for document in body["documents"]:
        assert document["provenance"]["corpus_id"] == "demo"
        assert document["provenance"]["has_original"] is True
        assert document["provenance"]["open_url"]


async def test_every_returned_document_belongs_to_the_requested_corpus(corpus: Corpus, settings: Settings) -> None:
    """Mixing is unreachable rather than checked, and this is the assertion that says
    so at the edge."""
    client, manager = await api_for(corpus, settings)
    try:
        body = (
            await client.post(
                "/api/corpus-search",
                json={"corpus_id": "demo", "question": "coolant radiator batteries filter", "limit": 5},
            )
        ).json()
    finally:
        await manager.__aexit__(None, None, None)

    known = set(corpus.documents)
    assert {document["document_id"] for document in body["documents"]} <= known
    assert all(image["url"].startswith("/api/corpus-images/demo/") for image in body["images"])


async def test_only_selected_images_are_returned_not_every_image_in_the_document(
    corpus: Corpus, settings: Settings
) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        body = (
            await client.post(
                "/api/corpus-search",
                json={"corpus_id": "demo", "question": "insert alkaline batteries", "limit": 1, "image_limit": 1},
            )
        ).json()
    finally:
        await manager.__aexit__(None, None, None)

    assert len(body["images"]) == 1
    assert body["images"][0]["reason"]
    assert body["images"][0]["strategy"] == "document-order"


async def test_an_image_with_no_upstream_original_says_so_rather_than_inventing_one(
    corpus: Corpus, settings: Settings
) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        body = (
            await client.post(
                "/api/corpus-search",
                json={"corpus_id": "demo", "question": "insert alkaline batteries", "limit": 1, "image_limit": 5},
            )
        ).json()
    finally:
        await manager.__aexit__(None, None, None)

    by_id = {image["image_id"]: image for image in body["images"]}
    assert by_id["Remote_1"]["source_url"] is None
    assert by_id["Remote_0"]["source_url"].startswith("https://")


async def test_the_procedures_corpus_is_refused_by_name_not_by_a_generic_404(
    corpus: Corpus, settings: Settings
) -> None:
    """ "Wrong endpoint" and "no such corpus" are different mistakes."""
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.post(
            "/api/corpus-search", json={"corpus_id": "synthetic-procedures", "question": "anything"}
        )
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404
    assert response.json()["error"] == "wrong_endpoint_for_corpus"
    assert "/api/answer" in response.json()["message"]


async def test_an_unknown_corpus_is_a_404_that_lists_what_exists(corpus: Corpus, settings: Settings) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.post("/api/corpus-search", json={"corpus_id": "nope", "question": "anything"})
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404
    assert response.json()["error"] == "corpus_not_found"
    assert response.json()["available"] == ["demo"]


async def test_a_method_the_corpus_does_not_offer_is_a_404_that_lists_the_ones_it_does(
    corpus: Corpus, settings: Settings
) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.post(
            "/api/corpus-search",
            json={"corpus_id": "demo", "question": "anything", "method": "multimodal"},
        )
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 404
    assert response.json()["error"] == "method_not_available"
    assert response.json()["available"] == ["lexical"]


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"question": "anything"}, "corpus_id is required, so a caller must state what it searched"),
        ({"corpus_id": "demo"}, "question is required"),
        ({"corpus_id": "demo", "question": ""}, "an empty question is not a search"),
        ({"corpus_id": "demo", "question": "a", "method": "visual"}, "there is no visual method"),
        ({"corpus_id": "demo", "question": "a", "dataset_ids": ["x"]}, "extra fields are forbidden"),
        ({"corpus_id": "demo", "question": "a", "limit": 0}, "a limit of zero would return nothing"),
        ({"corpus_id": "demo", "question": "a", "image_limit": -1}, "a negative image limit is meaningless"),
    ],
)
async def test_a_malformed_request_is_refused_by_the_schema(
    corpus: Corpus, settings: Settings, payload: dict[str, Any], why: str
) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        response = await client.post("/api/corpus-search", json=payload)
    finally:
        await manager.__aexit__(None, None, None)

    assert response.status_code == 422, why


async def test_there_is_no_parameter_that_names_a_dataset(corpus: Corpus, settings: Settings) -> None:
    """The strongest statement of the isolation guarantee: a caller cannot reach past
    the corpus id to the index it maps to."""
    from agentpath.api.schemas import CorpusSearchRequest

    fields = set(CorpusSearchRequest.model_fields)

    assert fields == {"corpus_id", "question", "method", "limit", "image_limit"}
    assert not any("dataset" in field for field in fields)


async def test_a_passage_reports_how_its_offsets_were_established(corpus: Corpus, settings: Settings) -> None:
    client, manager = await api_for(corpus, settings)
    try:
        body = (
            await client.post(
                "/api/corpus-search",
                json={"corpus_id": "demo", "question": "coolant radiator flush", "limit": 1},
            )
        ).json()
    finally:
        await manager.__aexit__(None, None, None)

    passages = body["documents"][0]["passages"]
    assert passages
    assert all(passage["offset_source"] == "bundle" for passage in passages)
    assert all(passage["char_start"] is not None for passage in passages)

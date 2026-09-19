"""What a corpus retrieval method is, and what it returns.

Stage 2B1 has to compare three methods on the same corpus and report the
comparison honestly, so the three are three implementations of one protocol rather
than three programs. The protocol is deliberately small: a method takes a question
and a limit, and returns ranked documents, selected images, and enough diagnostics
to explain both.

Three separations the shapes enforce.

**Document retrieval is not image selection.** A method is a pair: a
:class:`DocumentRetriever` that ranks documents and an :class:`ImageSelector` that
picks images from the documents that won. Keeping them apart is what makes the
evaluation able to say *which* half a number belongs to - a poor Image Recall over
good documents is an image-ranking problem, and the same number over bad documents
is a retrieval problem. Collapsing them would make those two indistinguishable.

**Neither of them generates an answer.** Generation consumes a
:class:`CorpusRetrievalResult` and adds nothing to it. So retrieval can be measured
without a model loaded, which is the only way the retrieval table in the report can
be produced separately from the generation table.

**A method is bound to one corpus.** :attr:`DocumentRetriever.corpus_id` is fixed
at construction, and there is no parameter anywhere below this line that could
widen a search to a second corpus. Mixed results are not forbidden by a check; they
are unreachable.

Why a separate enum from :class:`agentpath.domain.RetrievalMode`
---------------------------------------------------------------
``RetrievalMode.HYBRID`` means "query understanding with the travel lexicon, then
RAGFlow, then reranking" - the Stage 2A procedures path. The hybrid method here is
not that: it does not apply the travel-domain lexicon, because expanding "how do I
reset the filter" with hotel and refund synonyms would inject noise into a device
manual and the expansion would be measured as if it were retrieval quality. Two
different things should not share one label, so they do not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class RetrievalMethod(StrEnum):
    """Which of the three methods produced a result."""

    LEXICAL = "lexical"
    """In-process BM25 over the exported chunks. No RAGFlow, no embeddings.

    The reference point. Its purpose is to make the other two answerable: "hybrid
    reaches 92%" means nothing without knowing what keyword matching alone reaches
    on the same questions.
    """

    HYBRID = "hybrid"
    """RAGFlow's blend of term and vector similarity over the corpus dataset.

    The architecture as it exists, pointed at an external corpus.
    """

    MULTIMODAL = "multimodal"
    """Hybrid documents, then context-window ranking to choose images.

    Named for what it does - it ranks images - and *not* for seeing them. No model
    on this path reads a pixel; see ``docs/RETRIEVAL_MODES.md``.
    """


class OffsetSource(StrEnum):
    """How a passage's character offsets were obtained.

    Reported per passage rather than assumed, because a citation with no offsets
    still opens the document - it just opens at the top - and the difference
    between "opened at the right paragraph" and "opened at the top" is the
    difference between a usable citation and a gesture. The evaluation counts these.
    """

    BUNDLE = "bundle"
    """The chunk is ours, so the offsets arrived with it. Exact by construction."""

    EXACT_MATCH = "exact_match"
    """The engine's text was found verbatim in the document."""

    NORMALISED_MATCH = "normalised_match"
    """Found after collapsing whitespace. The span is right; its edges may be off
    by the whitespace that was collapsed."""

    UNRESOLVED = "unresolved"
    """Not found. No offsets are reported rather than plausible ones."""


@dataclass(frozen=True, slots=True)
class CorpusPassage:
    """One retrieved span of one corpus document.

    ``score`` is engine-native and **not comparable across methods**: BM25 is
    unbounded and RAGFlow's similarity is a 0-1 blend. Normalising them onto a
    shared scale was rejected - dividing by the top score makes every method's best
    result 1.000, which reads as confidence and measures nothing. Ranks are what
    the methods are compared on, and ranks are scale-free.
    """

    chunk_id: str
    document_id: str
    corpus_id: str
    text: str
    score: float
    offset_source: OffsetSource
    char_start: int | None = None
    char_end: int | None = None
    section: str | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    #: Images whose placeholder falls inside this span, in document order. A
    #: candidate set for image selection, not a selection.
    image_ids: tuple[str, ...] = ()

    @property
    def has_offsets(self) -> bool:
        return self.char_start is not None and self.char_end is not None


@dataclass(frozen=True, slots=True)
class DocumentHit:
    """One document, assembled from the passages that matched inside it.

    Grouping is duplicated from ``agentpath.retrieval.aggregate`` rather than
    reused, and the duplication is about twenty lines. Reusing it would have meant
    putting a corpus document id into ``Passage.procedure_id`` and reporting
    document recall through a type called ``ProcedureHit``; the saving is not worth
    a label that lies in every log line and every evaluation column.
    """

    document_id: str
    title: str
    score: float
    passages: tuple[CorpusPassage, ...]
    rank: int = 0

    @property
    def best_passage(self) -> CorpusPassage:
        return self.passages[0]

    @property
    def matched_passage_count(self) -> int:
        return len(self.passages)

    @property
    def candidate_image_ids(self) -> tuple[str, ...]:
        """Images reachable through this document's matched passages, in order.

        The *candidates* an image selector may choose from on the evidence this
        document actually matched. A selector is free to widen to the whole
        document, and :class:`DocumentOrderImageSelector` does; naming the narrower
        set here is what lets the difference be stated.
        """
        seen: list[str] = []
        for passage in self.passages:
            for image_id in passage.image_ids:
                if image_id not in seen:
                    seen.append(image_id)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class ImageHit:
    """One image an :class:`ImageSelector` chose, and why.

    ``reason`` and ``evidence`` exist because "the system returned this image" is
    not a claim anyone can check. The window text that scored is carried so a
    reviewer can see what the ranking actually read - which for a caption-less
    corpus is the only thing it read.
    """

    image_id: str
    document_id: str
    score: float
    strategy: str
    reason: str
    rank: int = 0
    #: The text the selector scored, truncated for display. Empty when the
    #: strategy scored nothing (the document-order baseline).
    evidence: str = ""
    #: Terms shared by the query and the window, which is what the score is made of.
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusRetrievalResult:
    """Everything one method produced for one question.

    ``problems`` is part of the result rather than a log line: an unresolved offset
    or a missing image file is a fact about this answer's provenance, and the
    evaluation's broken-source and broken-image rates are computed from it.
    """

    corpus_id: str
    method: RetrievalMethod
    question: str
    documents: tuple[DocumentHit, ...] = ()
    images: tuple[ImageHit, ...] = ()
    #: Time inside the retrieval engine only, excluding image selection.
    retrieval_ms: float = 0.0
    #: Time spent choosing images.
    image_selection_ms: float = 0.0
    total_ms: float = 0.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @property
    def document_ids(self) -> tuple[str, ...]:
        """Ranked document ids, best first. What Document Recall is computed over."""
        return tuple(hit.document_id for hit in self.documents)

    @property
    def image_ids(self) -> tuple[str, ...]:
        """Ranked image ids, best first. What Image Recall is computed over."""
        return tuple(hit.image_id for hit in self.images)

    @property
    def top_document(self) -> DocumentHit | None:
        return self.documents[0] if self.documents else None

    @property
    def unresolved_offsets(self) -> int:
        return sum(
            1 for hit in self.documents for passage in hit.passages if passage.offset_source is OffsetSource.UNRESOLVED
        )


@runtime_checkable
class DocumentRetriever(Protocol):
    """Ranks the documents of one corpus for one question."""

    @property
    def method(self) -> RetrievalMethod:
        """The method this retriever is the document half of."""
        ...

    @property
    def corpus_id(self) -> str:
        """Fixed at construction. There is no parameter that can widen it."""
        ...

    async def retrieve(self, question: str, *, limit: int) -> tuple[tuple[DocumentHit, ...], dict[str, Any]]:
        """Return the top *limit* documents, best first, plus diagnostics.

        An empty result is a legitimate answer and must not raise: "no document
        matched" drives abstention, and turning it into an error would make
        abstention indistinguishable from a broken engine. Implementations raise
        :class:`agentpath.retrieval.ports.RetrievalUnavailable` only when the engine
        itself is unreachable.
        """
        ...

    async def health(self) -> str:
        """A one-line readiness statement, for the report and for ``/health``."""
        ...


@runtime_checkable
class ImageSelector(Protocol):
    """Chooses which images to show, given the documents that won.

    Deliberately synchronous and pure: every input is already in memory, so an
    async signature would promise IO that does not happen.
    """

    @property
    def strategy(self) -> str:
        """Names what this selector reads, for the UI and the evaluation table."""
        ...

    def select(
        self,
        question: str,
        documents: Sequence[DocumentHit],
        *,
        limit: int,
    ) -> tuple[tuple[ImageHit, ...], dict[str, Any]]:
        """Return the top *limit* images, best first, plus diagnostics."""
        ...


__all__ = [
    "CorpusPassage",
    "CorpusRetrievalResult",
    "DocumentHit",
    "DocumentRetriever",
    "ImageHit",
    "ImageSelector",
    "OffsetSource",
    "RetrievalMethod",
]

"""Choosing which images to show. Two strategies, named for what they read.

The brief asked to start from the benchmark's official image captions. That
starting point does not exist: all 2,599 MRAMG-Manual captions are the empty string,
verified over the whole subset (``docs/RETRIEVAL_MODES.md``). So the replacement
reads the document text around each image's ``<PIC>`` marker, and it is called
**context-window image retrieval** rather than caption-based so the name cannot be
mistaken for something it is not. It is also not visual retrieval: no model on this
path sees a pixel, and calling it multimodal would be a claim about the wrong thing.
The :class:`~agentpath.corpus_retrieval.ports.RetrievalMethod.MULTIMODAL` label
describes the *result* - text and images together - not a vision model.

Why two strategies rather than one. Part 5 of the brief forbids showing an image just
because it sits in a retrieved document, and the only way to demonstrate that the
ranking earns its place is to measure it against exactly that: the document-order
selector *is* "show what is in the document", kept as a baseline so the difference
between the two is a number in the evaluation table rather than an assertion here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import structlog

from agentpath.corpus import Corpus
from agentpath.corpus_retrieval.ports import DocumentHit, ImageHit
from agentpath.query.normalize import normalize, tokenize

logger = structlog.get_logger(__name__)

#: How many characters of the window travel back with the hit for display. Enough to
#: read the sentence that scored, short enough not to be a second copy of the
#: document in every response.
EVIDENCE_CHARS = 240

#: A term must be absent from at least half the windows to count. Below that it is
#: doing no work: its idf is ``ln(2)`` at exactly half and falls towards zero as it
#: approaches every window, so "the", "of" and "in" contribute rounding error to the
#: score while dominating the explanation shown to a reviewer.
#:
#: Measured before this existed: the top image for "How to insert batteries in remote
#: control of the air conditioner?" reported nine shared terms, of which four were
#: stopwords. The number is not tuned - it is the idf of a term present in exactly
#: half the corpus, which is the point at which a term stops distinguishing anything.
MIN_INFORMATIVE_IDF = math.log(2.0)

#: Terms named in an image's ``reason``. More than this and the line stops being a
#: reason and becomes a word list.
MAX_REASON_TERMS = 6


#: Documents deep into the ranking contribute images at a discount, so an image from
#: the winning document outranks an equally-worded image from the third-place one.
#: Reciprocal rank rather than a tuned curve: it needs no constants to defend, and it
#: encodes the one thing that is certainly true - rank 1 is more likely right than
#: rank 3.
def _rank_weight(rank: int) -> float:
    return 1.0 / max(rank, 1)


class DocumentOrderImageSelector:
    """Every image of the winning documents, in document order. No ranking.

    The naive baseline, and a fair representation of what a retrieval system that
    has not thought about images does. Its weakness is the point: MRAMG-Manual
    documents hold a median of 54 images, and 40 of the 390 questions expect no image
    at all, so "show the document's images" both floods the answer and cannot ever
    abstain.
    """

    strategy = "document-order"

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    def select(
        self,
        question: str,
        documents: Sequence[DocumentHit],
        *,
        limit: int,
    ) -> tuple[tuple[ImageHit, ...], dict[str, Any]]:
        """Take images from the best document first, in the order they appear.

        The question is ignored, and that is not an oversight: a baseline that peeked
        at the question would not be the baseline being argued against.
        """
        hits: list[ImageHit] = []
        # Checked before the loop, not only inside it. The inner guard fires *after* an
        # append, so a limit of zero used to yield exactly one image - and zero is a
        # value a caller passes on purpose, to retrieve documents without images.
        if limit <= 0:
            return (), self._diagnostics(documents, 0)

        seen: set[str] = set()
        for document in documents:
            for record in self._corpus.images_for(document.document_id):
                if record.image_id in seen:
                    continue
                seen.add(record.image_id)
                hits.append(
                    ImageHit(
                        image_id=record.image_id,
                        document_id=document.document_id,
                        # Positional, not a relevance estimate. Derived from the
                        # document's rank so the ordering is defined, and reported as
                        # a score only because the shape requires one.
                        score=_rank_weight(document.rank),
                        strategy=self.strategy,
                        reason=(
                            f"image {record.position_index + 1} of document {document.document_id}, "
                            f"which ranked {document.rank}"
                        ),
                        rank=len(hits) + 1,
                    )
                )
                if len(hits) >= limit:
                    return tuple(hits), self._diagnostics(documents, len(hits))
        return tuple(hits), self._diagnostics(documents, len(hits))

    def _diagnostics(self, documents: Sequence[DocumentHit], selected: int) -> dict[str, Any]:
        available = sum(len(self._corpus.images_for(hit.document_id)) for hit in documents)
        return {
            "strategy": self.strategy,
            "reads": "nothing; position only",
            "images_available": available,
            "images_selected": selected,
        }


class ContextWindowImageSelector:
    """Ranks images by how well the text around them matches the question.

    The scoring is deliberately simple and deliberately explainable: the terms shared
    by the question and the window, weighted by how rare each term is across all of
    this corpus's windows, so "filter" in a vacuum-cleaner manual counts for less than
    "HEPA". An embedding of each window would likely score better and would cost the
    ability to say *which words* made an image win, which is the only thing that makes
    a returned image checkable by a reviewer. That trade-off is recorded rather than
    silently taken: it is the obvious next experiment.

    Candidates come from the passages that actually matched, not from the whole
    document. On a corpus with 54 images per document, widening to the document would
    let the ranking pick an image from a section the question never touched.
    """

    strategy = "context-window"

    def __init__(self, corpus: Corpus, *, minimum_score: float = 0.0) -> None:
        self._corpus = corpus
        self._minimum_score = minimum_score
        # Built once per corpus, at startup: the idf of a window term is a property
        # of the corpus, and computing it per request would mean tokenising 2,599
        # windows inside every search.
        self._windows: dict[str, list[str]] = {}
        frequency: Counter[str] = Counter()
        for image_id in corpus.images:
            tokens = tokenize(normalize(corpus.image_context(image_id)))
            self._windows[image_id] = tokens
            frequency.update(set(tokens))
        total = len(self._windows)
        self._idf = {term: math.log(1.0 + (total - count + 0.5) / (count + 0.5)) for term, count in frequency.items()}
        logger.info(
            "corpus.image_windows_built",
            corpus=corpus.corpus_id,
            images=total,
            vocabulary=len(self._idf),
            empty_windows=sum(1 for tokens in self._windows.values() if not tokens),
        )

    def select(
        self,
        question: str,
        documents: Sequence[DocumentHit],
        *,
        limit: int,
    ) -> tuple[tuple[ImageHit, ...], dict[str, Any]]:
        query_terms = set(tokenize(normalize(question)))
        candidates: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for document in documents:
            for image_id in document.candidate_image_ids:
                if image_id in seen:
                    continue
                seen.add(image_id)
                candidates.append((image_id, document.document_id, document.rank))

        # Filtered once, before scoring, so a term that cannot influence the ranking
        # cannot appear in the explanation either. A question made entirely of common
        # words leaves nothing to score, and selecting no image is then the correct
        # outcome rather than a ranking of stopword overlap.
        informative = {term for term in query_terms if self._idf.get(term, 0.0) >= MIN_INFORMATIVE_IDF}

        scored: list[tuple[float, str, str, tuple[str, ...]]] = []
        for image_id, document_id, rank in candidates:
            tokens = self._windows.get(image_id, [])
            if not tokens:
                continue
            matched = informative.intersection(tokens)
            if not matched:
                continue
            overlap = sum(self._idf[term] for term in matched)
            # Divided by the window's length in the mildest way that works: the
            # square root, so a long window is not rewarded for merely containing
            # more words while a short window is not made unbeatable.
            score = overlap / math.sqrt(len(tokens)) * _rank_weight(rank)
            if score <= self._minimum_score:
                continue
            # Ordered by how much each term contributed, so the reason leads with the
            # word that actually won rather than with whichever sorted first.
            ranked_terms = tuple(sorted(matched, key=lambda term: (-self._idf[term], term)))
            scored.append((score, image_id, document_id, ranked_terms))

        scored.sort(key=lambda row: (-row[0], row[1]))
        hits = tuple(
            ImageHit(
                image_id=image_id,
                document_id=document_id,
                score=score,
                strategy=self.strategy,
                reason=(
                    f"window shares {len(matched)} informative term(s) with the question: "
                    f"{', '.join(matched[:MAX_REASON_TERMS])}"
                ),
                rank=index,
                evidence=self._corpus.image_context(image_id)[:EVIDENCE_CHARS],
                matched_terms=matched,
            )
            for index, (score, image_id, document_id, matched) in enumerate(scored[:limit], start=1)
        )

        return hits, {
            "strategy": self.strategy,
            "reads": "document text around the <PIC> marker; captions are empty in this corpus",
            "candidates": len(candidates),
            "scored": len(scored),
            "images_selected": len(hits),
            "query_terms": len(query_terms),
            "informative_terms": sorted(informative),
            # An empty selection over non-empty candidates is a real and correct
            # outcome - 40 of the 390 questions expect no image - so it is reported
            # rather than backfilled with the document's images.
            "selected_none_despite_candidates": bool(candidates) and not hits,
        }


__all__ = [
    "EVIDENCE_CHARS",
    "MAX_REASON_TERMS",
    "MIN_INFORMATIVE_IDF",
    "ContextWindowImageSelector",
    "DocumentOrderImageSelector",
]

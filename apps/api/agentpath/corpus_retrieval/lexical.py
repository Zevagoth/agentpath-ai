"""The lexical baseline: Okapi BM25 over the exported chunks, in this process.

Why a baseline exists at all. "Hybrid retrieval reaches 92% Document Recall@3" is
not a result, it is a number. It becomes a result next to what keyword matching
alone reaches on the same questions with the same grouping: if BM25 also reaches
92%, the embedder, the vector index and the 571 seconds of parsing bought nothing on
this corpus, and that is a finding worth printing rather than hiding.

Why it is implemented here rather than installed. BM25 is forty lines and one
dependency saved, and - more to the point - a library's BM25 would come with its own
tokeniser, and the comparison would then be partly a comparison of tokenisers. This
uses ``agentpath.query.normalize``, the same normalisation the production query path
uses, so the two methods see the same words.

Why it reads the bundle and not RAGFlow. The baseline has to be independent of the
thing it is a baseline for. It scores the 1,506 chunks this repository computed,
whose offsets are known exactly, which also makes it the one method whose citations
need no offset recovery.

Parameters are the standard ones, ``k1=1.2`` and ``b=0.75``. They are not tuned:
tuning the baseline against the same questions the other methods are graded on would
make it a fourth method rather than a reference, and tuning anything on the held-out
split is forbidden outright.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

import structlog

from agentpath.corpus import Corpus
from agentpath.corpus_retrieval.grouping import group_into_documents
from agentpath.corpus_retrieval.ports import (
    CorpusPassage,
    DocumentHit,
    OffsetSource,
    RetrievalMethod,
)
from agentpath.query.normalize import normalize, tokenize

logger = structlog.get_logger(__name__)

#: Term-frequency saturation. The standard value. Above it, a term repeated ten
#: times counts nearly ten times; at 1.2 the fourth occurrence adds little, which is
#: the behaviour that makes BM25 better than raw counting.
BM25_K1 = 1.2

#: Length normalisation. The standard value. Matters more here than usual: chunks
#: are cut to about 1,400 characters but the last chunk of a document can be much
#: shorter, and without this a two-line tail chunk would outrank a full section for
#: containing the query term once.
BM25_B = 0.75

#: Candidate chunks scored down to before grouping. Generous: several chunks of one
#: document routinely rank together, so the document count after grouping is much
#: smaller than the chunk count before it.
CANDIDATE_MULTIPLIER = 8
MIN_CANDIDATES = 40


class LexicalCorpusRetriever:
    """Ranks a corpus's documents by BM25 over its exported chunks.

    The index is built once, at construction, because construction happens at
    startup and scoring happens in a request. For MRAMG-Manual that is 1,506 chunks
    and about 300,000 tokens; the build time is logged rather than assumed.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        started = time.perf_counter()

        self._chunk_ids: list[str] = []
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for chunk_id, chunk in corpus.chunks.items():
            tokens = tokenize(normalize(chunk.text))
            self._chunk_ids.append(chunk_id)
            counts = Counter(tokens)
            self._term_frequencies.append(counts)
            self._lengths.append(len(tokens))
            document_frequency.update(counts.keys())

        count = len(self._chunk_ids)
        self._average_length = (sum(self._lengths) / count) if count else 0.0
        # Precomputed per term: the idf is a property of the index, not of a query,
        # and recomputing a logarithm per query term per request buys nothing.
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._build_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "corpus.lexical_index_built",
            corpus=corpus.corpus_id,
            chunks=count,
            vocabulary=len(self._idf),
            average_chunk_tokens=round(self._average_length, 1),
            ms=round(self._build_ms, 1),
        )

    # -- identity ----------------------------------------------------------- #

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.LEXICAL

    @property
    def corpus_id(self) -> str:
        return self._corpus.corpus_id

    @property
    def build_ms(self) -> float:
        return self._build_ms

    # -- retrieval ---------------------------------------------------------- #

    async def retrieve(self, question: str, *, limit: int) -> tuple[tuple[DocumentHit, ...], dict[str, Any]]:
        """Score every chunk, keep the best, group into documents.

        ``async`` to satisfy the protocol, not because anything awaits: this method
        does no IO at all, which is exactly what makes it a usable reference - its
        latency column measures scoring and nothing else.
        """
        terms = tokenize(normalize(question))
        if not terms or not self._chunk_ids:
            return (), {
                "method": self.method.value,
                "reason": "empty query" if not terms else "empty index",
                "candidates_scored": 0,
                "query_terms": list(terms),
            }

        # Terms absent from the whole corpus contribute nothing and are reported, so
        # a question that scores badly because none of its words are in the corpus
        # looks different from one that was simply outranked.
        known = [term for term in terms if term in self._idf]
        scores = self._score(known)

        candidates = max(limit * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
        ranked = sorted(scores.items(), key=lambda row: (-row[1], self._chunk_ids[row[0]]))[:candidates]

        passages = tuple(self._passage(index, score) for index, score in ranked if score > 0.0)
        titles = {document_id: record.title for document_id, record in self._corpus.documents.items()}
        documents = group_into_documents(passages, titles=titles, limit=limit)

        return documents, {
            "method": self.method.value,
            "query_terms": list(terms),
            "terms_in_corpus": known,
            "terms_absent_from_corpus": [term for term in terms if term not in self._idf],
            "chunks_indexed": len(self._chunk_ids),
            "candidates_scored": len(scores),
            "passages_kept": len(passages),
            "k1": BM25_K1,
            "b": BM25_B,
        }

    def _score(self, terms: list[str]) -> dict[int, float]:
        """BM25 for every chunk containing at least one query term.

        Only those chunks: a chunk sharing no term scores exactly zero, and there is
        no reason to visit 1,506 chunks to establish that for most of them.
        """
        scores: dict[int, float] = {}
        if not terms or self._average_length <= 0:
            return scores

        for term in terms:
            idf = self._idf[term]
            for index, counts in enumerate(self._term_frequencies):
                frequency = counts.get(term)
                if not frequency:
                    continue
                length_ratio = self._lengths[index] / self._average_length
                denominator = frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio)
                scores[index] = scores.get(index, 0.0) + idf * (frequency * (BM25_K1 + 1.0)) / denominator
        return scores

    def _passage(self, index: int, score: float) -> CorpusPassage:
        """One scored chunk as a passage.

        Offsets come straight from the bundle - this is our own chunk - so this is
        the one path where a citation needs no recovery and
        :attr:`OffsetSource.BUNDLE` is the honest label.
        """
        chunk_id = self._chunk_ids[index]
        chunk = self._corpus.chunks[chunk_id]
        return CorpusPassage(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            corpus_id=chunk.corpus_id,
            text=chunk.text,
            score=score,
            offset_source=OffsetSource.BUNDLE,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            section=chunk.section,
            # BM25 *is* the lexical score, and there is no semantic half to report.
            # Reporting one as ``0.0`` would read as "the vector search found
            # nothing" rather than "no vector search happened".
            lexical_score=score,
            semantic_score=None,
            image_ids=chunk.image_ids,
        )

    async def health(self) -> str:
        return f"in-process BM25 over {len(self._chunk_ids)} chunks of {self._corpus.corpus_id}"


__all__ = [
    "BM25_B",
    "BM25_K1",
    "CANDIDATE_MULTIPLIER",
    "MIN_CANDIDATES",
    "LexicalCorpusRetriever",
]

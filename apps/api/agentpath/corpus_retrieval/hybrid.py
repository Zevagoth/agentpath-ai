"""The hybrid method: RAGFlow's term-plus-vector retrieval over one corpus dataset.

This is the Stage 2A architecture pointed at somebody else's corpus, which is the
whole question Stage 2B1 was set to answer. What it shares with the procedures path
is everything that matters - the same RAGFlow instance, the same ``bge-m3``
embeddings, the same blend of BM25 and vector similarity - and the differences are
two, both deliberate.

**The travel lexicon is not applied.** The procedures path expands a query with
domain synonyms (``إلغاء`` -> ``cancellation``, "no room" -> "walk", and so on)
because its corpus is hotel and refund policy. MRAMG-Manual is device manuals.
Expanding "how do I clean the filter" with refund vocabulary would add noise, and
worse, it would be *measured* as retrieval quality: a drop caused by injected
travel terms would be reported as the architecture failing on external data. The
question the query embeds is the question the user asked.

**The dataset is fixed at construction.** ``agentpath-mramg-manual`` and nothing
else. There is no parameter on :meth:`retrieve` that can add a dataset, so a mixed
result is not a thing this class can produce.

Offsets are recovered rather than returned. RAGFlow chunked the documents, so a
passage arrives as text; :mod:`agentpath.corpus_retrieval.offsets` locates it in the
verbatim document, and a passage that cannot be located is reported instead of being
given a plausible position.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agentpath.corpus import Corpus, heading_label
from agentpath.corpus_retrieval.grouping import group_into_documents
from agentpath.corpus_retrieval.offsets import locate_passage
from agentpath.corpus_retrieval.ports import (
    CorpusPassage,
    DocumentHit,
    OffsetSource,
    RetrievalMethod,
)
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable
from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError, RAGFlowUnavailable

logger = structlog.get_logger(__name__)

#: Chunks requested per document wanted. RAGFlow returns chunks and the caller wants
#: documents; on this corpus one document commonly supplies several neighbouring
#: chunks, so asking for `limit` chunks would routinely yield fewer than `limit`
#: documents after grouping.
CANDIDATE_MULTIPLIER = 6
MIN_PAGE_SIZE = 24


class HybridCorpusRetriever:
    """Ranks a corpus's documents through RAGFlow's hybrid retrieval."""

    def __init__(
        self,
        client: RAGFlowClient,
        corpus: Corpus,
        *,
        dataset_name: str,
        vector_similarity_weight: float = 0.6,
        similarity_threshold: float = 0.15,
        rerank_model: str | None = None,
        rerank_candidates: int = 32,
    ) -> None:
        self._client = client
        self._corpus = corpus
        self._dataset_name = dataset_name
        self._vector_similarity_weight = vector_similarity_weight
        self._similarity_threshold = similarity_threshold
        self._rerank_model = rerank_model
        self._rerank_candidates = rerank_candidates
        self._dataset_id: str | None = None

    @property
    def method(self) -> RetrievalMethod:
        return RetrievalMethod.HYBRID

    @property
    def corpus_id(self) -> str:
        return self._corpus.corpus_id

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    # -- dataset ------------------------------------------------------------ #

    async def dataset_id(self) -> str:
        """Resolve and cache this corpus's dataset id.

        A missing dataset is an error with an instruction attached, not an empty
        result: "nothing matched your question" and "the corpus was never indexed"
        must not look the same, because only one of them is fixed by asking
        differently.
        """
        if self._dataset_id:
            return self._dataset_id
        try:
            found = await self._client.find_dataset_id(self._dataset_name)
        except RAGFlowUnavailable as exc:
            raise RetrievalUnavailable(str(exc), engine="ragflow") from exc
        except RAGFlowError as exc:
            raise RetrievalError(str(exc), engine="ragflow") from exc
        if not found:
            raise RetrievalError(
                f"RAGFlow dataset {self._dataset_name!r} does not exist. "
                f"Run: make corpus-ingest CORPUS={self._corpus.corpus_id}",
                engine="ragflow",
            )
        self._dataset_id = found
        return found

    # -- retrieval ---------------------------------------------------------- #

    async def retrieve(self, question: str, *, limit: int) -> tuple[tuple[DocumentHit, ...], dict[str, Any]]:
        if not question.strip():
            return (), {"method": self.method.value, "reason": "empty query"}

        dataset_id = await self.dataset_id()
        page_size = max(limit * CANDIDATE_MULTIPLIER, MIN_PAGE_SIZE)

        started = time.perf_counter()
        try:
            data = await self._client.retrieve(
                question=question,
                # One dataset. The isolation guarantee, expressed as the only value
                # this list can hold.
                dataset_ids=[dataset_id],
                page_size=page_size,
                similarity_threshold=self._similarity_threshold,
                vector_similarity_weight=self._vector_similarity_weight,
                rerank_id=self._rerank_model,
                rerank_candidates_count=max(self._rerank_candidates, page_size),
                highlight=False,
                include_metadata=True,
            )
        except RAGFlowUnavailable as exc:
            raise RetrievalUnavailable(str(exc), engine="ragflow", detail=exc.path) from exc
        except RAGFlowError as exc:
            raise RetrievalError(str(exc), engine="ragflow", detail=exc.path) from exc
        engine_ms = (time.perf_counter() - started) * 1000

        chunks = data.get("chunks") or []
        passages: list[CorpusPassage] = []
        problems: list[str] = []
        offset_sources: dict[str, int] = {}
        foreign = 0

        for chunk in chunks:
            passage, problem = self._to_passage(chunk)
            if passage is None:
                foreign += 1
                if problem:
                    problems.append(problem)
                continue
            passages.append(passage)
            offset_sources[passage.offset_source.value] = offset_sources.get(passage.offset_source.value, 0) + 1
            if problem:
                problems.append(problem)

        titles = {document_id: record.title for document_id, record in self._corpus.documents.items()}
        documents = group_into_documents(tuple(passages), titles=titles, limit=limit)

        logger.debug(
            "corpus.hybrid_retrieved",
            corpus=self._corpus.corpus_id,
            chunks=len(chunks),
            documents=len(documents),
            engine_ms=round(engine_ms, 1),
            unresolved=offset_sources.get(OffsetSource.UNRESOLVED.value, 0),
        )

        return documents, {
            "method": self.method.value,
            "dataset_id": dataset_id,
            "dataset_name": self._dataset_name,
            "engine_ms": round(engine_ms, 2),
            "chunks_returned": len(chunks),
            "total_candidates": data.get("total", len(chunks)),
            "vector_similarity_weight": self._vector_similarity_weight,
            "similarity_threshold": self._similarity_threshold,
            "rerank_model": self._rerank_model,
            "offset_sources": offset_sources,
            # Should always be zero. It is counted rather than assumed, because a
            # non-zero value is the isolation guarantee failing and that must be
            # visible in the result rather than inferred from a missing document.
            "chunks_from_unknown_documents": foreign,
            "problems": problems,
        }

    def _to_passage(self, chunk: dict[str, Any]) -> tuple[CorpusPassage | None, str | None]:
        """Map one RAGFlow chunk onto a passage, recovering its offsets.

        Returns ``(None, problem)`` when the chunk names a document this corpus does
        not contain. Dropping it is the right call - a passage we cannot attribute
        cannot be cited - but it is also counted, because the only way it can happen
        is a dataset holding something the bundle does not.
        """
        metadata = chunk.get("document_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        document_id = str(metadata.get("document_id") or "")
        if not document_id:
            document_id = _document_id_from_name(str(chunk.get("document_keyword") or chunk.get("docnm_kwd") or ""))

        document = self._corpus.document(document_id)
        if document is None:
            return None, f"retrieved chunk names document {document_id!r}, which is not in corpus {self.corpus_id!r}"

        text = str(chunk.get("content") or chunk.get("content_with_weight") or "")
        located = locate_passage(document.text, text)

        problem = None
        if located.problem:
            problem = f"document {document_id}: {located.problem}"

        # Images are those whose placeholder falls inside the located span. With no
        # span there is no defensible subset, so the candidate list is empty rather
        # than the document's whole set - showing every image of a 54-image manual
        # because one passage could not be located is the failure mode Part 5 of the
        # brief rules out.
        image_ids: tuple[str, ...] = ()
        if located.found:
            assert located.char_start is not None and located.char_end is not None
            image_ids = tuple(
                record.image_id
                for record in self._corpus.images_for(document_id)
                if located.char_start <= record.char_offset < located.char_end
            )

        return (
            CorpusPassage(
                chunk_id=str(chunk.get("id") or chunk.get("chunk_id") or ""),
                document_id=document_id,
                corpus_id=self.corpus_id,
                text=text,
                score=_as_float(chunk.get("similarity")),
                offset_source=located.source,
                char_start=located.char_start,
                char_end=located.char_end,
                section=_section_from_text(text),
                lexical_score=_optional_float(chunk.get("term_similarity")),
                semantic_score=_optional_float(chunk.get("vector_similarity")),
                image_ids=image_ids,
            ),
            problem,
        )

    async def health(self) -> str:
        try:
            dataset_id = await self.dataset_id()
        except (RetrievalError, RetrievalUnavailable) as exc:
            return f"unavailable: {exc}"
        return f"RAGFlow dataset {self._dataset_name!r} ({dataset_id})"


def _document_id_from_name(document_name: str) -> str:
    """``mramg-manual-50020.md`` -> ``50020``.

    Fallback for a chunk that arrives without metadata. The filename is built by
    ``agentpath_ingestion.corpora.ingest._filename_for`` as
    ``{corpus_id}-{document_id}.md``, and the document id is the part after the last
    hyphen because a corpus id contains hyphens and a document id does not.
    """
    if not document_name:
        return ""
    stem = document_name.rsplit(".", 1)[0]
    return stem.rsplit("-", 1)[-1] if "-" in stem else stem


def _section_from_text(text: str) -> str | None:
    """The first markdown heading inside the passage, when it has one.

    Weaker than the bundle's ``section`` field, which is the heading *in force* at
    the span even when the span contains none, and that difference is why the
    lexical baseline's sections are better. Recomputing the governing heading here
    would mean re-scanning a 168,000-character document per passage per request; the
    citation carries offsets either way, so the heading is a label rather than the
    thing that makes it openable.

    Trimmed through :func:`heading_label` for the reason it exists: these documents are
    flattened, so a "line" beginning with ``#`` runs to the end of the section. Without
    the trim this returned 1,300-character labels - observed live on document 50024,
    whose citation position read as a page of installation steps.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            label = heading_label(stripped)
            if label:
                return label
    return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CANDIDATE_MULTIPLIER",
    "MIN_PAGE_SIZE",
    "HybridCorpusRetriever",
]

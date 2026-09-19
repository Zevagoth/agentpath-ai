"""One method = one document retriever + one image selector, and the registry of three.

This is where the three methods become comparable objects instead of three ideas.
Each is a :class:`CorpusSearchService` over the same corpus, so the evaluation asks
all three the same question through the same call and the only thing that differs is
what is inside.

The three, and what each is for:

``lexical``     BM25 over our chunks, images by document order. Reads nothing that
                the other two do not, and uses no model at all, so it is the floor
                every other number is measured against.
``hybrid``      RAGFlow term+vector, images by document order. The architecture as
                Stage 2A built it, pointed at an external corpus. Isolating this from
                ``multimodal`` is what makes the context-window ranking's
                contribution a measured quantity rather than a claim.
``multimodal``  RAGFlow term+vector, images by context-window ranking. The proposal.

Provenance is built here rather than in the API layer because the evaluation needs
exactly the same citations the UI shows - a "citation validity 100%" that was
measured over a different object than the one rendered would be worth nothing.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agentpath.config import Settings
from agentpath.corpus import Corpus, dataset_name_for
from agentpath.corpus_retrieval.hybrid import HybridCorpusRetriever
from agentpath.corpus_retrieval.images import ContextWindowImageSelector, DocumentOrderImageSelector
from agentpath.corpus_retrieval.lexical import LexicalCorpusRetriever
from agentpath.corpus_retrieval.ports import (
    CorpusRetrievalResult,
    DocumentHit,
    DocumentRetriever,
    ImageSelector,
    RetrievalMethod,
)
from agentpath.provenance import Provenance, SourceLocator, build_provenance
from agentpath.retrieval.ragflow_client import RAGFlowClient

logger = structlog.get_logger(__name__)

#: Documents returned by default. Three because the headline metric is Document
#: Recall@3 and an agent reading three candidates is plausible; @1 and @5 are computed
#: from a deeper run in the evaluation rather than by changing this.
DEFAULT_DOCUMENT_LIMIT = 3

#: Images returned by default. Five because Image Recall@5 is the target metric, and
#: because the questions expect between zero and six images - a limit below five would
#: cap the metric below its own target.
DEFAULT_IMAGE_LIMIT = 5


class CorpusSearchService:
    """Runs one retrieval method over one corpus, end to end.

    Holds a corpus and a method and exposes one call. There is no parameter here that
    names a corpus or a dataset: both were fixed when this object was built, which is
    what makes "results from two corpora mixed together" unreachable rather than
    merely checked for.
    """

    def __init__(
        self,
        *,
        corpus: Corpus,
        retriever: DocumentRetriever,
        images: ImageSelector,
        method: RetrievalMethod,
        dataset_id: str | None = None,
    ) -> None:
        self._corpus = corpus
        self._retriever = retriever
        self._images = images
        self._method = method
        self._dataset_id = dataset_id

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def corpus_id(self) -> str:
        return self._corpus.corpus_id

    @property
    def method(self) -> RetrievalMethod:
        return self._method

    @property
    def image_strategy(self) -> str:
        return self._images.strategy

    async def search(
        self,
        question: str,
        *,
        limit: int = DEFAULT_DOCUMENT_LIMIT,
        image_limit: int = DEFAULT_IMAGE_LIMIT,
    ) -> CorpusRetrievalResult:
        """Retrieve documents, then choose images from them.

        Timed in two parts because they are two different claims. "Retrieval p95 under
        two seconds" is about the engine; image selection is in-process arithmetic over
        data already in memory, and folding the two together would let one hide the
        other.
        """
        total_started = time.perf_counter()

        retrieval_started = time.perf_counter()
        documents, retrieval_diagnostics = await self._retriever.retrieve(question, limit=limit)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        selection_started = time.perf_counter()
        images, image_diagnostics = self._images.select(question, documents, limit=image_limit)
        selection_ms = (time.perf_counter() - selection_started) * 1000

        problems = list(retrieval_diagnostics.pop("problems", []) or [])
        problems.extend(self._image_problems(images))

        total_ms = (time.perf_counter() - total_started) * 1000

        logger.info(
            "corpus.search_completed",
            corpus=self.corpus_id,
            method=self._method.value,
            image_strategy=self._images.strategy,
            documents=[hit.document_id for hit in documents],
            images=len(images),
            retrieval_ms=round(retrieval_ms, 1),
            selection_ms=round(selection_ms, 1),
            problems=len(problems),
        )

        return CorpusRetrievalResult(
            corpus_id=self.corpus_id,
            method=self._method,
            question=question,
            documents=documents,
            images=images,
            retrieval_ms=retrieval_ms,
            image_selection_ms=selection_ms,
            total_ms=total_ms,
            diagnostics={
                "corpus_id": self.corpus_id,
                "method": self._method.value,
                "image_retrieval_mode": self._corpus.image_retrieval_mode,
                "retrieval": retrieval_diagnostics,
                "image_selection": image_diagnostics,
            },
            problems=tuple(problems),
        )

    def _image_problems(self, images: tuple[Any, ...]) -> list[str]:
        """Check that every selected image's bytes are still on disk.

        One ``stat`` per returned image, at most five. Worth the cost because the
        evaluation reports a broken-image rate, and a rate computed from a registry
        built at startup would only ever report zero - it would be measuring the
        registry, not the filesystem the browser will fetch from.
        """
        problems: list[str] = []
        for hit in images:
            record = self._corpus.image(hit.image_id)
            if record is None:
                problems.append(f"selected image {hit.image_id!r} is not in the corpus registry")
            elif not record.path.is_file():
                problems.append(f"image {hit.image_id!r} is registered but its file is missing")
        return problems

    def citations(self, result: CorpusRetrievalResult) -> tuple[Provenance, ...]:
        """One provenance record per retrieved document, in rank order."""
        return tuple(record for hit in result.documents if (record := self.citation_for(hit, result)) is not None)

    def citation_for(self, hit: DocumentHit, result: CorpusRetrievalResult) -> Provenance | None:
        """The full trail for one retrieved document, opening at its best passage.

        The locator is the best passage's span, because that is the part of the
        document the retrieval actually matched and therefore the place a reader
        should land. A passage whose offsets could not be recovered contributes no
        locator at all, so the document opens at the top and the page says so - which
        is the honest behaviour, not a fallback offset that looks precise.
        """
        document = self._corpus.document(hit.document_id)
        if document is None:
            return None

        passage = hit.best_passage
        locator = None
        if passage.has_offsets:
            locator = SourceLocator(
                section=passage.section,
                char_start=passage.char_start,
                char_end=passage.char_end,
            )

        images = tuple(
            record.provenance()
            for image_hit in result.images
            if image_hit.document_id == hit.document_id and (record := self._corpus.image(image_hit.image_id))
        )

        return build_provenance(
            corpus_id=self._corpus.corpus_id,
            dataset_id=self._dataset_id,
            document_id=document.document_id,
            document_title=document.title,
            chunk_id=passage.chunk_id or None,
            source_url=document.source_url,
            source_id=document.source_id,
            locator=locator,
            images=images,
            # Licence and attribution are deliberately not here. They are properties
            # of the corpus, not of one citation, so they are reported once per corpus
            # rather than repeated on every document - the same fact stamped forty
            # times is the same fact, and stamping it invites it to disagree with
            # itself.
        )

    async def health(self) -> str:
        return await self._retriever.health()


def build_methods(
    corpus: Corpus,
    *,
    settings: Settings,
    client: RAGFlowClient | None = None,
) -> dict[RetrievalMethod, CorpusSearchService]:
    """Build every method available for *corpus*, keyed by method.

    The lexical baseline is always available: it needs nothing but the bundle already
    in memory. The two RAGFlow-backed methods appear only when a client is supplied,
    so a deployment or a test with no RAGFlow gets a working baseline and an honestly
    absent hybrid rather than a method that raises on first use.

    Two objects are shared on purpose. The hybrid retriever serves both ``hybrid`` and
    ``multimodal``, because the two must differ *only* in image selection and one
    instance is a stronger guarantee of that than two configured identically. The
    document-order selector serves both ``lexical`` and ``hybrid`` for the same reason.

    Reranking is off here even when it is on for the procedures path. The reranker was
    measured to be killed by the kernel on this machine (exit 137), and a method that
    dies under load would make the comparison a comparison of what survived.
    """
    methods: dict[RetrievalMethod, CorpusSearchService] = {}

    document_order = DocumentOrderImageSelector(corpus)
    lexical = LexicalCorpusRetriever(corpus)
    methods[RetrievalMethod.LEXICAL] = CorpusSearchService(
        corpus=corpus,
        retriever=lexical,
        images=document_order,
        method=RetrievalMethod.LEXICAL,
    )

    if client is not None:
        hybrid = HybridCorpusRetriever(
            client,
            corpus,
            dataset_name=dataset_name_for(corpus.corpus_id),
            vector_similarity_weight=settings.vector_similarity_weight,
            similarity_threshold=settings.similarity_threshold,
            rerank_model=None,
            rerank_candidates=settings.rerank_candidates,
        )
        methods[RetrievalMethod.HYBRID] = CorpusSearchService(
            corpus=corpus,
            retriever=hybrid,
            images=document_order,
            method=RetrievalMethod.HYBRID,
        )
        methods[RetrievalMethod.MULTIMODAL] = CorpusSearchService(
            corpus=corpus,
            retriever=hybrid,
            images=ContextWindowImageSelector(corpus),
            method=RetrievalMethod.MULTIMODAL,
        )

    logger.info(
        "corpus.methods_built",
        corpus=corpus.corpus_id,
        methods=[method.value for method in methods],
        ragflow=client is not None,
    )
    return methods


def default_method(methods: dict[RetrievalMethod, CorpusSearchService]) -> RetrievalMethod:
    """The method used when a request does not name one.

    ``multimodal`` when it exists, because it is the proposal and the one the console
    should show; otherwise ``lexical``, which always exists. Never a silent
    substitution that the response does not report: every result carries its own
    ``method``.
    """
    for candidate in (RetrievalMethod.MULTIMODAL, RetrievalMethod.HYBRID, RetrievalMethod.LEXICAL):
        if candidate in methods:
            return candidate
    raise ValueError("no retrieval method is available for this corpus")


__all__ = [
    "DEFAULT_DOCUMENT_LIMIT",
    "DEFAULT_IMAGE_LIMIT",
    "CorpusSearchService",
    "build_methods",
    "default_method",
]

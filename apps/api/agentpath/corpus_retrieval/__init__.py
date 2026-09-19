"""Retrieval over an external corpus: three methods, one interface.

Kept in its own package rather than added to ``agentpath.retrieval`` because the two
answer different questions. ``agentpath.retrieval`` ranks *procedures* for a
customer-service case, with product-line filters, procedure-id lookups and version
metadata. This ranks *documents and images* of a third-party corpus, with character
offsets and image positions. Sharing one module would have meant a type carrying
five fields that mean nothing on each side.

Start at :mod:`agentpath.corpus_retrieval.ports` for the contract and
:mod:`agentpath.corpus_retrieval.service` for the three methods.
"""

from agentpath.corpus_retrieval.hybrid import HybridCorpusRetriever
from agentpath.corpus_retrieval.images import ContextWindowImageSelector, DocumentOrderImageSelector
from agentpath.corpus_retrieval.lexical import LexicalCorpusRetriever
from agentpath.corpus_retrieval.offsets import Located, locate_passage
from agentpath.corpus_retrieval.ports import (
    CorpusPassage,
    CorpusRetrievalResult,
    DocumentHit,
    DocumentRetriever,
    ImageHit,
    ImageSelector,
    OffsetSource,
    RetrievalMethod,
)
from agentpath.corpus_retrieval.service import (
    DEFAULT_DOCUMENT_LIMIT,
    DEFAULT_IMAGE_LIMIT,
    CorpusSearchService,
    build_methods,
    default_method,
)

__all__ = [
    "DEFAULT_DOCUMENT_LIMIT",
    "DEFAULT_IMAGE_LIMIT",
    "ContextWindowImageSelector",
    "CorpusPassage",
    "CorpusRetrievalResult",
    "CorpusSearchService",
    "DocumentHit",
    "DocumentOrderImageSelector",
    "DocumentRetriever",
    "HybridCorpusRetriever",
    "ImageHit",
    "ImageSelector",
    "LexicalCorpusRetriever",
    "Located",
    "OffsetSource",
    "RetrievalMethod",
    "build_methods",
    "default_method",
    "locate_passage",
]

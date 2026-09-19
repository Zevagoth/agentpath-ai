"""Corpus adapters: how an outside knowledge base enters AgentPath AI.

Stage 2B1 exists to answer whether this system can take a knowledge base it did
not author. The answer is only meaningful if the entry point is generic, so the
work is split in two:

* :mod:`agentpath_ingestion.corpora.base` declares what *any* corpus must be able
  to hand over - documents with their text unmodified, images with their positions,
  retrieval chunks, and the questions and ground truth an independent evaluation
  needs.
* :mod:`agentpath_ingestion.corpora.mramg` is one implementation of that, for
  MRAMG-Bench's Manual subset. It is the first, not the shape.

Nothing outside these modules knows the word "MRAMG".
"""

from agentpath_ingestion.corpora.base import (
    CorpusAdapter,
    CorpusBundle,
    CorpusChunk,
    CorpusDocument,
    CorpusImage,
    CorpusQuestion,
    FetchedArtefact,
    FetchReport,
)

__all__ = [
    "CorpusAdapter",
    "CorpusBundle",
    "CorpusChunk",
    "CorpusDocument",
    "CorpusImage",
    "CorpusQuestion",
    "FetchReport",
    "FetchedArtefact",
]

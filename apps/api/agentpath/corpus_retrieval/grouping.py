"""Collapse ranked passages into ranked documents.

Engines rank spans; the benchmark's ground truth names documents. One document can
match on four sections and four documents can each match on one, so the two have to
be reconciled somewhere, and doing it in one place means the lexical and hybrid
methods are compared on identical grouping rules. A difference in their numbers is
then a difference in retrieval, which is the only thing the comparison is allowed to
be about.

A document's score is its best passage's score, not a blend. A blend would stop
being comparable to the engine's own similarity while looking more sophisticated,
and it would reward long documents for containing many mediocre passages - on a
corpus whose largest document is 168,000 characters and 97 chunks, that is not a
hypothetical.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentpath.corpus_retrieval.ports import CorpusPassage, DocumentHit

#: Smaller than any meaningful score difference in either engine, so it can only
#: ever decide an exact tie.
TIE_EPSILON = 1e-6


def group_into_documents(
    passages: Sequence[CorpusPassage],
    *,
    titles: dict[str, str],
    limit: int,
) -> tuple[DocumentHit, ...]:
    """Group *passages* by document and return the best *limit*, ranked.

    Ties break first on how many passages matched - a document that matched four
    sections is more likely to be the one being asked about than a document that
    matched one at the same score - and then on document id, so a run is
    reproducible. Reproducibility is not a nicety here: the evaluation numbers in
    the report have to be the same numbers on a re-run.
    """
    grouped: dict[str, list[CorpusPassage]] = {}
    for passage in passages:
        grouped.setdefault(passage.document_id, []).append(passage)

    scored: list[tuple[float, str, tuple[CorpusPassage, ...]]] = []
    for document_id, group in grouped.items():
        ordered = tuple(sorted(group, key=lambda item: (-item.score, item.chunk_id)))
        bump = min(len(ordered) - 1, 5) * TIE_EPSILON
        scored.append((ordered[0].score + bump, document_id, ordered))

    scored.sort(key=lambda row: (-row[0], row[1]))

    return tuple(
        DocumentHit(
            document_id=document_id,
            title=titles.get(document_id) or document_id,
            score=score,
            passages=passages_of,
            rank=rank,
        )
        for rank, (score, document_id, passages_of) in enumerate(scored[:limit], start=1)
    )


__all__ = ["TIE_EPSILON", "group_into_documents"]

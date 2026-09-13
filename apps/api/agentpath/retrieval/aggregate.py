"""Turn engine passages into procedure-level hits.

Agents think in procedures, engines think in chunks. One procedure can match on
three sections; three procedures can each match on one. This module collapses
chunks into procedures without inventing information.

Scoring choice worth stating plainly: a procedure's score is the score of its
best matching passage, not a blend across passages. A blended or bonus-adjusted
number would look more sophisticated while meaning less: it would no longer be
comparable to the engine's own similarity, and the confidence shown to the agent
would drift away from anything measurable. The number of matching passages is
used only to break ties.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentpath.domain import Passage, ProcedureHit, rank_hits


def aggregate_passages(passages: Sequence[Passage], *, limit: int = 3) -> tuple[ProcedureHit, ...]:
    """Group *passages* by procedure and return the top *limit* procedures.

    Passages inside a hit are ordered best-first, so ``best_passage`` is the
    snippet worth showing and the rest are supporting context.
    """
    grouped: dict[str, list[Passage]] = {}
    for passage in passages:
        grouped.setdefault(passage.procedure_id, []).append(passage)

    hits: list[ProcedureHit] = []
    for procedure_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: -item.score)
        best = ordered[0]
        hits.append(
            ProcedureHit(
                procedure_id=procedure_id,
                title=_pick_title(ordered),
                score=best.score,
                passages=tuple(ordered),
                product_line=_first_not_none(ordered, "product_line"),
                version=_first_not_none(ordered, "version"),
                effective_date=_first_not_none(ordered, "effective_date"),
                scenario_tags=_merge_scenario_tags(ordered),
            )
        )

    ranked = rank_hits(_break_ties_by_passage_count(hits))
    return ranked[:limit]


def _break_ties_by_passage_count(hits: Sequence[ProcedureHit]) -> list[ProcedureHit]:
    """Nudge a procedure ahead of an exact-score peer when it matched more sections.

    The nudge is deliberately smaller than any meaningful score difference, so it
    only ever decides genuine ties.
    """
    if not hits:
        return []
    epsilon = 1e-6
    adjusted: list[ProcedureHit] = []
    for hit in hits:
        bump = min(hit.matched_passage_count - 1, 5) * epsilon
        adjusted.append(
            ProcedureHit(
                procedure_id=hit.procedure_id,
                title=hit.title,
                score=hit.score + bump,
                passages=hit.passages,
                product_line=hit.product_line,
                version=hit.version,
                effective_date=hit.effective_date,
                scenario_tags=hit.scenario_tags,
            )
        )
    return adjusted


def _pick_title(passages: Sequence[Passage]) -> str:
    for passage in passages:
        if passage.title:
            return passage.title
    return passages[0].procedure_id


def _first_not_none(passages: Sequence[Passage], attribute: str) -> str | None:
    for passage in passages:
        value = getattr(passage, attribute)
        if value:
            return str(value)
    return None


def _merge_scenario_tags(passages: Sequence[Passage]) -> tuple[str, ...]:
    tags: list[str] = []
    for passage in passages:
        for tag in passage.scenario_tags:
            if tag not in tags:
                tags.append(tag)
    return tuple(tags)

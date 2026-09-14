"""Evidence grading and the abstention rule.

There is exactly one place in this system that decides "we do not have an answer",
and it is this module. Keeping it in one function means the API, the CLI and the
benchmark all abstain identically, and the benchmark can measure the rule instead
of measuring three approximations of it.

The rule uses two signals and no model:

* absolute score of the best procedure - is anything actually similar?
* margin over the runner-up - is one procedure clearly the right one, or is the
  engine undecided between several?

Thresholds live in configuration and were tuned against the evaluation set; see
EVALUATION.md. They are not universal truths, and the wording shown to agents is
chosen so that "strong" never reads as "certain".
"""

from __future__ import annotations

from collections.abc import Sequence

from agentpath.domain import EvidenceLevel, ProcedureHit

#: Shown verbatim when the system abstains. Fixed by the product specification.
ABSTENTION_MESSAGE = "No confirmed procedure was found in the available knowledge base."


def grade_evidence(
    hits: Sequence[ProcedureHit],
    *,
    strong_score: float,
    strong_margin: float,
    partial_score: float,
) -> tuple[EvidenceLevel, str]:
    """Classify how well *hits* support answering, with a human-readable reason."""
    if not hits:
        return (
            EvidenceLevel.INSUFFICIENT,
            "No passage in the knowledge base scored above the similarity threshold.",
        )

    top = hits[0]
    runner_up = hits[1] if len(hits) > 1 else None
    margin = top.score - runner_up.score if runner_up else None

    if top.score >= strong_score and (margin is None or margin >= strong_margin):
        detail = f"{top.procedure_id} scored {top.score:.2f}" + (
            f", ahead of the next procedure by {margin:.2f}." if margin is not None else " and was the only match."
        )
        return EvidenceLevel.STRONG, detail

    if top.score >= partial_score:
        if margin is not None and margin < strong_margin:
            detail = (
                f"{top.procedure_id} scored {top.score:.2f}, only {margin:.2f} ahead of "
                f"{hits[1].procedure_id}. Verify which one applies before acting."
            )
        else:
            detail = (
                f"{top.procedure_id} scored {top.score:.2f}, below the level at which a single "
                "procedure can be recommended without checking the source."
            )
        return EvidenceLevel.PARTIAL, detail

    return (
        EvidenceLevel.INSUFFICIENT,
        f"The best match ({top.procedure_id}) scored {top.score:.2f}, under the minimum of {partial_score:.2f}.",
    )


def detect_scenario_overlap(
    hits: Sequence[ProcedureHit],
    *,
    minimum_score: float,
) -> tuple[str, ...]:
    """Flag procedures that appear to cover the same scenario.

    Two procedures matching the same scenario tag with similar scores is the
    retrieval-stage shadow of a policy conflict. We surface it and prefer
    neither; version metadata is shown so a human can judge. Resolving conflicts
    automatically is out of scope until the answer layer exists, and guessing
    here would be worse than saying nothing.
    """
    notices: list[str] = []
    candidates = [hit for hit in hits if hit.score >= minimum_score]

    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            shared = sorted(set(left.scenario_tags) & set(right.scenario_tags))
            if not shared:
                continue
            notices.append(
                f"{left.procedure_id} (v{left.version or '?'}, effective {left.effective_date or '?'}) and "
                f"{right.procedure_id} (v{right.version or '?'}, effective {right.effective_date or '?'}) "
                f"both cover: {', '.join(shared)}. Confirm which one applies to this case."
            )
    return tuple(notices)

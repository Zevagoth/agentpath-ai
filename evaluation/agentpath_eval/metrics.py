"""Retrieval metrics.

Definitions are spelled out because benchmark numbers are only comparable when
the reader knows exactly what was counted:

* **Recall@k** - share of *supported* queries whose expected procedure appears in
  the top k. A query may list several acceptable procedures (a vague case can be
  legitimately served by either), and matching any one of them counts as a hit.
* **MRR** - mean of ``1 / rank`` of the first acceptable procedure, over supported
  queries. A query with no acceptable procedure in the results contributes 0.
* **Abstention accuracy** - share of *unsupported* queries where the system
  refused to answer. Measured only on unsupported queries.
* **False abstention rate** - share of supported queries where the system refused
  even though an answer existed. This is the cost of a cautious threshold, and
  reporting recall without it would flatter the system.
* **Latency percentiles** - computed over all recorded runs after discarding a
  warm-up run per query, so first-call model loading and connection setup do not
  contaminate the numbers.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    """``1 / rank`` of the first acceptable item, or 0.0 when absent."""
    acceptable = set(expected)
    for index, item in enumerate(ranked, start=1):
        if item in acceptable:
            return 1.0 / index
    return 0.0


def hit_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> bool:
    acceptable = set(expected)
    return any(item in acceptable for item in ranked[:k])


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with a few dozen samples an interpolated p95
    invents a number that no request actually experienced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # Ceiling, not rounding: the p-th percentile is the smallest observation with
    # at least p of the data at or below it. Rounding would return the 2nd of five
    # samples for p50, which is not the median.
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass
class LatencySummary:
    """Latency distribution for one mode."""

    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0

    @property
    def p50_ms(self) -> float:
        return percentile(self.samples, 0.50)

    @property
    def p95_ms(self) -> float:
        return percentile(self.samples, 0.95)

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.count,
            "mean_ms": round(self.mean_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "max_ms": round(self.max_ms, 1),
        }


@dataclass
class QueryOutcome:
    """What one mode produced for one query."""

    query_id: str
    query: str
    type: str
    language: str
    expected: tuple[str, ...]
    supported: bool
    ranked: tuple[str, ...]
    abstained: bool
    evidence: str
    latency_ms: float
    top_score: float = 0.0
    notes: str = ""

    @property
    def hit_at_1(self) -> bool:
        return hit_at_k(self.ranked, self.expected, 1)

    @property
    def hit_at_3(self) -> bool:
        return hit_at_k(self.ranked, self.expected, 3)

    @property
    def reciprocal_rank(self) -> float:
        return reciprocal_rank(self.ranked, self.expected)

    @property
    def correct_abstention(self) -> bool:
        """Abstained on a query that genuinely has no answer."""
        return not self.supported and self.abstained

    @property
    def false_abstention(self) -> bool:
        """Abstained even though an answer existed."""
        return self.supported and self.abstained

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "type": self.type,
            "language": self.language,
            "supported": self.supported,
            "expected": list(self.expected),
            "ranked": list(self.ranked),
            "abstained": self.abstained,
            "evidence": self.evidence,
            "latency_ms": round(self.latency_ms, 1),
            "top_score": round(self.top_score, 4),
            "hit_at_1": self.hit_at_1 if self.supported else None,
            "hit_at_3": self.hit_at_3 if self.supported else None,
            "reciprocal_rank": round(self.reciprocal_rank, 4) if self.supported else None,
            "notes": self.notes,
        }


@dataclass
class ModeReport:
    """Aggregated results for one retrieval mode."""

    mode: str
    description: str
    outcomes: list[QueryOutcome] = field(default_factory=list)
    latency: LatencySummary = field(default_factory=LatencySummary)
    error: str | None = None

    # -- populations -------------------------------------------------------- #

    @property
    def supported(self) -> list[QueryOutcome]:
        return [outcome for outcome in self.outcomes if outcome.supported]

    @property
    def unsupported(self) -> list[QueryOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.supported]

    # -- retrieval quality, supported queries only -------------------------- #

    @property
    def recall_at_1(self) -> float:
        population = self.supported
        return sum(1 for o in population if o.hit_at_1) / len(population) if population else 0.0

    @property
    def recall_at_3(self) -> float:
        population = self.supported
        return sum(1 for o in population if o.hit_at_3) / len(population) if population else 0.0

    @property
    def mrr(self) -> float:
        population = self.supported
        return statistics.fmean([o.reciprocal_rank for o in population]) if population else 0.0

    # -- abstention behaviour ---------------------------------------------- #

    @property
    def abstention_accuracy(self) -> float:
        population = self.unsupported
        return sum(1 for o in population if o.correct_abstention) / len(population) if population else 0.0

    @property
    def false_abstention_rate(self) -> float:
        population = self.supported
        return sum(1 for o in population if o.false_abstention) / len(population) if population else 0.0

    # -- breakdowns --------------------------------------------------------- #

    def by_type(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[QueryOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.type, []).append(outcome)

        summary: dict[str, dict[str, float | int]] = {}
        for query_type, group in sorted(grouped.items()):
            supported = [o for o in group if o.supported]
            if supported:
                summary[query_type] = {
                    "queries": len(group),
                    "recall_at_1": round(sum(1 for o in supported if o.hit_at_1) / len(supported), 3),
                    "recall_at_3": round(sum(1 for o in supported if o.hit_at_3) / len(supported), 3),
                    "mrr": round(statistics.fmean([o.reciprocal_rank for o in supported]), 3),
                }
            else:
                summary[query_type] = {
                    "queries": len(group),
                    "abstention_accuracy": round(sum(1 for o in group if o.correct_abstention) / len(group), 3),
                }
        return summary

    def failures(self) -> list[QueryOutcome]:
        """Supported queries that missed the top 3, worst first."""
        return sorted(
            (o for o in self.supported if not o.hit_at_3),
            key=lambda o: o.query_id,
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "description": self.description,
            "error": self.error,
            "counts": {
                "queries": len(self.outcomes),
                "supported": len(self.supported),
                "unsupported": len(self.unsupported),
            },
            "retrieval_quality_supported_only": {
                "recall_at_1": round(self.recall_at_1, 4),
                "recall_at_3": round(self.recall_at_3, 4),
                "mrr": round(self.mrr, 4),
                "false_abstention_rate": round(self.false_abstention_rate, 4),
            },
            "abstention_unsupported_only": {
                "abstention_accuracy": round(self.abstention_accuracy, 4),
            },
            "latency": self.latency.to_dict(),
            "by_type": self.by_type(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

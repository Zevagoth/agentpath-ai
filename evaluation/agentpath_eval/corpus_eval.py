"""Retrieval evaluation over an external corpus: three methods, two splits.

Every definition is spelled out, because a benchmark number is only comparable when
the reader knows exactly what was counted. Where a metric could reasonably be defined
two ways, both are reported rather than one being chosen quietly.

Document metrics, over every question in the split
-------------------------------------------------
**Document Recall@k** - share of questions whose provenance document appears in the
top *k* returned documents. Every MRAMG-Manual question names exactly one provenance
document (verified over all 390), so this is unambiguous here; a question naming
several would count as a hit if any appeared.

**Document MRR** - mean of ``1 / rank`` of the first correct document. A question whose
document is absent contributes 0.

Image metrics, over questions that expect at least one image
-----------------------------------------------------------
Two different things are both called "recall@k", so both are reported.

**Image Recall@k** - *coverage*: mean over questions of
``|truth ∩ top k| / |truth|``. A question expecting three images and receiving one of
them scores 0.33. This is the metric the target refers to.

**Image Hit@k** - mean over questions of "at least one correct image in the top *k*".
Always at least Recall@k, and much more forgiving on multi-image questions. Reported so
a high hit rate cannot be mistaken for high coverage.

**Image precision** - mean over questions of ``|truth ∩ returned| / |returned|``,
measured on the set actually returned rather than at a cutoff, because that is the set a
user is shown. **Image F1** is the harmonic mean of that precision and the
whole-set recall.

Questions expecting no image at all
-----------------------------------
40 of MRAMG-Manual's 390 expect none. They are excluded from every image metric above,
because recall over an empty truth set is undefined and precision over it is always
zero, and scored instead as **image abstention accuracy**: the share of them for which
no image was returned. A method that floods every answer with figures is visibly wrong
here and invisible everywhere else.

Provenance metrics, over every returned document and image
----------------------------------------------------------
**Citation validity** - share of returned documents carrying a provenance record that
names its corpus and document *and* offers an openable link. The target is 100%, so any
shortfall is listed individually rather than only as a rate.

**Broken source rate** - share of citations whose link cannot be resolved: a local
source id absent from the corpus's registry, or no link at all. Upstream URLs are
checked for shape only; this evaluation makes no network requests, and a URL that
404s upstream is not something this system can be held to.

**Broken image rate** - share of returned images whose bytes are not on disk. Checked
with a filesystem call per image rather than against the startup registry, because a
rate computed from the registry would only ever be zero.

**Offset recovery** - share of passages by how their character offsets were
established. Not a pass/fail metric but the one that says how precise the citations
actually are: ``bundle`` and ``exact_match`` are exact, ``normalised_match`` is right to
within collapsed whitespace, and ``unresolved`` means the document opens at the top.

Latency
-------
One measured run per question, not several. Repeating the same question measures the
engine's caches rather than the system's behaviour across a query set, and 195 distinct
questions is a better distribution for a p95 than 65 questions repeated three times.
A single discarded warm-up query per method covers connection setup and dataset-id
resolution.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentpath.corpus import Corpus
from agentpath.corpus_retrieval.ports import OffsetSource
from agentpath.corpus_retrieval.service import CorpusSearchService
from agentpath.provenance import SourceKind
from agentpath_eval.metrics import LatencySummary, hit_at_k, percentile, reciprocal_rank

#: Depths reported for both document and image metrics. Five because the image target
#: is stated at @5, three because the document target is stated at @3, one because a
#: single-answer system is what an agent actually reads first.
DEPTHS = (1, 3, 5)

#: Documents and images requested per question. Both are the largest depth reported, so
#: no metric is capped by the request rather than by the ranking.
DOCUMENT_LIMIT = 5
IMAGE_LIMIT = 5

#: Targets from the brief, checked and reported rather than assumed.
TARGET_DOCUMENT_RECALL_AT_3 = 0.90
TARGET_IMAGE_RECALL_AT_5 = 0.80
TARGET_CITATION_VALIDITY = 1.0
TARGET_RETRIEVAL_P95_MS = 2_000.0


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """What one method produced for one question, with nothing aggregated yet.

    Kept per question so a failure can be looked at rather than inferred from a rate.
    """

    question_id: str
    question: str
    stratum: str
    expected_documents: tuple[str, ...]
    expected_images: tuple[str, ...]
    ranked_documents: tuple[str, ...]
    ranked_images: tuple[str, ...]
    retrieval_ms: float
    image_selection_ms: float
    #: Everything this process spent, engine time included. Recorded so the claim that
    #: the latency figures are the engine's can be checked rather than asserted: the gap
    #: is scoring and object construction, and if it ever stops being small the latency
    #: numbers need restating.
    wall_ms: float = 0.0
    #: Offset-source counts across every passage of every returned document.
    offset_sources: dict[str, int] = field(default_factory=dict)
    #: One entry per returned document: whether its citation is usable, and why not.
    citation_problems: tuple[str, ...] = ()
    #: Returned documents whose citation was complete and openable.
    valid_citations: int = 0
    returned_documents: int = 0
    #: Returned images whose bytes were not on disk.
    broken_images: int = 0
    problems: tuple[str, ...] = ()

    # -- documents ---------------------------------------------------------- #

    def document_hit_at(self, k: int) -> bool:
        return hit_at_k(self.ranked_documents, self.expected_documents, k)

    @property
    def document_reciprocal_rank(self) -> float:
        return reciprocal_rank(self.ranked_documents, self.expected_documents)

    # -- images ------------------------------------------------------------- #

    @property
    def expects_images(self) -> bool:
        return bool(self.expected_images)

    def image_coverage_at(self, k: int) -> float:
        """``|truth ∩ top k| / |truth|``. Undefined without truth, so 0.0 by contract."""
        if not self.expected_images:
            return 0.0
        found = set(self.ranked_images[:k]) & set(self.expected_images)
        return len(found) / len(self.expected_images)

    def image_hit_at(self, k: int) -> bool:
        return hit_at_k(self.ranked_images, self.expected_images, k)

    @property
    def image_precision(self) -> float:
        if not self.ranked_images:
            # Nothing returned: precision is undefined, and 1.0 would reward silence.
            # Excluded from the mean by the aggregator rather than scored here.
            return 0.0
        found = set(self.ranked_images) & set(self.expected_images)
        return len(found) / len(self.ranked_images)

    @property
    def image_recall(self) -> float:
        """Coverage over the whole returned set, the partner of :attr:`image_precision`."""
        if not self.expected_images:
            return 0.0
        found = set(self.ranked_images) & set(self.expected_images)
        return len(found) / len(self.expected_images)

    @property
    def image_f1(self) -> float:
        precision, recall = self.image_precision, self.image_recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def abstained_from_images(self) -> bool:
        return not self.ranked_images

    @property
    def abstained_from_documents(self) -> bool:
        return not self.ranked_documents

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "stratum": self.stratum,
            "expected_documents": list(self.expected_documents),
            "expected_images": list(self.expected_images),
            "ranked_documents": list(self.ranked_documents),
            "ranked_images": list(self.ranked_images),
            "document_hit_at_1": self.document_hit_at(1),
            "document_hit_at_3": self.document_hit_at(3),
            "document_hit_at_5": self.document_hit_at(5),
            "document_reciprocal_rank": round(self.document_reciprocal_rank, 4),
            "image_recall_at_5": round(self.image_coverage_at(5), 4) if self.expects_images else None,
            "image_precision": round(self.image_precision, 4) if self.expects_images else None,
            "image_f1": round(self.image_f1, 4) if self.expects_images else None,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "image_selection_ms": round(self.image_selection_ms, 2),
            "wall_ms": round(self.wall_ms, 1),
            "offset_sources": dict(self.offset_sources),
            "returned_documents": self.returned_documents,
            "valid_citations": self.valid_citations,
            "citation_problems": list(self.citation_problems),
            "broken_images": self.broken_images,
            "problems": list(self.problems),
        }


@dataclass
class MethodReport:
    """One method on one split."""

    corpus_id: str
    method: str
    image_strategy: str
    split: str
    outcomes: list[QuestionOutcome] = field(default_factory=list)
    latency: LatencySummary = field(default_factory=LatencySummary)
    error: str | None = None

    # -- populations -------------------------------------------------------- #

    @property
    def with_images(self) -> list[QuestionOutcome]:
        return [outcome for outcome in self.outcomes if outcome.expects_images]

    @property
    def without_images(self) -> list[QuestionOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.expects_images]

    # -- documents ---------------------------------------------------------- #

    def document_recall_at(self, k: int) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.document_hit_at(k)) / len(self.outcomes)

    @property
    def document_mrr(self) -> float:
        if not self.outcomes:
            return 0.0
        return statistics.fmean(outcome.document_reciprocal_rank for outcome in self.outcomes)

    # -- images ------------------------------------------------------------- #

    def image_recall_at(self, k: int) -> float:
        population = self.with_images
        if not population:
            return 0.0
        return statistics.fmean(outcome.image_coverage_at(k) for outcome in population)

    def image_hit_at(self, k: int) -> float:
        population = self.with_images
        if not population:
            return 0.0
        return sum(1 for outcome in population if outcome.image_hit_at(k)) / len(population)

    @property
    def image_precision(self) -> float:
        """Averaged over questions that expect images *and* received some.

        A question that received nothing has no precision to report, and scoring it as
        zero would make precision fall whenever the ranking correctly stayed silent.
        """
        population = [outcome for outcome in self.with_images if outcome.ranked_images]
        if not population:
            return 0.0
        return statistics.fmean(outcome.image_precision for outcome in population)

    @property
    def image_recall(self) -> float:
        population = self.with_images
        if not population:
            return 0.0
        return statistics.fmean(outcome.image_recall for outcome in population)

    @property
    def image_f1(self) -> float:
        precision, recall = self.image_precision, self.image_recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def image_abstention_accuracy(self) -> float:
        """Share of no-image questions for which no image was returned."""
        population = self.without_images
        if not population:
            return 0.0
        return sum(1 for outcome in population if outcome.abstained_from_images) / len(population)

    @property
    def document_abstention_rate(self) -> float:
        """Share of questions where no document came back at all.

        Every question in this corpus is answerable from it, so this is a failure
        rate rather than a virtue - unlike the procedures corpus, where abstaining on an
        unanswerable case is the correct behaviour.
        """
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.abstained_from_documents) / len(self.outcomes)

    # -- provenance --------------------------------------------------------- #

    @property
    def returned_documents(self) -> int:
        return sum(outcome.returned_documents for outcome in self.outcomes)

    @property
    def valid_citations(self) -> int:
        return sum(outcome.valid_citations for outcome in self.outcomes)

    @property
    def citation_validity(self) -> float:
        total = self.returned_documents
        return self.valid_citations / total if total else 0.0

    @property
    def broken_source_rate(self) -> float:
        total = self.returned_documents
        if not total:
            return 0.0
        return (total - self.valid_citations) / total

    @property
    def returned_images(self) -> int:
        return sum(len(outcome.ranked_images) for outcome in self.outcomes)

    @property
    def broken_images(self) -> int:
        return sum(outcome.broken_images for outcome in self.outcomes)

    @property
    def broken_image_rate(self) -> float:
        total = self.returned_images
        return self.broken_images / total if total else 0.0

    @property
    def harness_overhead_ms(self) -> float:
        """Median gap between wall time and the engine time the percentiles use.

        Reported so "these are the engine's numbers" is checkable. A large value would
        mean the scoring in this module had become part of what is being measured.
        """
        gaps = [outcome.wall_ms - outcome.retrieval_ms for outcome in self.outcomes if outcome.wall_ms]
        return statistics.median(gaps) if gaps else 0.0

    @property
    def offset_sources(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for outcome in self.outcomes:
            for key, count in outcome.offset_sources.items():
                totals[key] = totals.get(key, 0) + count
        return dict(sorted(totals.items()))

    @property
    def unresolved_offset_rate(self) -> float:
        totals = self.offset_sources
        total = sum(totals.values())
        return totals.get(OffsetSource.UNRESOLVED.value, 0) / total if total else 0.0

    # -- targets ------------------------------------------------------------ #

    def target_checks(self) -> dict[str, dict[str, Any]]:
        """Every stated target, with the measured value beside it.

        Reported as data rather than printed as pass/fail text, so the final report can
        show a failure as plainly as a pass.
        """
        return {
            "document_recall_at_3": _check(self.document_recall_at(3), TARGET_DOCUMENT_RECALL_AT_3, "at least"),
            "image_recall_at_5": _check(self.image_recall_at(5), TARGET_IMAGE_RECALL_AT_5, "at least"),
            "citation_validity": _check(self.citation_validity, TARGET_CITATION_VALIDITY, "at least"),
            "broken_source_rate": _check(self.broken_source_rate, 0.0, "at most"),
            "broken_image_rate": _check(self.broken_image_rate, 0.0, "at most"),
            "retrieval_p95_ms": _check(self.latency.p95_ms, TARGET_RETRIEVAL_P95_MS, "at most"),
        }

    def failures(self) -> list[QuestionOutcome]:
        """Questions whose document was not in the top 3, worst first."""
        return sorted(
            (outcome for outcome in self.outcomes if not outcome.document_hit_at(3)),
            key=lambda outcome: (outcome.document_reciprocal_rank, outcome.question_id),
        )

    def by_stratum(self) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[QuestionOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.stratum, []).append(outcome)

        summary: dict[str, dict[str, Any]] = {}
        for stratum, group in sorted(grouped.items()):
            with_images = [outcome for outcome in group if outcome.expects_images]
            summary[stratum] = {
                "questions": len(group),
                "document_recall_at_3": round(
                    sum(1 for outcome in group if outcome.document_hit_at(3)) / len(group), 4
                ),
                "image_recall_at_5": (
                    round(statistics.fmean(outcome.image_coverage_at(5) for outcome in with_images), 4)
                    if with_images
                    else None
                ),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "method": self.method,
            "image_strategy": self.image_strategy,
            "split": self.split,
            "error": self.error,
            "counts": {
                "questions": len(self.outcomes),
                "expecting_images": len(self.with_images),
                "expecting_no_image": len(self.without_images),
                "returned_documents": self.returned_documents,
                "returned_images": self.returned_images,
            },
            "documents": {
                **{f"recall_at_{k}": round(self.document_recall_at(k), 4) for k in DEPTHS},
                "mrr": round(self.document_mrr, 4),
                "abstention_rate": round(self.document_abstention_rate, 4),
            },
            "images": {
                **{f"recall_at_{k}": round(self.image_recall_at(k), 4) for k in DEPTHS},
                **{f"hit_at_{k}": round(self.image_hit_at(k), 4) for k in DEPTHS},
                "precision": round(self.image_precision, 4),
                "recall": round(self.image_recall, 4),
                "f1": round(self.image_f1, 4),
                "abstention_accuracy_on_no_image_questions": round(self.image_abstention_accuracy, 4),
            },
            "provenance": {
                "citation_validity": round(self.citation_validity, 4),
                "broken_source_rate": round(self.broken_source_rate, 4),
                "broken_image_rate": round(self.broken_image_rate, 4),
                "broken_images": self.broken_images,
                "offset_sources": self.offset_sources,
                "unresolved_offset_rate": round(self.unresolved_offset_rate, 4),
            },
            "latency_retrieval_only": {
                **self.latency.to_dict(),
                "median_harness_overhead_ms": round(self.harness_overhead_ms, 2),
            },
            "targets": self.target_checks(),
            "by_stratum": self.by_stratum(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def _check(measured: float, target: float, direction: str) -> dict[str, Any]:
    met = measured >= target if direction == "at least" else measured <= target
    return {
        "measured": round(measured, 4),
        "target": target,
        "direction": direction,
        "met": bool(met),
    }


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #


async def evaluate_method(
    service: CorpusSearchService,
    questions: Sequence[Any],
    *,
    split: str,
    document_limit: int = DOCUMENT_LIMIT,
    image_limit: int = IMAGE_LIMIT,
    warm_up: bool = True,
) -> MethodReport:
    """Run one method over one split and score every question.

    One measured run per question. A single warm-up query is issued first and thrown
    away, which covers connection setup and dataset-id resolution without letting
    engine caches flatter the percentiles.
    """
    report = MethodReport(
        corpus_id=service.corpus_id,
        method=service.method.value,
        image_strategy=service.image_strategy,
        split=split,
    )

    if warm_up and questions:
        await service.search(questions[0].question, limit=document_limit, image_limit=image_limit)

    for question in questions:
        started = time.perf_counter()
        result = await service.search(question.question, limit=document_limit, image_limit=image_limit)
        wall_ms = (time.perf_counter() - started) * 1000

        offset_sources: dict[str, int] = {}
        citation_problems: list[str] = []
        valid = 0
        for hit in result.documents:
            for passage in hit.passages:
                key = passage.offset_source.value
                offset_sources[key] = offset_sources.get(key, 0) + 1
            problem = _citation_problem(service, hit, result)
            if problem is None:
                valid += 1
            else:
                citation_problems.append(problem)

        broken_images = sum(1 for image_id in result.image_ids if not _image_on_disk(service.corpus, image_id))

        report.outcomes.append(
            QuestionOutcome(
                question_id=question.question_id,
                question=question.question,
                stratum=f"images={len(question.ground_truth_image_ids)}",
                expected_documents=tuple(question.provenance_document_ids),
                expected_images=tuple(question.ground_truth_image_ids),
                ranked_documents=result.document_ids,
                ranked_images=result.image_ids,
                # The engine's own timing, not the wall clock, so image selection and
                # scoring are excluded from the latency claim.
                retrieval_ms=result.retrieval_ms,
                image_selection_ms=result.image_selection_ms,
                wall_ms=wall_ms,
                offset_sources=offset_sources,
                citation_problems=tuple(citation_problems),
                valid_citations=valid,
                returned_documents=len(result.documents),
                broken_images=broken_images,
                problems=result.problems,
            )
        )
        # The engine's time is what the percentiles are built from. The wall time is
        # kept on the outcome so the difference is auditable.
        report.latency.samples.append(result.retrieval_ms)

    return report


def _citation_problem(service: CorpusSearchService, hit: Any, result: Any) -> str | None:
    """Why this document's citation is not usable, or ``None`` when it is.

    Three ways it can fail, all of which would otherwise show up as a silently
    unopenable button in the console.
    """
    trail = service.citation_for(hit, result)
    if trail is None:
        return f"{hit.document_id}: no provenance record could be built"
    if not trail.corpus_id or not trail.document_id:
        return f"{hit.document_id}: provenance does not name its corpus and document"
    if trail.kind is SourceKind.NONE or trail.link() is None:
        return f"{hit.document_id}: no openable link"
    if trail.kind is SourceKind.LOCAL_DOCUMENT and service.corpus.source(trail.source_id) is None:
        # The one check that catches a link that looks fine and 404s: the id has to be
        # in the registry the route serves from.
        return f"{hit.document_id}: local source id {trail.source_id!r} is not in the corpus registry"
    return None


def _image_on_disk(corpus: Corpus, image_id: str) -> bool:
    record = corpus.image(image_id)
    return record is not None and record.path.is_file()


def summarise(reports: Sequence[MethodReport]) -> dict[str, Any]:
    """Cross-method view: the comparison the report is built on."""
    return {
        "methods": [report.method for report in reports],
        "splits": sorted({report.split for report in reports}),
        "rows": [
            {
                "split": report.split,
                "method": report.method,
                "image_strategy": report.image_strategy,
                "questions": len(report.outcomes),
                "document_recall_at_1": round(report.document_recall_at(1), 4),
                "document_recall_at_3": round(report.document_recall_at(3), 4),
                "document_recall_at_5": round(report.document_recall_at(5), 4),
                "document_mrr": round(report.document_mrr, 4),
                "image_recall_at_1": round(report.image_recall_at(1), 4),
                "image_recall_at_3": round(report.image_recall_at(3), 4),
                "image_recall_at_5": round(report.image_recall_at(5), 4),
                "image_precision": round(report.image_precision, 4),
                "image_f1": round(report.image_f1, 4),
                "image_abstention_accuracy": round(report.image_abstention_accuracy, 4),
                "citation_validity": round(report.citation_validity, 4),
                "broken_source_rate": round(report.broken_source_rate, 4),
                "broken_image_rate": round(report.broken_image_rate, 4),
                "retrieval_p50_ms": round(report.latency.p50_ms, 1),
                "retrieval_p95_ms": round(report.latency.p95_ms, 1),
                "unresolved_offset_rate": round(report.unresolved_offset_rate, 4),
            }
            for report in reports
        ],
    }


__all__ = [
    "DEPTHS",
    "DOCUMENT_LIMIT",
    "IMAGE_LIMIT",
    "TARGET_CITATION_VALIDITY",
    "TARGET_DOCUMENT_RECALL_AT_3",
    "TARGET_IMAGE_RECALL_AT_5",
    "TARGET_RETRIEVAL_P95_MS",
    "MethodReport",
    "QuestionOutcome",
    "evaluate_method",
    "percentile",
    "summarise",
]

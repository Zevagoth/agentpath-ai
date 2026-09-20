"""Can the model actually read a flowchart? Six metrics, in the order they depend on each other.

The order is not arbitrary. Each metric is the ceiling of the next: a node that was not read
cannot be connected, an edge that was not found cannot be pointed the right way, and a
direction that is wrong makes the path wrong. Reporting them together is what turns "40%
accuracy" into a diagnosis.

1. **Text (OCR) recall** - what share of the node labels drawn in the figure appear anywhere
   in the model's answer. The floor: text it never read cannot be structured.
2. **Node detection** - precision, recall and F1 over the set of node labels, after
   normalisation. Plus count accuracy, which is a weaker but alignment-free check.
3. **Edge detection** - the same over **undirected** pairs, so a direction mistake is counted
   once, by metric 4, rather than twice.
4. **Arrow direction** - of the edges found, how many point as drawn. Also scored against the
   corpus's own probe: it supplies one ordered pair that *is* drawn and one that is not.
5. **Branch detection** - over nodes with more than one outgoing edge. These are the decision
   points, and they are the entire reason a workflow needs understanding rather than showing.
6. **Path reachability** - given the extracted graph, is the true pair reachable and the false
   pair not. The closest this corpus comes to "did you route the case correctly".

Alignment, and why two numbers are reported
-------------------------------------------
The model reports the words in a box; the corpus calls that box ``entity3``. So everything is
aligned on labels, normalised by casefolding, collapsing whitespace and stripping punctuation.
A label the model read with one character wrong then counts as missed.

That is strict, and strictness here can look like structural failure when it is really OCR
noise, so a **relaxed** variant is reported beside it: a label matches if one is contained in
the other after normalisation. Neither is the "real" number on its own - the gap between them
is how much of the error is reading rather than structure.

Nothing in this module is used to select a threshold, and none of it may be run against the
held-out split before the dev numbers are settled.
"""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentpath_ingestion.corpora.flowlearn import FlowchartRecord
from agentpath_ingestion.corpora.workflow import ExtractionStatus, WorkflowGraph

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace. The strict alignment key."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.casefold())).strip()


def relaxed_match(candidate: str, target: str) -> bool:
    """Whether two labels are the same thing allowing for reading noise.

    Containment either way, on normalised text, with a length floor so a two-letter fragment
    cannot match everything. Reported alongside the strict number, never instead of it.
    """
    left, right = normalise(candidate), normalise(target)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and shorter in longer


@dataclass(frozen=True, slots=True)
class SetScore:
    """Precision, recall and F1 for one flowchart's worth of one thing."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def exact(self) -> bool:
        return self.false_positives == 0 and self.false_negatives == 0


def score_sets(predicted: Sequence[str], truth: Sequence[str], *, relaxed: bool = False) -> SetScore:
    """Match predicted labels against truth, each truth label consumed at most once."""
    remaining = list(truth)
    hits = 0
    spurious = 0
    for item in predicted:
        found = None
        for index, candidate in enumerate(remaining):
            if relaxed_match(item, candidate) if relaxed else normalise(item) == normalise(candidate):
                found = index
                break
        if found is None:
            spurious += 1
        else:
            remaining.pop(found)
            hits += 1
    return SetScore(true_positives=hits, false_positives=spurious, false_negatives=len(remaining))


def score_pairs(
    predicted: Sequence[tuple[str, str]],
    truth: Sequence[tuple[str, str]],
    *,
    directed: bool,
    relaxed: bool = False,
) -> SetScore:
    """The same for edges. ``directed`` decides whether ``(a, b)`` and ``(b, a)`` differ."""

    def same(left: tuple[str, str], right: tuple[str, str]) -> bool:
        def eq(one: str, two: str) -> bool:
            return relaxed_match(one, two) if relaxed else normalise(one) == normalise(two)

        if eq(left[0], right[0]) and eq(left[1], right[1]):
            return True
        return not directed and eq(left[0], right[1]) and eq(left[1], right[0])

    remaining = list(truth)
    hits = 0
    spurious = 0
    for item in predicted:
        found = next((index for index, candidate in enumerate(remaining) if same(item, candidate)), None)
        if found is None:
            spurious += 1
        else:
            remaining.pop(found)
            hits += 1
    return SetScore(true_positives=hits, false_positives=spurious, false_negatives=len(remaining))


def reachable(edges: Sequence[tuple[str, str]], source: str, target: str) -> bool:
    """Whether *target* is reachable from *source* along directed edges.

    Breadth-first with a visited set, so a cycle terminates. Labels are normalised on the way
    in, because the graph came from a model and the query came from the corpus.
    """
    start, goal = normalise(source), normalise(target)
    adjacency: dict[str, list[str]] = {}
    for left, right in edges:
        adjacency.setdefault(normalise(left), []).append(normalise(right))
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == goal:
            return True
        for neighbour in adjacency.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return goal in seen and goal != start


@dataclass(frozen=True, slots=True)
class FlowchartOutcome:
    """Every metric for one flowchart, plus why it was excluded if it was."""

    flowchart_id: str
    status: str
    #: Set when this flowchart contributed to no metric.
    excluded: str | None = None
    #: 1. text
    label_recall: float = 0.0
    label_recall_relaxed: float = 0.0
    #: 2. nodes
    nodes: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    nodes_relaxed: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    node_count_correct: bool = False
    #: 3. edges, undirected
    edges: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    edges_relaxed: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    edge_count_correct: bool = False
    #: 4. direction
    directed_edges: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    direction_of_found_edges: float | None = None
    arrow_probe_correct: bool | None = None
    #: 5. branches
    branches: SetScore = field(default_factory=lambda: SetScore(0, 0, 0))
    #: 6. path
    reachability_correct: bool | None = None
    #: Operational
    needs_review: bool = False
    review_reasons: tuple[str, ...] = ()
    seconds: float = 0.0
    expected_nodes: int = 0
    expected_edges: int = 0
    predicted_nodes: int = 0
    predicted_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "flowchart_id": self.flowchart_id,
            "status": self.status,
            "excluded": self.excluded,
            "label_recall": round(self.label_recall, 4),
            "label_recall_relaxed": round(self.label_recall_relaxed, 4),
            "node_f1": round(self.nodes.f1, 4),
            "node_f1_relaxed": round(self.nodes_relaxed.f1, 4),
            "node_count_correct": self.node_count_correct,
            "edge_f1_undirected": round(self.edges.f1, 4),
            "edge_f1_undirected_relaxed": round(self.edges_relaxed.f1, 4),
            "edge_count_correct": self.edge_count_correct,
            "directed_edge_f1": round(self.directed_edges.f1, 4),
            "direction_of_found_edges": (
                round(self.direction_of_found_edges, 4) if self.direction_of_found_edges is not None else None
            ),
            "arrow_probe_correct": self.arrow_probe_correct,
            "branch_f1": round(self.branches.f1, 4),
            "reachability_correct": self.reachability_correct,
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "expected": {"nodes": self.expected_nodes, "edges": self.expected_edges},
            "predicted": {"nodes": self.predicted_nodes, "edges": self.predicted_edges},
            "seconds": round(self.seconds, 2),
        }


def score_flowchart(record: FlowchartRecord, graph: WorkflowGraph) -> FlowchartOutcome:
    """Score one extraction against its ground truth.

    A flowchart whose *ground truth* failed verification is excluded with the reason, because
    scoring against a graph we mis-parsed would report our parser's bug as the model's.

    An extraction the validator sent to review is **still scored**. Refusing to show a path to
    a user and refusing to measure are different things: the point of the numbers is to say
    how often the refusal was right.
    """
    if not record.trustworthy:
        return FlowchartOutcome(
            flowchart_id=record.flowchart_id,
            status=graph.status.value,
            excluded=f"ground truth unusable: {record.ground_truth_problems[0]}",
        )

    truth_labels = list(record.graph.labels)
    truth_edges = list(record.graph.label_edges)
    truth_branches = list(record.graph.branch_nodes)

    predicted_labels = [node.text for node in graph.nodes if node.text.strip()]
    by_id = {node.node_id: node.text for node in graph.nodes}
    predicted_edges = [
        (by_id[edge.source], by_id[edge.target])
        for edge in graph.edges
        if edge.source in by_id and edge.target in by_id
    ]

    # 1. text
    answer_text = normalise(" ".join([graph.visible_text, *predicted_labels]))
    strict_found = sum(1 for label in truth_labels if normalise(label) in answer_text)
    answer_pieces = [graph.visible_text, *predicted_labels]
    relaxed_found = sum(1 for label in truth_labels if any(relaxed_match(label, piece) for piece in answer_pieces))

    # 2, 3, 4
    nodes = score_sets(predicted_labels, truth_labels)
    nodes_relaxed = score_sets(predicted_labels, truth_labels, relaxed=True)
    undirected = score_pairs(predicted_edges, truth_edges, directed=False)
    undirected_relaxed = score_pairs(predicted_edges, truth_edges, directed=False, relaxed=True)
    directed = score_pairs(predicted_edges, truth_edges, directed=True)
    direction_rate = (directed.true_positives / undirected.true_positives) if undirected.true_positives else None

    # 4b: the corpus's own ordered-pair probe
    probe = None
    true_pair = record.arrow_a_to_b.get("true") or {}
    false_pair = record.arrow_a_to_b.get("false") or {}
    if true_pair and false_pair:
        has_true = any(
            relaxed_match(left, true_pair.get("a", "")) and relaxed_match(right, true_pair.get("b", ""))
            for left, right in predicted_edges
        )
        has_false = any(
            relaxed_match(left, false_pair.get("a", "")) and relaxed_match(right, false_pair.get("b", ""))
            for left, right in predicted_edges
        )
        probe = bool(has_true and not has_false)

    # 5
    predicted_branch_labels: list[str] = []
    outgoing: dict[str, int] = {}
    for left, _right in predicted_edges:
        outgoing[left] = outgoing.get(left, 0) + 1
    predicted_branch_labels = [label for label, count in outgoing.items() if count > 1]
    branches = score_sets(predicted_branch_labels, truth_branches, relaxed=True)

    # 6
    reachability = None
    if true_pair and false_pair:
        reachability = bool(
            reachable(predicted_edges, true_pair.get("a", ""), true_pair.get("b", ""))
            and not reachable(predicted_edges, false_pair.get("a", ""), false_pair.get("b", ""))
        )

    return FlowchartOutcome(
        flowchart_id=record.flowchart_id,
        status=graph.status.value,
        label_recall=strict_found / len(truth_labels) if truth_labels else 0.0,
        label_recall_relaxed=relaxed_found / len(truth_labels) if truth_labels else 0.0,
        nodes=nodes,
        nodes_relaxed=nodes_relaxed,
        node_count_correct=len(predicted_labels) == record.expected_nodes,
        edges=undirected,
        edges_relaxed=undirected_relaxed,
        edge_count_correct=len(predicted_edges) == record.expected_arrows,
        directed_edges=directed,
        direction_of_found_edges=direction_rate,
        arrow_probe_correct=probe,
        branches=branches,
        reachability_correct=reachability,
        needs_review=graph.status is not ExtractionStatus.OK,
        review_reasons=graph.review_reasons,
        seconds=graph.seconds or 0.0,
        expected_nodes=record.expected_nodes,
        expected_edges=record.expected_arrows,
        predicted_nodes=len(predicted_labels),
        predicted_edges=len(predicted_edges),
    )


@dataclass
class WorkflowReport:
    """Aggregated metrics for one split."""

    corpus_id: str
    split: str
    model: str
    outcomes: list[FlowchartOutcome] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def scored(self) -> list[FlowchartOutcome]:
        return [outcome for outcome in self.outcomes if outcome.excluded is None]

    @property
    def excluded(self) -> list[FlowchartOutcome]:
        return [outcome for outcome in self.outcomes if outcome.excluded is not None]

    def _mean(self, extract) -> float:
        values = [value for outcome in self.scored if (value := extract(outcome)) is not None]
        return statistics.fmean(values) if values else 0.0

    @property
    def metrics(self) -> dict[str, Any]:
        scored = self.scored
        failed = [outcome for outcome in scored if outcome.status == "failed"]
        return {
            "1_text": {
                "label_recall": round(self._mean(lambda o: o.label_recall), 4),
                "label_recall_relaxed": round(self._mean(lambda o: o.label_recall_relaxed), 4),
            },
            "2_nodes": {
                "precision": round(self._mean(lambda o: o.nodes.precision), 4),
                "recall": round(self._mean(lambda o: o.nodes.recall), 4),
                "f1": round(self._mean(lambda o: o.nodes.f1), 4),
                "f1_relaxed": round(self._mean(lambda o: o.nodes_relaxed.f1), 4),
                "exact_set_rate": round(sum(1 for o in scored if o.nodes.exact) / len(scored), 4) if scored else 0.0,
                "count_accuracy": round(sum(1 for o in scored if o.node_count_correct) / len(scored), 4)
                if scored
                else 0.0,
            },
            "3_edges_undirected": {
                "precision": round(self._mean(lambda o: o.edges.precision), 4),
                "recall": round(self._mean(lambda o: o.edges.recall), 4),
                "f1": round(self._mean(lambda o: o.edges.f1), 4),
                "f1_relaxed": round(self._mean(lambda o: o.edges_relaxed.f1), 4),
                "count_accuracy": round(sum(1 for o in scored if o.edge_count_correct) / len(scored), 4)
                if scored
                else 0.0,
            },
            "4_arrow_direction": {
                "directed_edge_f1": round(self._mean(lambda o: o.directed_edges.f1), 4),
                "direction_correct_among_found_edges": round(self._mean(lambda o: o.direction_of_found_edges), 4),
                "corpus_ordered_pair_probe": round(self._mean(lambda o: 1.0 if o.arrow_probe_correct else 0.0), 4),
            },
            "5_branches": {
                "precision": round(self._mean(lambda o: o.branches.precision), 4),
                "recall": round(self._mean(lambda o: o.branches.recall), 4),
                "f1": round(self._mean(lambda o: o.branches.f1), 4),
            },
            "6_path_reachability": {
                "accuracy": round(self._mean(lambda o: 1.0 if o.reachability_correct else 0.0), 4),
            },
            "operational": {
                "flowcharts": len(self.outcomes),
                "scored": len(scored),
                "excluded_unusable_ground_truth": len(self.excluded),
                "needs_review_rate": round(sum(1 for o in scored if o.needs_review) / len(scored), 4)
                if scored
                else 0.0,
                "extraction_failed": len(failed),
                "median_seconds": round(statistics.median([o.seconds for o in scored]), 2) if scored else 0.0,
                "total_seconds": round(self.seconds, 1),
            },
        }

    def review_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.scored:
            for reason in outcome.review_reasons:
                # The reason's first clause is the category; the numbers vary per flowchart.
                key = reason.split(";")[0].split(":")[0][:70]
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda row: -row[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "split": self.split,
            "model": self.model,
            "metrics": self.metrics,
            "review_reasons": self.review_reason_counts(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def evaluate(
    records: Sequence[FlowchartRecord],
    extractor,
    *,
    corpus_id: str,
    split: str,
    on_progress=None,
) -> WorkflowReport:
    """Extract and score every flowchart in *records*."""
    report = WorkflowReport(corpus_id=corpus_id, split=split, model=getattr(extractor, "model", "unknown"))
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        graph = extractor.extract(
            record.image_path,
            image_id=record.flowchart_id,
            document_id=record.flowchart_id,
            corpus_id=corpus_id,
        )
        report.outcomes.append(score_flowchart(record, graph))
        if on_progress is not None:
            on_progress(index, len(records), report.outcomes[-1])
    report.seconds = time.perf_counter() - started
    return report


__all__ = [
    "FlowchartOutcome",
    "SetScore",
    "WorkflowReport",
    "evaluate",
    "normalise",
    "reachable",
    "relaxed_match",
    "score_flowchart",
    "score_pairs",
    "score_sets",
]

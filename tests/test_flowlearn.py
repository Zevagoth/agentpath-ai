"""FlowLearn: parsing the ground truth, and scoring an extraction against it.

Two things are under test and the first matters more than the second.

**The ground-truth parser.** If it mis-reads the corpus's Mermaid, every metric becomes a
measurement of our bug while looking like a measurement of the model. So it is tested
against every arrow style the corpus uses, and there is a verification step that refuses a
flowchart whose parse disagrees with the counts the corpus itself states.

**The scorers.** Precision, recall, direction and reachability are arithmetic, and arithmetic
that is wrong in our favour is the easiest way to publish a flattering number. Each is tested
on hand-built cases where the right answer is obvious by inspection.

No network and no model: the licence permits evaluation use, and a test suite that needed a
3.39 GB download to run would not be run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentpath_eval.workflow_eval import (
    SetScore,
    normalise,
    reachable,
    relaxed_match,
    score_flowchart,
    score_pairs,
    score_sets,
)
from agentpath_ingestion.corpora.flowlearn import (
    CORPUS_ID,
    CORPUS_LICENCE,
    SAMPLE_SEED,
    FlowchartRecord,
    MermaidGraph,
    corpus_fingerprint,
    parse_mermaid,
    selected_ids,
    verify_parse,
)
from agentpath_ingestion.corpora.workflow import ExtractionStatus, NodeKind, WorkflowEdge, WorkflowGraph, WorkflowNode

# --------------------------------------------------------------------------- #
# Parsing the corpus's Mermaid
# --------------------------------------------------------------------------- #

#: A real sample, copied from the corpus. Three arrow styles appear in FlowLearn - plain
#: `-->`, thick `==>` and dotted `-..->` - and all three carry the same meaning.
REAL_SAMPLE = """flowchart TB
entity0(lansing)
entity1(thievishness)
entity2(strauss testier)
entity3(tartryl flockmaster)
entity4(supercargoship)
entity0 -..-> entity3
entity1 -..-> entity2
entity2 ==> entity3
entity2 -..-> entity4
entity2 -..-> entity0
entity3 --> entity4
"""


def test_the_real_sample_parses_to_the_right_shape() -> None:
    graph = parse_mermaid(REAL_SAMPLE)

    assert graph.orientation == "TB"
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 6
    assert graph.nodes["entity0"] == "lansing"
    assert graph.nodes["entity3"] == "tartryl flockmaster"


@pytest.mark.parametrize(
    "arrow",
    ["-->", "==>", "-..->", "-.->", "---", "===", "-.-"],
)
def test_every_arrow_style_the_corpus_uses_is_recognised(arrow: str) -> None:
    """A style silently dropped here would look like the model missing an edge."""
    source = f"flowchart TB\nentity0(a)\nentity1(b)\nentity0 {arrow} entity1\n"

    graph = parse_mermaid(source)

    assert graph.edges == (("entity0", "entity1"),), f"{arrow} was not parsed"


@pytest.mark.parametrize("brackets", ["({})", "[{}]", "{{{}}}"])
def test_every_node_shape_is_recognised(brackets: str) -> None:
    """Mermaid uses the bracket shape to mean round, square or diamond. All are nodes."""
    body = brackets.format("some label")
    graph = parse_mermaid(f"flowchart TB\nentity0{body}\n")

    assert graph.nodes.get("entity0") == "some label"


def test_a_labelled_edge_is_parsed_without_swallowing_the_label() -> None:
    graph = parse_mermaid("flowchart TB\nentity0(a)\nentity1(b)\nentity0 -->|Yes| entity1\n")

    assert graph.edges == (("entity0", "entity1"),)


def test_a_fenced_mermaid_block_is_unwrapped() -> None:
    """The corpus's VQA field wraps the source in a markdown fence."""
    graph = parse_mermaid("```mermaid\nflowchart TB\nentity0(a)\nentity1(b)\nentity0 --> entity1\n```")

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


def test_an_unrecognised_line_is_ignored_rather_than_guessed_at() -> None:
    graph = parse_mermaid("flowchart TB\nentity0(a)\n%% a comment\nsubgraph cluster\nend\n")

    assert len(graph.nodes) == 1
    assert graph.edges == ()


def test_labels_and_label_edges_use_the_words_not_the_entity_ids() -> None:
    """A model can only report the words in a box, so alignment happens on labels."""
    graph = parse_mermaid(REAL_SAMPLE)

    assert "lansing" in graph.labels
    assert ("strauss testier", "tartryl flockmaster") in graph.label_edges
    assert all("entity" not in left for left, _ in graph.label_edges)


def test_an_edge_naming_an_undeclared_node_is_left_out_of_label_edges() -> None:
    graph = parse_mermaid("flowchart TB\nentity0(a)\nentity0 --> entity9\n")

    assert graph.edges == (("entity0", "entity9"),)
    assert graph.label_edges == (), "it cannot be expressed as labels, so it is not offered as one"


def test_branch_nodes_are_those_with_more_than_one_way_out() -> None:
    """The decision points. The entire reason a workflow needs understanding."""
    graph = parse_mermaid(REAL_SAMPLE)

    # entity2 goes to entity3, entity4 and entity0.
    assert graph.branch_nodes == ("strauss testier",)


def test_a_chart_with_no_branch_reports_none() -> None:
    source = "flowchart TB\nentity0(a)\nentity1(b)\nentity2(c)\nentity0 --> entity1\nentity1 --> entity2\n"
    graph = parse_mermaid(source)

    assert graph.branch_nodes == ()


# --------------------------------------------------------------------------- #
# Refusing ground truth we cannot trust
# --------------------------------------------------------------------------- #


def test_a_parse_matching_the_corpus_counts_has_no_problems() -> None:
    graph = parse_mermaid(REAL_SAMPLE)

    assert verify_parse(graph, expected_nodes=5, expected_arrows=6) == []


def test_a_parse_disagreeing_on_node_count_is_refused() -> None:
    """This is the check that keeps our parser's bug from being reported as the model's."""
    graph = parse_mermaid(REAL_SAMPLE)

    problems = verify_parse(graph, expected_nodes=9, expected_arrows=6)

    assert problems
    assert "9" in problems[0]


def test_a_parse_disagreeing_on_edge_count_is_refused() -> None:
    graph = parse_mermaid(REAL_SAMPLE)

    problems = verify_parse(graph, expected_nodes=5, expected_arrows=99)

    assert any("99" in problem for problem in problems)


def test_two_nodes_sharing_a_label_is_refused_as_unalignable() -> None:
    """If two boxes read the same, a model's answer cannot be matched to one of them."""
    graph = MermaidGraph(nodes={"entity0": "same", "entity1": "same"}, edges=(("entity0", "entity1"),))

    problems = verify_parse(graph, expected_nodes=2, expected_arrows=1)

    assert any("share a label" in problem for problem in problems)


# --------------------------------------------------------------------------- #
# Which flowcharts, reproducibly
# --------------------------------------------------------------------------- #


def test_the_sample_is_deterministic_and_order_independent() -> None:
    annotations = {f"{index}.jpeg": {} for index in range(200)}
    reversed_annotations = dict(reversed(list(annotations.items())))

    first = selected_ids(annotations, sample=20)
    second = selected_ids(reversed_annotations, sample=20)

    assert first == second
    assert len(first) == 20


def test_a_different_sample_seed_selects_differently() -> None:
    annotations = {f"{index}.jpeg": {} for index in range(200)}

    assert selected_ids(annotations, sample=20, seed=1) != selected_ids(annotations, sample=20, seed=2)


def test_a_larger_sample_extends_rather_than_reshuffles() -> None:
    """So downloading more flowcharts later does not invalidate earlier numbers."""
    annotations = {f"{index}.jpeg": {} for index in range(200)}

    small = selected_ids(annotations, sample=10)
    large = selected_ids(annotations, sample=40)

    assert large[:10] == small


def test_the_sample_seed_is_recorded_as_a_constant() -> None:
    assert SAMPLE_SEED == 20260920


def test_the_licence_is_recorded_on_the_corpus() -> None:
    """Non-commercial. Asserted so it cannot be quietly dropped from a report."""
    assert CORPUS_LICENCE == "CC-BY-NC-SA-4.0"
    assert CORPUS_ID == "flowlearn-sim"


def test_the_fingerprint_depends_only_on_which_flowcharts_are_present() -> None:
    a = _record("1.jpeg")
    b = _record("2.jpeg")

    assert corpus_fingerprint((a, b)) == corpus_fingerprint((b, a))
    assert corpus_fingerprint((a,)) != corpus_fingerprint((a, b))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_normalisation_folds_case_punctuation_and_whitespace() -> None:
    assert normalise("  Strauss,  Testier! ") == "strauss testier"


def test_a_perfect_set_scores_one() -> None:
    score = score_sets(["a", "b"], ["b", "a"])

    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0
    assert score.exact


def test_a_missed_item_costs_recall_and_not_precision() -> None:
    score = score_sets(["a"], ["a", "b"])

    assert score.precision == 1.0
    assert score.recall == 0.5
    assert not score.exact


def test_a_spurious_item_costs_precision_and_not_recall() -> None:
    score = score_sets(["a", "invented"], ["a"])

    assert score.recall == 1.0
    assert score.precision == 0.5


def test_a_truth_item_is_consumed_once_so_duplicates_do_not_score_twice() -> None:
    """Otherwise a model repeating one correct label would score full recall."""
    score = score_sets(["a", "a", "a"], ["a", "b", "c"])

    assert score.true_positives == 1
    assert score.false_positives == 2
    assert score.false_negatives == 2


def test_strict_matching_rejects_a_misread_label_and_relaxed_accepts_it() -> None:
    """The gap between the two is how much of the error is reading rather than structure."""
    strict = score_sets(["strauss testie"], ["strauss testier"])
    relaxed = score_sets(["strauss testie"], ["strauss testier"], relaxed=True)

    assert strict.true_positives == 0
    assert relaxed.true_positives == 1


def test_relaxed_matching_will_not_let_a_short_fragment_match_everything() -> None:
    assert not relaxed_match("ab", "abcdefgh")
    assert relaxed_match("abcd", "abcdefgh")


def test_an_empty_score_is_zero_rather_than_a_division_error() -> None:
    score = SetScore(0, 0, 0)

    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_undirected_edge_scoring_accepts_either_order() -> None:
    score = score_pairs([("b", "a")], [("a", "b")], directed=False)

    assert score.true_positives == 1


def test_directed_edge_scoring_rejects_the_reverse() -> None:
    """The distinction the whole arrow-direction metric rests on."""
    score = score_pairs([("b", "a")], [("a", "b")], directed=True)

    assert score.true_positives == 0
    assert score.false_positives == 1


def test_reachability_follows_direction() -> None:
    edges = [("a", "b"), ("b", "c")]

    assert reachable(edges, "a", "c")
    assert not reachable(edges, "c", "a")


def test_reachability_terminates_on_a_cycle() -> None:
    edges = [("a", "b"), ("b", "a")]

    assert reachable(edges, "a", "b")
    assert not reachable(edges, "a", "z")


def test_reachability_normalises_labels_before_comparing() -> None:
    """The graph came from a model and the query came from the corpus."""
    assert reachable([("Node A", "node  b")], "node a", "Node B")


# --------------------------------------------------------------------------- #
# Scoring a whole extraction
# --------------------------------------------------------------------------- #


def _record(flowchart_id: str = "1.jpeg", *, problems: tuple[str, ...] = ()) -> FlowchartRecord:
    graph = parse_mermaid(
        "flowchart TB\nentity0(start box)\nentity1(middle box)\nentity2(end box)\n"
        "entity0 --> entity1\nentity1 --> entity2\nentity0 --> entity2\n"
    )
    return FlowchartRecord(
        flowchart_id=flowchart_id,
        image_path=Path("/nonexistent.jpeg"),
        graph=graph,
        mermaid_source="",
        expected_nodes=3,
        expected_arrows=3,
        ocr_text="start box",
        description="start box points to middle box.",
        arrow_a_to_b={"true": {"a": "start box", "b": "middle box"}, "false": {"a": "end box", "b": "start box"}},
        arrow_between={},
        ground_truth_problems=problems,
    )


def _extraction(
    *,
    nodes: tuple[tuple[str, str], ...] = (("n0", "start box"), ("n1", "middle box"), ("n2", "end box")),
    edges: tuple[tuple[str, str], ...] = (("n0", "n1"), ("n1", "n2"), ("n0", "n2")),
    status: ExtractionStatus = ExtractionStatus.OK,
) -> WorkflowGraph:
    return WorkflowGraph(
        image_id="1.jpeg",
        document_id="1.jpeg",
        corpus_id=CORPUS_ID,
        status=status,
        confidence=0.9,
        nodes=tuple(WorkflowNode(node_id=nid, kind=NodeKind.ACTION, text=text) for nid, text in nodes),
        edges=tuple(WorkflowEdge(source=source, target=target) for source, target in edges),
        visible_text="start box middle box end box",
    )


def test_a_perfect_extraction_scores_perfectly_on_every_metric() -> None:
    outcome = score_flowchart(_record(), _extraction())

    assert outcome.excluded is None
    assert outcome.label_recall == 1.0
    assert outcome.nodes.f1 == 1.0
    assert outcome.node_count_correct
    assert outcome.edges.f1 == 1.0
    assert outcome.edge_count_correct
    assert outcome.directed_edges.f1 == 1.0
    assert outcome.direction_of_found_edges == 1.0
    assert outcome.arrow_probe_correct is True
    assert outcome.branches.f1 == 1.0
    assert outcome.reachability_correct is True


def test_reversing_every_arrow_keeps_edge_detection_and_destroys_direction() -> None:
    """The separation that makes the report a diagnosis instead of one number."""
    reversed_edges = (("n1", "n0"), ("n2", "n1"), ("n2", "n0"))

    outcome = score_flowchart(_record(), _extraction(edges=reversed_edges))

    assert outcome.edges.f1 == 1.0, "the connections are all still found"
    assert outcome.directed_edges.f1 == 0.0, "and every one of them points the wrong way"
    assert outcome.direction_of_found_edges == 0.0
    assert outcome.arrow_probe_correct is False


def test_a_missing_edge_lowers_recall_and_leaves_direction_alone() -> None:
    outcome = score_flowchart(_record(), _extraction(edges=(("n0", "n1"), ("n1", "n2"))))

    assert outcome.edges.recall == pytest.approx(2 / 3)
    assert outcome.edges.precision == 1.0
    assert outcome.direction_of_found_edges == 1.0
    assert not outcome.edge_count_correct


def test_an_invented_edge_lowers_precision() -> None:
    outcome = score_flowchart(_record(), _extraction(edges=(("n0", "n1"), ("n1", "n2"), ("n0", "n2"), ("n2", "n0"))))

    assert outcome.edges.recall == 1.0
    assert outcome.edges.precision < 1.0


def test_an_edge_naming_a_node_that_was_not_extracted_is_ignored() -> None:
    outcome = score_flowchart(_record(), _extraction(edges=(("n0", "n1"), ("n0", "ghost"))))

    assert outcome.predicted_edges == 1


def test_a_flowchart_with_unusable_ground_truth_is_excluded_not_scored() -> None:
    """Scoring against a graph we mis-parsed would report our bug as the model's."""
    outcome = score_flowchart(_record(problems=("parsed 4 nodes but the corpus says 3",)), _extraction())

    assert outcome.excluded is not None
    assert "ground truth unusable" in outcome.excluded


def test_an_extraction_sent_to_review_is_still_scored() -> None:
    """Refusing to show a path and refusing to measure are different things. The numbers
    exist to say how often the refusal was right."""
    outcome = score_flowchart(_record(), _extraction(status=ExtractionStatus.NEEDS_REVIEW))

    assert outcome.excluded is None
    assert outcome.needs_review
    assert outcome.nodes.f1 == 1.0


def test_an_empty_extraction_scores_zero_without_raising() -> None:
    outcome = score_flowchart(_record(), _extraction(nodes=(), edges=()))

    assert outcome.nodes.f1 == 0.0
    assert outcome.edges.f1 == 0.0
    assert outcome.direction_of_found_edges is None
    assert outcome.reachability_correct is False


def test_text_recall_counts_labels_found_anywhere_in_the_answer() -> None:
    """The floor metric: text never read cannot be structured."""
    graph = _extraction(nodes=(("n0", "start box"),), edges=())

    outcome = score_flowchart(_record(), graph)

    # visible_text mentions all three labels even though only one node was extracted.
    assert outcome.label_recall == 1.0
    assert outcome.nodes.recall == pytest.approx(1 / 3)


def test_a_branch_the_model_missed_lowers_branch_recall() -> None:
    outcome = score_flowchart(_record(), _extraction(edges=(("n0", "n1"), ("n1", "n2"))))

    # The truth has start box branching to both middle and end; the extraction does not.
    assert outcome.branches.recall == 0.0

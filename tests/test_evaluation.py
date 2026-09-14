"""Tests for the benchmark harness.

A benchmark that reports the wrong number is worse than no benchmark, so the
metric arithmetic is tested directly against hand-computed cases, and the query
set is checked for the contradictions that would silently corrupt a report.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentpath_eval.baseline import LexicalBaseline, build_index
from agentpath_eval.metrics import (
    LatencySummary,
    ModeReport,
    QueryOutcome,
    hit_at_k,
    percentile,
    reciprocal_rank,
)
from agentpath_eval.queries import QuerySetError, load_queries, queries_path, split_by_support

# --------------------------------------------------------------------------- #
# Metric arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ranked", "expected", "result"),
    [
        (["A", "B", "C"], ["A"], 1.0),
        (["B", "A", "C"], ["A"], 0.5),
        (["B", "C", "A"], ["A"], 1 / 3),
        (["B", "C", "D"], ["A"], 0.0),
        ([], ["A"], 0.0),
        (["B", "A"], ["A", "B"], 1.0),
    ],
)
def test_reciprocal_rank(ranked: list[str], expected: list[str], result: float) -> None:
    assert reciprocal_rank(ranked, expected) == pytest.approx(result)


@pytest.mark.parametrize(
    ("ranked", "k", "hit"),
    [
        (["A", "B", "C"], 1, True),
        (["B", "A", "C"], 1, False),
        (["B", "A", "C"], 3, True),
        (["B", "C", "D", "A"], 3, False),
        ([], 3, False),
    ],
)
def test_hit_at_k(ranked: list[str], k: int, hit: bool) -> None:
    assert hit_at_k(ranked, ["A"], k) is hit


def test_percentile_is_nearest_rank_not_interpolated() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.95) == 100.0
    # Every reported percentile is a latency some request actually had.
    assert percentile(values, 0.95) in values


def test_percentile_handles_empty_input() -> None:
    assert percentile([], 0.95) == 0.0


def test_latency_summary_statistics() -> None:
    summary = LatencySummary(samples=[100.0, 200.0, 300.0])
    assert summary.count == 3
    assert summary.mean_ms == pytest.approx(200.0)
    assert summary.p50_ms == 200.0
    assert summary.max_ms == 300.0


# --------------------------------------------------------------------------- #
# Population separation
# --------------------------------------------------------------------------- #


def _outcome(
    *,
    supported: bool,
    ranked: tuple[str, ...],
    abstained: bool,
    expected: tuple[str, ...] = ("A",),
    query_type: str = "synonym",
) -> QueryOutcome:
    return QueryOutcome(
        query_id="q",
        query="q",
        type=query_type,
        language="en",
        expected=expected if supported else (),
        supported=supported,
        ranked=ranked,
        abstained=abstained,
        evidence="partial",
        latency_ms=100.0,
    )


def test_recall_counts_supported_queries_only() -> None:
    """An unsupported query must never enter the recall denominator."""
    report = ModeReport(mode="test", description="")
    report.outcomes = [
        _outcome(supported=True, ranked=("A",), abstained=False),
        _outcome(supported=True, ranked=("B",), abstained=False),
        _outcome(supported=False, ranked=(), abstained=True, expected=()),
        _outcome(supported=False, ranked=(), abstained=True, expected=()),
    ]
    assert report.recall_at_1 == pytest.approx(0.5)
    assert len(report.supported) == 2
    assert len(report.unsupported) == 2


def test_abstention_accuracy_counts_unsupported_queries_only() -> None:
    report = ModeReport(mode="test", description="")
    report.outcomes = [
        _outcome(supported=True, ranked=("A",), abstained=False),
        _outcome(supported=False, ranked=(), abstained=True, expected=()),
        _outcome(supported=False, ranked=("A",), abstained=False, expected=()),
    ]
    assert report.abstention_accuracy == pytest.approx(0.5)


def test_false_abstention_is_reported_separately() -> None:
    """Refusing an answerable query is a distinct failure from missing it."""
    report = ModeReport(mode="test", description="")
    report.outcomes = [
        _outcome(supported=True, ranked=(), abstained=True),
        _outcome(supported=True, ranked=("A",), abstained=False),
    ]
    assert report.false_abstention_rate == pytest.approx(0.5)
    assert report.recall_at_1 == pytest.approx(0.5)


def test_mrr_averages_over_supported_only() -> None:
    report = ModeReport(mode="test", description="")
    report.outcomes = [
        _outcome(supported=True, ranked=("A",), abstained=False),
        _outcome(supported=True, ranked=("B", "A"), abstained=False),
        _outcome(supported=False, ranked=(), abstained=True, expected=()),
    ]
    assert report.mrr == pytest.approx(0.75)


def test_by_type_reports_abstention_for_unsupported_groups() -> None:
    report = ModeReport(mode="test", description="")
    report.outcomes = [
        _outcome(supported=True, ranked=("A",), abstained=False, query_type="typo"),
        _outcome(supported=False, ranked=(), abstained=True, expected=(), query_type="unsupported"),
    ]
    breakdown = report.by_type()
    assert breakdown["typo"]["recall_at_3"] == 1.0
    assert breakdown["unsupported"]["abstention_accuracy"] == 1.0


def test_report_serializes_the_two_populations_separately() -> None:
    report = ModeReport(mode="test", description="d")
    report.outcomes = [_outcome(supported=True, ranked=("A",), abstained=False)]
    payload = report.to_dict()
    assert "retrieval_quality_supported_only" in payload
    assert "abstention_unsupported_only" in payload
    assert payload["counts"]["supported"] == 1


# --------------------------------------------------------------------------- #
# Query set
# --------------------------------------------------------------------------- #


def test_real_query_set_loads_and_covers_the_required_categories() -> None:
    queries = load_queries()
    supported, unsupported = split_by_support(queries)

    assert len(queries) >= 12, "the slice-1 target is at least 12 queries"
    assert supported and unsupported
    types = {query.type for query in queries}
    for required in ("exact_title", "synonym", "arabic", "typo", "abbreviation", "unsupported"):
        assert required in types, required


def test_every_expected_procedure_exists_in_the_dataset() -> None:
    """A typo in an expectation would quietly depress recall forever."""
    from agentpath_ingestion.dataset import load_procedures

    known = {item.procedure_id for item in load_procedures()}
    for query in load_queries():
        for procedure_id in query.expected:
            assert procedure_id in known, f"{query.id} expects unknown procedure {procedure_id}"


def test_unsupported_queries_expect_abstention() -> None:
    for query in load_queries():
        if query.type == "unsupported":
            assert query.expect_abstain
            assert not query.expected


def test_query_set_rejects_contradictory_expectations(tmp_path: Path) -> None:
    payload = {
        "version": "1.0",
        "queries": [
            {
                "id": "q1",
                "query": "x",
                "type": "synonym",
                "language": "en",
                "expected": ["NW-HTL-014"],
                "expect_abstain": True,
            }
        ],
    }
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(QuerySetError, match="cannot both"):
        load_queries(path)


def test_query_set_rejects_a_query_with_no_expectation_at_all(tmp_path: Path) -> None:
    payload = {
        "version": "1.0",
        "queries": [{"id": "q1", "query": "x", "type": "synonym", "language": "en"}],
    }
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(QuerySetError, match="expect_abstain"):
        load_queries(path)


def test_query_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    entry = {
        "id": "q1",
        "query": "x",
        "type": "synonym",
        "language": "en",
        "expected": ["NW-HTL-014"],
    }
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump({"version": "1.0", "queries": [entry, entry]}), encoding="utf-8")
    with pytest.raises(QuerySetError, match="duplicate"):
        load_queries(path)


def test_queries_file_is_where_the_loader_expects_it() -> None:
    assert queries_path().is_file()


# --------------------------------------------------------------------------- #
# Lexical baseline
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def baseline() -> LexicalBaseline:
    return LexicalBaseline()


def test_baseline_indexes_every_procedure(baseline: LexicalBaseline) -> None:
    assert baseline.document_count >= 5


def test_baseline_finds_an_exact_title(baseline: LexicalBaseline) -> None:
    ranked, abstained = baseline.search("Property unable to accommodate a confirmed guest")
    assert not abstained
    assert ranked[0][0] == "NW-HTL-014"


def test_baseline_reaches_a_listed_synonym_but_ranks_it_imperfectly(
    baseline: LexicalBaseline,
) -> None:
    """Synonyms live in the document text, so keyword overlap can reach them.

    Ranking them correctly is another matter: "hotel has no room" shares the terms
    "hotel" and "room" with the booking-not-found procedure too, and plain term
    overlap has no way to prefer the right one. The expected procedure lands in the
    top 3 rather than first, which is precisely the gap the hybrid pipeline closes.
    """
    ranked, abstained = baseline.search("hotel has no room")
    assert not abstained
    assert "NW-HTL-014" in [procedure_id for procedure_id, _ in ranked]


def test_baseline_cannot_handle_typos(baseline: LexicalBaseline) -> None:
    """This is the gap the AgentPath pipeline exists to close.

    A keyword index has no way to connect "overboking" to "overbooking", so it
    returns nothing at all - which is exactly why the benchmark needs this mode as
    the reference point.
    """
    ranked, abstained = baseline.search("overboking at the hotle")
    assert ranked == []
    assert abstained


def test_baseline_abstains_on_an_unrelated_question(baseline: LexicalBaseline) -> None:
    ranked, abstained = baseline.search("how do I reset my email password")
    assert abstained
    assert ranked == []


def test_baseline_respects_a_product_line_filter(baseline: LexicalBaseline) -> None:
    ranked, _ = baseline.search("booking", limit=5, product_line="refunds")
    assert all(procedure_id.startswith("NW-REF") for procedure_id, _ in ranked)


def test_baseline_ranking_is_deterministic(baseline: LexicalBaseline) -> None:
    first, _ = baseline.search("hotel cannot find the booking")
    second, _ = baseline.search("hotel cannot find the booking")
    assert first == second


def test_baseline_ignores_stopwords_only_queries(baseline: LexicalBaseline) -> None:
    ranked, abstained = baseline.search("the and of to")
    assert abstained
    assert ranked == []


def test_baseline_index_holds_weighted_terms() -> None:
    documents = build_index()
    hotel = next(doc for doc in documents if doc.procedure_id == "NW-HTL-014")
    # A title term must outweigh the same term appearing once in the body.
    assert hotel.term_weights.get("accommodate", 0) > 0
    assert hotel.term_weights.get("overbooking", 0) > 0

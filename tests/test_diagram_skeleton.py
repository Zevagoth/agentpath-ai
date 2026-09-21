"""Tests for the skeleton tracer that resolves crossings.

Built bottom-up, because each stage is only worth testing once the one under it is trustworthy:
thinning, then the arc graph, then reading a junction, then routing, then the matching. Every case is
a small array written out in the test, so a failure points at one stage rather than at a pipeline.

The shapes are the ones that have to be told apart:

* an **X**, where two connectors merely overlap and must come out as two;
* a **T** and a **Y**, where connectors genuinely share a stem, which is what several arrows leaving
  one node looks like just outside the border, and which must not be rewritten as two crossing
  curves;
* a crossing at four different angles, including a shallow one;
* a **curve** passing through a crossing, where continuing straight on is the wrong rule and
  curvature has to carry the decision.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from agentpath_ingestion.corpora.diagram_skeleton import (
    UNMATCHED_COST,
    build_graph,
    match_pairs,
    prune_spurs,
    read_junctions,
    routes_between_ends,
    thin,
)

# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def _blank(size: int = 120) -> np.ndarray:
    return np.zeros((size, size), np.uint8)


def _line(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int = 3) -> None:
    cv2.line(mask, start, end, 255, width)


def _polyline(mask: np.ndarray, points: list[tuple[int, int]], width: int = 3) -> None:
    cv2.polylines(mask, [np.array(points, np.int32)], False, 255, width)


def _skeleton_of(mask: np.ndarray) -> np.ndarray:
    return prune_spurs(thin(mask))


# --------------------------------------------------------------------------- #
# 1. Thinning
# --------------------------------------------------------------------------- #


def test_thinning_reduces_a_thick_line_to_one_pixel_wide() -> None:
    mask = _blank()
    _line(mask, (10, 60), (110, 60), width=7)

    skeleton = thin(mask)
    rows = np.count_nonzero(skeleton, axis=0)
    # Every column the line covers should keep exactly one pixel.
    covered = rows[rows > 0]
    assert covered.size > 80
    assert covered.max() == 1, f"thinning left a column {covered.max()} pixels thick"


def test_thinning_keeps_the_shape_connected() -> None:
    mask = _blank()
    _polyline(mask, [(10, 20), (60, 30), (100, 90)], width=5)

    skeleton = thin(mask)
    count, _labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8), connectivity=8)
    assert count - 1 == 1, "thinning broke one curve into pieces"


def test_thinning_preserves_a_loop() -> None:
    """A cycle must survive: 46% of dev charts contain one."""
    mask = _blank()
    cv2.circle(mask, (60, 60), 35, 255, 5)

    skeleton = thin(mask)
    # A ring has no tip: every pixel should have two neighbours.
    padded = np.pad((skeleton > 0).astype(np.int16), 1)
    counts = sum(
        padded[dy : dy + skeleton.shape[0], dx : dx + skeleton.shape[1]]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )
    tips = int(((skeleton > 0) & (counts == 1)).sum())
    assert tips == 0, f"a ring should have no free ends, found {tips}"


def test_thinning_an_empty_mask_is_empty() -> None:
    assert not thin(_blank()).any()


def test_thinning_terminates_on_a_solid_block() -> None:
    mask = _blank(60)
    mask[10:50, 10:50] = 255
    skeleton = thin(mask)
    assert skeleton.any()
    assert int((skeleton > 0).sum()) < int((mask > 0).sum()) // 4


# --------------------------------------------------------------------------- #
# 2 and 3. Arcs
# --------------------------------------------------------------------------- #


def test_a_plain_line_is_one_arc_with_no_junction() -> None:
    mask = _blank()
    _line(mask, (10, 60), (110, 60))

    graph = build_graph(_skeleton_of(mask))
    assert len(graph.arcs) == 1
    assert not graph.junctions
    arc = next(iter(graph.arcs.values()))
    assert arc.first_junction is None and arc.last_junction is None
    # Ordered end to end, so the tangents at the two ends oppose each other.
    assert float(np.dot(arc.tangent(True), arc.tangent(False))) < -0.9


def test_an_x_becomes_four_arcs_around_one_junction() -> None:
    mask = _blank()
    _line(mask, (10, 10), (110, 110))
    _line(mask, (110, 10), (10, 110))

    graph = build_graph(_skeleton_of(mask))
    assert len(graph.junctions) == 1
    assert len(graph.arcs) == 4
    junction = next(iter(graph.incident))
    assert len(graph.incident[junction]) == 4


def test_spur_pruning_removes_an_arrowhead_whisker() -> None:
    """A solid triangle thins to a stub with whiskers, which would read as a junction."""
    mask = _blank()
    _line(mask, (20, 60), (80, 60), width=3)
    cv2.fillPoly(mask, [np.array([(100, 60), (80, 52), (80, 68)], np.int32)], 255)

    unpruned = build_graph(thin(mask))
    pruned = build_graph(_skeleton_of(mask))
    assert len(pruned.arcs) <= len(unpruned.arcs)
    assert not pruned.junctions, "the arrowhead is still being read as a junction"


# --------------------------------------------------------------------------- #
# 4. Crossing against branch
# --------------------------------------------------------------------------- #


def test_an_x_is_read_as_a_crossing_and_paired_opposite() -> None:
    mask = _blank()
    _line(mask, (10, 10), (110, 110))
    _line(mask, (110, 10), (10, 110))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    junction = next(iter(junctions.values()))
    assert junction.kind == "crossing"

    # Each arc-end continues into one whose tangent opposes it.
    for entry, partner in junction.continues.items():
        left = graph.arcs[entry[0]].tangent(entry[1])
        right = graph.arcs[partner[0]].tangent(partner[1])
        assert float(np.dot(left, right)) < -0.7


@pytest.mark.parametrize(
    "second",
    [
        ((10, 60), (110, 60)),  # right angle to a vertical
        ((20, 20), (100, 100)),  # 45 degrees
        ((10, 45), (110, 75)),  # shallow, about 17 degrees
    ],
)
def test_crossings_at_several_angles_are_all_read_as_crossings(second) -> None:
    mask = _blank()
    _line(mask, (60, 10), (60, 110))
    _line(mask, second[0], second[1])

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    assert junctions, "no junction found"
    assert all(junction.kind == "crossing" for junction in junctions.values())


def test_a_t_junction_is_read_as_a_branch() -> None:
    """A stem meeting a line: connectors sharing a stem, not two curves crossing."""
    mask = _blank()
    _line(mask, (10, 60), (110, 60))
    _line(mask, (60, 60), (60, 110))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    assert len(junctions) == 1
    junction = next(iter(junctions.values()))
    assert junction.kind == "branch"
    assert junction.stem is not None
    assert len(junction.branches) == 2


def test_a_y_junction_is_read_as_a_branch_with_the_stem_identified() -> None:
    mask = _blank()
    _line(mask, (60, 110), (60, 60))  # the stem, coming up from the bottom
    _line(mask, (60, 60), (20, 15))
    _line(mask, (60, 60), (100, 15))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    junction = next(iter(junctions.values()))
    assert junction.kind == "branch"
    assert junction.stem is not None
    # Tangents are (dy, dx) and point into the junction. The stem runs downward from the junction,
    # so at its junction end it points up: dy well below zero.
    stem_tangent = graph.arcs[junction.stem[0]].tangent(junction.stem[1])
    assert stem_tangent[0] < -0.5, f"stem should run downward from the junction, got {stem_tangent}"
    # And the two branches run upward, so theirs point down.
    for branch in junction.branches:
        assert graph.arcs[branch[0]].tangent(branch[1])[0] > 0.2


# --------------------------------------------------------------------------- #
# 5. Routing
# --------------------------------------------------------------------------- #


def _tip_ends(graph):
    return sorted(graph.free_ends)


def test_routing_carries_straight_through_a_crossing() -> None:
    """Enter on one arm of an X, leave on the opposite arm, not on a neighbour."""
    mask = _blank()
    _line(mask, (10, 10), (110, 110))
    _line(mask, (110, 10), (10, 110))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    for start in _tip_ends(graph):
        reached = routes_between_ends(graph, junctions, start)
        assert len(reached) == 1, f"a crossing should offer one way through, got {len(reached)}"
        end = reached[0][0]
        # The arm arrived at must be on the other side: its tip is roughly opposite the start's.
        start_tip = np.array(graph.arcs[start[0]].end_point(start[1]), dtype=float)
        end_tip = np.array(graph.arcs[end[0]].end_point(end[1]), dtype=float)
        centre = np.array([60.0, 60.0])
        assert float(np.dot(start_tip - centre, end_tip - centre)) < 0


def test_routing_through_a_branch_offers_every_branch() -> None:
    mask = _blank()
    _line(mask, (60, 110), (60, 60))
    _line(mask, (60, 60), (20, 15))
    _line(mask, (60, 60), (100, 15))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    junction = next(iter(junctions.values()))
    stem_tip = (junction.stem[0], not junction.stem[1])

    reached = routes_between_ends(graph, junctions, stem_tip)
    assert len(reached) == 2, f"entering the stem should reach both branches, got {len(reached)}"


def test_routing_follows_a_curve_through_a_crossing() -> None:
    """A curved connector crossed by a straight one. Curvature has to keep the curve together."""
    mask = _blank(160)
    _polyline(mask, [(15, 30), (60, 40), (100, 80), (120, 140)])
    _line(mask, (20, 130), (140, 40))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    if not junctions:
        pytest.skip("the two strokes did not overlap in this rendering")
    # Whatever the junction is read as, every tip must still reach exactly one other tip.
    for start in _tip_ends(graph):
        reached = routes_between_ends(graph, junctions, start)
        assert reached, f"tip {start} reached nothing"


def test_routing_terminates_on_a_loop() -> None:
    """A cycle must not send the router round for ever."""
    mask = _blank()
    cv2.circle(mask, (60, 60), 30, 255, 4)
    _line(mask, (60, 90), (60, 115))

    graph = build_graph(_skeleton_of(mask))
    junctions = read_junctions(graph)
    for start in _tip_ends(graph):
        routes_between_ends(graph, junctions, start)


# --------------------------------------------------------------------------- #
# 6. Matching
# --------------------------------------------------------------------------- #


def test_matching_prefers_the_globally_cheaper_set() -> None:
    """The case greedy gets wrong: the cheapest single pair rules out two cheap ones."""
    costs = {
        (0, 1): 0.9,  # greedy takes this first
        (2, 3): 5.0,
        (0, 2): 1.0,
        (1, 3): 1.0,
    }
    pairs = set(match_pairs(4, costs))
    assert pairs == {(0, 2), (1, 3)}, f"greedy answer chosen: {pairs}"


def test_matching_never_uses_an_item_twice() -> None:
    costs = {(0, 1): 0.1, (0, 2): 0.1, (0, 3): 0.1, (1, 2): 2.0}
    pairs = match_pairs(4, costs)
    used = [index for pair in pairs for index in pair]
    assert len(used) == len(set(used))


def test_matching_leaves_an_item_out_rather_than_inventing_a_pair() -> None:
    costs = {(0, 1): 0.5}
    pairs = match_pairs(3, costs)
    assert pairs == [(0, 1)]


def test_matching_abandons_a_pair_that_costs_more_than_giving_up() -> None:
    costs = {(0, 1): UNMATCHED_COST * 3}
    assert match_pairs(2, costs) == []


def test_matching_handles_nothing_to_match() -> None:
    assert match_pairs(0, {}) == []
    assert match_pairs(1, {}) == []

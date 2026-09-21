"""Regression tests for the geometric diagram parser.

Every case here is a drawing built in the test, a few hundred pixels across, with the answer known
by construction. No corpus image is committed: FlowLearn is CC BY-NC-SA 4.0 and lives under
``data/external``, so a test that needed one of its files could not run on a fresh clone and would
be distributing the corpus besides.

The cases are the situations that actually broke the parser while it was being built, so each one
is a guard against a specific regression rather than a general gesture at coverage:

* **An arrowhead touching its target's border.** A filled arrowhead is about 11 px across, wider
  than the 7 px opening kernel, so it survives the opening and fuses into the node's component.
  The bounding box then grows by the length of the arrow, the stroke layer deletes the arrowhead
  along with the node, and the head end ends up with *less* ink than the tail - so the direction
  inverts. One dev chart scored 0% on direction because of this.
* **A dashed connector.** Drawn as separate dashes, which are separate components until bridged.
* **Two curves that cross but are unrelated.** Their pixels touch, so they are one component, and
  reading it as one connector loses an edge and invents another.
* **Two curves that meet at a node.** A real fan-out, which must not be confused with a crossing.
* **A reciprocal pair.** 36% of dev charts draw one arrow each way between the same two nodes.
* **A cycle with no start and no end node.** Legal. 46% of dev charts contain a cycle, and every
  chart with no zero-in-degree node has one, so refusing them would refuse the real thing.
* **Duplicate connectors between one pair of nodes**, which must not collapse into one edge.

Topology is parsed with ``read_text=False`` throughout, so these tests do not need tesseract
installed and do not measure it. Text is injected where a test needs a label.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from agentpath_ingestion.corpora.diagram import (
    DiagramParse,
    _refine_box,
    bridge_dashes,
    detect_nodes,
    modal_background,
    parse_diagram,
    validate,
)
from agentpath_ingestion.corpora.diagram_graph import graph_from_parse, review_reasons
from agentpath_ingestion.corpora.workflow import ExtractionStatus

# --------------------------------------------------------------------------- #
# Drawing. Colours and sizes follow the corpus: a mid-grey page, near-white node
# fills, dark strokes three pixels wide, and nodes of uniform height.
# --------------------------------------------------------------------------- #

BACKGROUND = (192, 192, 192)
NODE_FILL = (245, 245, 250)
NODE_BORDER = (120, 120, 140)
STROKE = (40, 40, 40)

NODE_WIDTH = 104
NODE_HEIGHT = 34
STROKE_WIDTH = 3
HEAD_LENGTH = 11
HEAD_HALF_WIDTH = 6


def _canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), BACKGROUND, np.uint8)


def _node(canvas: np.ndarray, x: int, y: int, *, width: int = NODE_WIDTH, height: int = NODE_HEIGHT) -> tuple[int, ...]:
    """A filled box with a border, like Mermaid's rounded rectangle. Returns its rectangle."""
    cv2.rectangle(canvas, (x, y), (x + width, y + height), NODE_FILL, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), NODE_BORDER, 1)
    return (x, y, width, height)


def _unit(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = float(np.hypot(dx, dy)) or 1.0
    return (dx / length, dy / length)


def _arrowhead(canvas: np.ndarray, tip: tuple[int, int], approach: tuple[float, float]) -> None:
    """A solid triangle with its point at *tip*, arriving along *approach*."""
    dx, dy = approach
    base = (tip[0] - dx * HEAD_LENGTH, tip[1] - dy * HEAD_LENGTH)
    normal = (-dy, dx)
    left = (base[0] + normal[0] * HEAD_HALF_WIDTH, base[1] + normal[1] * HEAD_HALF_WIDTH)
    right = (base[0] - normal[0] * HEAD_HALF_WIDTH, base[1] - normal[1] * HEAD_HALF_WIDTH)
    cv2.fillPoly(canvas, [np.array([tip, left, right], np.int32)], STROKE)


def _connector(canvas: np.ndarray, points: list[tuple[int, int]], *, dashed: bool = False, head: bool = True) -> None:
    """A polyline from the first point to the last, with an arrowhead at the last by default."""
    path = [(int(x), int(y)) for x, y in points]
    if dashed:
        # Walk the path laying down 5 px dashes with 5 px gaps, which is what the corpus draws.
        dash, gap, carried = 5, 5, 0.0
        for first, second in pairwise(path):
            direction = _unit(first, second)
            length = float(np.hypot(second[0] - first[0], second[1] - first[1]))
            position = carried
            while position < length:
                end = min(position + dash, length)
                cv2.line(
                    canvas,
                    (round(first[0] + direction[0] * position), round(first[1] + direction[1] * position)),
                    (round(first[0] + direction[0] * end), round(first[1] + direction[1] * end)),
                    STROKE,
                    STROKE_WIDTH,
                    cv2.LINE_AA,
                )
                position = end + gap
            carried = max(position - length, 0.0)
    else:
        cv2.polylines(canvas, [np.array(path, np.int32)], False, STROKE, STROKE_WIDTH, cv2.LINE_AA)
    if head:
        _arrowhead(canvas, path[-1], _unit(path[-2], path[-1]))


def _write(tmp_path: Path, canvas: np.ndarray, name: str = "chart.png") -> Path:
    target = tmp_path / name
    cv2.imwrite(str(target), canvas)
    return target


# --------------------------------------------------------------------------- #
# Reading the result back in the terms the drawing was written in
# --------------------------------------------------------------------------- #


def _name_nodes(parse: DiagramParse, drawn: dict[str, tuple[int, ...]]) -> dict[str, str]:
    """Map each drawn name to the detected node id whose centre is nearest to it.

    Node ids depend on layout order, so asserting on ``n0`` would make a test fail when a drawing
    is nudged. Asserting on the name the test drew keeps the failure message about the behaviour.
    """
    mapping: dict[str, str] = {}
    for name, (x, y, width, height) in drawn.items():
        target = (x + width / 2, y + height / 2)
        best = min(
            parse.nodes,
            key=lambda node: (node.centre[0] - target[0]) ** 2 + (node.centre[1] - target[1]) ** 2,
        )
        mapping[name] = best.node_id
    return mapping


def _directed(parse: DiagramParse, drawn: dict[str, tuple[int, ...]]) -> list[tuple[str, str]]:
    """Established edges as pairs of drawn names."""
    names = {node_id: name for name, node_id in _name_nodes(parse, drawn).items()}
    return sorted(
        (names.get(edge.source_id or "", "?"), names.get(edge.target_id or "", "?")) for edge in parse.directed_edges
    )


def _connected(parse: DiagramParse, drawn: dict[str, tuple[int, ...]]) -> list[tuple[str, str]]:
    """Edges with both ends established, as unordered pairs of drawn names."""
    names = {node_id: name for name, node_id in _name_nodes(parse, drawn).items()}
    return sorted(
        tuple(sorted((names.get(edge.source_id or "", "?"), names.get(edge.target_id or "", "?"))))
        for edge in parse.connected_edges
    )


def _with_text(parse: DiagramParse) -> DiagramParse:
    """Give every node a distinct label without running OCR."""
    parse.nodes = [replace(node, text=f"box {index}", text_confidence=0.9) for index, node in enumerate(parse.nodes)]
    return parse


# --------------------------------------------------------------------------- #
# The node box must not swallow the arrowhead that touches it
# --------------------------------------------------------------------------- #


def test_arrowhead_touching_a_node_does_not_inflate_its_box(tmp_path: Path) -> None:
    """The regression that cost the most: a fused arrowhead stretching the node's rectangle.

    The arrow's tip is drawn exactly on the target's top border, which is what Mermaid does. The
    detected box must still be the box, within a pixel or two of what was drawn.
    """
    canvas = _canvas(260, 220)
    top = _node(canvas, 70, 30)
    bottom = _node(canvas, 70, 150)
    # Straight down, tip landing on the lower box's top edge.
    _connector(canvas, [(122, top[1] + top[3]), (122, bottom[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    assert len(parse.nodes) == 2

    names = _name_nodes(parse, {"top": top, "bottom": bottom})
    detected = {node.node_id: node for node in parse.nodes}
    target = detected[names["bottom"]]
    # The arrowhead sits directly above this box. Before the box refinement its top edge moved up
    # by the length of the arrow, which is what deleted the arrowhead from the stroke layer.
    assert abs(target.y - bottom[1]) <= 3, f"box top moved to {target.y}, drawn at {bottom[1]}"
    assert abs(target.height - bottom[3]) <= 4


def test_direction_survives_node_carving(tmp_path: Path) -> None:
    """The head end must keep more ink than the tail end after the node regions are removed."""
    canvas = _canvas(260, 220)
    top = _node(canvas, 70, 30)
    bottom = _node(canvas, 70, 150)
    _connector(canvas, [(122, top[1] + top[3]), (122, bottom[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"top": top, "bottom": bottom}
    assert _directed(parse, drawn) == [("top", "bottom")]

    edge = parse.directed_edges[0]
    assert edge.target_ink > edge.source_ink
    assert edge.direction_ratio > 1.15


def test_arrowhead_overlapping_the_border_still_points_inward(tmp_path: Path) -> None:
    """Some renderings draw the tip a few pixels *inside* the target. Direction must not invert."""
    canvas = _canvas(260, 220)
    top = _node(canvas, 70, 30)
    bottom = _node(canvas, 70, 150)
    _connector(canvas, [(122, top[1] + top[3]), (122, bottom[1] + 4)])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    assert _directed(parse, {"top": top, "bottom": bottom}) == [("top", "bottom")]


# --------------------------------------------------------------------------- #
# Dashes, curves
# --------------------------------------------------------------------------- #


def test_dashed_connector_is_one_edge(tmp_path: Path) -> None:
    canvas = _canvas(260, 220)
    top = _node(canvas, 70, 30)
    bottom = _node(canvas, 70, 150)
    _connector(canvas, [(122, top[1] + top[3]), (122, bottom[1])], dashed=True)

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    assert _directed(parse, {"top": top, "bottom": bottom}) == [("top", "bottom")]


def test_bridging_joins_a_dashed_line_into_one_component() -> None:
    """The unit behind the dashed case, without the rest of the pipeline in the way."""
    mask = np.zeros((60, 200), np.uint8)
    for start in range(10, 185, 14):
        cv2.line(mask, (start, 30), (start + 5, 30), 255, 3)

    before = cv2.connectedComponents(mask)[0] - 1
    after = cv2.connectedComponents(bridge_dashes(mask))[0] - 1
    assert before > 5, "the drawing should start as many separate dashes"
    assert after == 1, f"bridging left {after} components instead of one"


def test_curved_connector_keeps_its_direction(tmp_path: Path) -> None:
    """A bezier-like route, which is what the corpus draws. Straight-line reasoning fails here."""
    canvas = _canvas(340, 260)
    left = _node(canvas, 20, 30)
    right = _node(canvas, 200, 190)
    _connector(
        canvas,
        [(72, left[1] + left[3]), (76, 110), (110, 150), (180, 160), (240, 172), (252, right[1])],
    )

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    assert _directed(parse, {"left": left, "right": right}) == [("left", "right")]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect. A connector arriving almost parallel to its target's border puts the head "
        "end's window half inside the node, where the strokes have been removed, while the tail "
        "end's window holds a full radius of shaft. The tail then measures more ink than the head "
        "and the direction inverts. Rare in the corpus - arrows there meet a border steeply, and "
        "direction is right on 97.7% of established dev edges - so it is recorded rather than "
        "papered over. The fix is to measure ink per unit of curve followed instead of per window."
    ),
)
def test_a_connector_arriving_almost_parallel_to_the_border(tmp_path: Path) -> None:
    canvas = _canvas(340, 240)
    left = _node(canvas, 20, 30)
    right = _node(canvas, 200, 170)
    # Last segment at about 14 degrees to the target's top edge.
    _connector(canvas, [(72, left[1] + left[3]), (80, 110), (140, 150), (210, 160), (250, right[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    assert _directed(parse, {"left": left, "right": right}) == [("left", "right")]


# --------------------------------------------------------------------------- #
# Crossings, which is where the parser is weakest and so where it is pinned down
# --------------------------------------------------------------------------- #


def test_two_crossing_connectors_stay_two_edges(tmp_path: Path) -> None:
    """Unrelated curves whose pixels cross. Reading the blob as one connector loses an edge."""
    canvas = _canvas(420, 300)
    top_left = _node(canvas, 20, 20)
    top_right = _node(canvas, 280, 20)
    bottom_left = _node(canvas, 20, 230)
    bottom_right = _node(canvas, 280, 230)
    # Two straight diagonals through the same middle point.
    _connector(canvas, [(72, top_left[1] + top_left[3]), (332, bottom_right[1])])
    _connector(canvas, [(332, top_right[1] + top_right[3]), (72, bottom_left[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"tl": top_left, "tr": top_right, "bl": bottom_left, "br": bottom_right}
    assert len(parse.nodes) == 4
    assert _connected(parse, drawn) == [("bl", "tr"), ("br", "tl")]


def test_two_connectors_meeting_at_one_node_is_not_a_crossing(tmp_path: Path) -> None:
    """A fan-out. Both edges leave the same node, which makes it a decision point.

    They leave from two points on the border rather than from one pixel, which is how the corpus
    draws a fan; the degenerate shared-pixel case is covered separately below.
    """
    canvas = _canvas(420, 300)
    source = _node(canvas, 150, 20)
    left = _node(canvas, 20, 230)
    right = _node(canvas, 280, 230)
    _connector(canvas, [(180, source[1] + source[3]), (140, 150), (72, left[1])])
    _connector(canvas, [(226, source[1] + source[3]), (264, 150), (332, right[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"source": source, "left": left, "right": right}
    assert _directed(parse, drawn) == [("source", "left"), ("source", "right")]

    names = _name_nodes(parse, drawn)
    assert parse.branch_node_ids == [names["source"]]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect. Two connectors leaving a node from the *same pixel* share one attachment, so "
        "the parser sees three attachments for two connectors and the shared end carries both "
        "stubs' worth of ink. Restricting the count to a corridor about each connector's own axis "
        "fixed the inversion for a Y whose arms diverge widely, but not for arms that leave "
        "together. The corpus fans out from separate points on the border, so this is recorded "
        "rather than fixed. The fix is to split a shared attachment by the traced paths through it."
    ),
)
def test_two_connectors_leaving_a_node_from_the_same_pixel(tmp_path: Path) -> None:
    canvas = _canvas(420, 300)
    source = _node(canvas, 150, 20)
    left = _node(canvas, 20, 230)
    right = _node(canvas, 280, 230)
    _connector(canvas, [(202, source[1] + source[3]), (140, 150), (72, left[1])])
    _connector(canvas, [(202, source[1] + source[3]), (264, 150), (332, right[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"source": source, "left": left, "right": right}
    assert _directed(parse, drawn) == [("source", "left"), ("source", "right")]


def test_several_connectors_arriving_at_one_node(tmp_path: Path) -> None:
    """Two arrowheads landing side by side on the same border, which can merge into one blob."""
    canvas = _canvas(420, 300)
    left = _node(canvas, 20, 20)
    right = _node(canvas, 280, 20)
    target = _node(canvas, 150, 230)
    _connector(canvas, [(72, left[1] + left[3]), (120, 150), (190, target[1])])
    _connector(canvas, [(332, right[1] + right[3]), (284, 150), (214, target[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"left": left, "right": right, "target": target}
    assert _directed(parse, drawn) == [("left", "target"), ("right", "target")]


# --------------------------------------------------------------------------- #
# Two edges between one pair of nodes
# --------------------------------------------------------------------------- #


def test_reciprocal_edges_are_both_reported(tmp_path: Path) -> None:
    """One arrow each way. 36% of dev charts draw this, and collapsing it loses a real branch."""
    canvas = _canvas(300, 260)
    top = _node(canvas, 90, 30)
    bottom = _node(canvas, 90, 180)
    # Down the left side, up the right side, well apart so they are two separate curves.
    _connector(canvas, [(110, top[1] + top[3]), (110, bottom[1])])
    _connector(canvas, [(174, bottom[1]), (174, top[1] + top[3])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"top": top, "bottom": bottom}
    assert _directed(parse, drawn) == [("bottom", "top"), ("top", "bottom")]


def test_duplicate_connectors_between_the_same_pair(tmp_path: Path) -> None:
    """Two arrows the same way round between one pair. Two drawn, two reported."""
    canvas = _canvas(300, 260)
    top = _node(canvas, 90, 30)
    bottom = _node(canvas, 90, 180)
    _connector(canvas, [(110, top[1] + top[3]), (110, bottom[1])])
    _connector(canvas, [(174, top[1] + top[3]), (174, bottom[1])])

    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    drawn = {"top": top, "bottom": bottom}
    assert _connected(parse, drawn) == [("bottom", "top"), ("bottom", "top")]


# --------------------------------------------------------------------------- #
# Cycles are legal
# --------------------------------------------------------------------------- #


def test_a_cycle_is_parsed_and_not_treated_as_a_fault(tmp_path: Path) -> None:
    """A -> B -> C -> A. No node has a zero in-degree, and that is not an error."""
    canvas = _canvas(420, 380)
    a = _node(canvas, 150, 20)
    b = _node(canvas, 280, 200)
    c = _node(canvas, 20, 200)
    _connector(canvas, [(240, a[1] + a[3]), (300, b[1])])
    _connector(canvas, [(300, b[1] + b[3]), (300, 320), (110, 320), (72, c[1] + c[3])])
    # Back up to A, landing on its left border rather than inside it: an arrowhead drawn wholly
    # inside the target is removed with the node and leaves no direction to measure.
    _connector(canvas, [(72, c[1]), (100, 60), (a[0], a[1] + 17)])

    parse = _with_text(parse_diagram(_write(tmp_path, canvas), read_text=False))
    drawn = {"a": a, "b": b, "c": c}
    assert len(parse.nodes) == 3
    assert _connected(parse, drawn) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert parse.has_cycle is True

    # Nothing in the validator may object to a cycle as such.
    assert not [problem for problem in validate(parse) if "cycle" in problem or "loop" in problem]
    assert not [reason for reason in review_reasons(parse) if "start" in reason or "end" in reason]


def test_a_cycle_is_accepted_by_the_gate(tmp_path: Path) -> None:
    """ "No start node" is what a loop looks like, and must not send a good parse to review."""
    canvas = _canvas(300, 300)
    a = _node(canvas, 90, 30)
    b = _node(canvas, 90, 190)
    _connector(canvas, [(110, a[1] + a[3]), (110, b[1])])
    _connector(canvas, [(174, b[1]), (174, a[1] + a[3])])

    parse = _with_text(parse_diagram(_write(tmp_path, canvas), read_text=False))
    graph = graph_from_parse(parse, image_id="c", document_id="c", corpus_id="test")
    assert graph.status is ExtractionStatus.OK, f"refused a legal loop: {graph.review_reasons}"


# --------------------------------------------------------------------------- #
# The gate refuses what it should
# --------------------------------------------------------------------------- #


def test_a_node_with_no_connector_sends_the_parse_to_review(tmp_path: Path) -> None:
    """No dev chart has an isolated node, so one here means an arrow was missed."""
    canvas = _canvas(300, 380)
    a = _node(canvas, 90, 30)
    b = _node(canvas, 90, 160)
    lonely = _node(canvas, 90, 300)
    _connector(canvas, [(142, a[1] + a[3]), (142, b[1])])

    parse = _with_text(parse_diagram(_write(tmp_path, canvas), read_text=False))
    drawn = {"a": a, "b": b, "lonely": lonely}
    names = _name_nodes(parse, drawn)

    reasons = review_reasons(parse)
    assert any("no connector at all" in reason for reason in reasons), reasons
    assert names["lonely"] in " ".join(reasons)

    graph = graph_from_parse(parse, image_id="c", document_id="c", corpus_id="test")
    assert graph.status is ExtractionStatus.NEEDS_REVIEW


def test_unreadable_node_text_sends_the_parse_to_review(tmp_path: Path) -> None:
    canvas = _canvas(260, 220)
    _node(canvas, 70, 30)
    _node(canvas, 70, 150)
    _connector(canvas, [(122, 64), (122, 150)])

    # Text left empty, as it is when OCR reads nothing.
    parse = parse_diagram(_write(tmp_path, canvas), read_text=False)
    graph = graph_from_parse(parse, image_id="c", document_id="c", corpus_id="test")
    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("no readable text" in reason for reason in graph.review_reasons)


# --------------------------------------------------------------------------- #
# Box refinement, on its own
# --------------------------------------------------------------------------- #


def test_refine_box_trims_a_spike_off_a_rectangle() -> None:
    """What a fused arrowhead looks like to the refinement: a few narrow rows above the box."""
    blob = np.zeros((60, 100), np.uint8)
    blob[20:56, 5:95] = 255  # the node
    blob[8:20, 40:52] = 255  # an arrowhead stuck to its top edge

    x, y, width, height = _refine_box(blob)
    assert y == pytest.approx(20, abs=2), f"kept {height} rows starting at {y}"
    assert height == pytest.approx(36, abs=3)
    assert x == pytest.approx(5, abs=2)
    assert width == pytest.approx(90, abs=3)


def test_refine_box_leaves_a_plain_rectangle_alone() -> None:
    blob = np.zeros((50, 80), np.uint8)
    blob[10:40, 10:70] = 255

    x, y, width, height = _refine_box(blob)
    assert (x, y) == pytest.approx((10, 10), abs=2)
    assert (width, height) == pytest.approx((60, 30), abs=3)


def test_refine_box_keeps_the_original_when_trimming_would_take_most_of_it() -> None:
    """A blob that is not a node with something attached must be left for the other filters."""
    blob = np.zeros((60, 60), np.uint8)
    cv2.circle(blob, (30, 30), 22, 255, -1)

    _x, _y, width, height = _refine_box(blob)
    assert width >= 30 and height >= 30


# --------------------------------------------------------------------------- #
# Background and thresholds
# --------------------------------------------------------------------------- #


def test_a_dark_background_is_found_and_not_treated_as_ink(tmp_path: Path) -> None:
    """Nine background colours appear across 16 dev images, including near-black.

    Assuming ink is dark marked 87-90% of a dark-background chart as ink.
    """
    canvas = np.full((220, 260, 3), (14, 6, 36), np.uint8)
    cv2.rectangle(canvas, (70, 30), (174, 64), (230, 230, 245), -1)
    cv2.rectangle(canvas, (70, 150), (174, 184), (230, 230, 245), -1)
    _connector(canvas, [(122, 64), (122, 150)])

    path = _write(tmp_path, canvas)
    image = cv2.imread(str(path))
    background = modal_background(image)
    assert tuple(int(value) for value in background) == (14, 6, 36)

    nodes, _threshold, _opened = detect_nodes(image, background)
    assert len(nodes) == 2

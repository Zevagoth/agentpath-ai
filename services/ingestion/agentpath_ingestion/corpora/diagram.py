"""Recover a flowchart's topology from its pixels, not from a language model.

Stage 2B1 measured what happens when a vision-language model is asked for a whole graph:
`qwen3.5:4b` read the boxes well (89.9% node F1) and invented the arrows, scoring 65.0% on
directed edges and 56.7% on path reachability - a coin flip. Scaling to 9B moved path
reachability by 2.5 points. The problem is not model size, it is that a language model has no
mechanism for *not* producing an answer about a line it cannot see.

So the topology is measured here instead, and the model is never allowed to add an edge. What
the model is for - reading a node's meaning, interpreting a ``Yes``/``No`` label, matching a
case description to a branch - happens elsewhere, on top of this.

The pipeline, and the evidence each stage rests on
-------------------------------------------------
Every threshold below was chosen by measuring on the FlowLearn **dev** split. None was chosen
on held-out.

1. **Background.** The modal colour. These renderings use a uniform background, and it is not
   always light: measured across 16 dev images there were 9 distinct background colours,
   including white (255,255,255) and near-black (14,6,36). So nothing here may assume "ink is
   dark" - that assumption marked 87-90% of a dark-background image as ink.

2. **Foreground at several thresholds.** ``|pixel - background|`` above a threshold. A single
   threshold fails: at 40 it finds nothing on a white-background chart whose node fill is also
   near-white. The sweep tries 40, 24, 14, 8 and picks the first that yields a *self-consistent*
   set of boxes, scored without ground truth by how many box heights agree with the median -
   Mermaid draws nodes at a near-uniform height. Measured: exact node count on **39 of 40** dev
   images, against 11 of 14 for a single threshold.

3. **Nodes by morphological opening.** A node fill is thick; a connector is a few pixels wide.
   An opening with a 7-pixel ellipse destroys the connectors and leaves the node interiors,
   with no reference to colour. Kernel 7 was measured against 5, 9 and 11: exact on 7 of 8
   where 5 got 4 and 11 got 3.

   Border-contour detection was tried as an alternative and **rejected**: it scored 3 of 14,
   because glyphs and arrowheads are also closed loops that fill their bounding box.

4. **Connectors get their own threshold.** Node contrast and connector contrast are not the
   same thing. On ``8267.jpeg`` the connectors are dark brown on dark navy and the node
   threshold catches 0.13% of the page, which produced *zero* edges. The stroke threshold is
   therefore chosen separately, by the number of connectors it makes resolvable - a criterion
   that needs no ground truth because it counts components joining exactly two nodes.

5. **Dashes are bridged along the curve, not with a square kernel.** Many of these charts draw
   connectors dotted. Measured on 40 dev images, square closing left 108 components reaching
   only one node and 282 reaching none, against 253 real edges, and growing the square to 13x13
   barely helped: a square has to be as wide as the gap in every direction at once, so it
   merges neighbouring curves before it bridges anything. Closing with a thin **line** kernel at
   12 orientations and unioning the results cut those to 6 and 5 - the fragmentation problem
   essentially disappeared.

6. **A connector is a component; a bundle is traced.** After bridging, a component touching
   exactly two nodes is one edge and needs no interpretation. Crossings and fans leave
   components touching three or more, which held roughly 70 of the 276 real edges, so they are
   split by *following* each curve: step along the mask preferring to carry straight on, and
   stop at whichever node is reached. Carrying straight on is precisely what tells two crossing
   curves apart, which is why this is a tracer rather than another morphological trick.
   Measured on 40 dev images: 207 edges from components plus 45 from tracing, against 253 real
   edges, with 24 of 37 charts exact on count.

7. **Direction comes from ink, not from thickness.** The arrowhead end of a connector carries
   more ink than the tail end. Comparing *maximum thickness* was tried first and tied on 78 of
   about 180 edges, because the distance transform is coarse and a thin barb is barely wider
   than its own shaft - and arrowhead size is not even comparable between charts: ``8035.jpeg``
   draws solid triangles of radius 4.6 against a line radius of 1.9, while ``8113.jpeg`` draws
   barbs of radius 2.3 against the same 1.9. Comparing the *amount of ink* in a disc at each end
   of the same connector is a relative test within one curve, so chart-to-chart variation
   cancels. Ties fell from 78 to 9 and direction accuracy rose from about 77% to 87.7%.

8. **Ambiguity is recorded, never resolved by guessing.** Endpoints that could not be
   established and directions that could not be established are separate facts, tracked
   separately, and both are drawn on the overlay. A consumer may use an edge's *adjacency*
   while refusing to trust its *direction*.

Loops are legal. Nothing here assumes a DAG, and a cycle is never a reason to reject a graph:
"no? go back and retry" is a real branch in a real troubleshooting chart.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Constants, all measured on dev
# --------------------------------------------------------------------------- #

#: Tried in this order for *nodes*. The first producing a self-consistent box set wins. 40
#: handles the common coloured backgrounds; 14 and 8 exist for white-on-white, where the node
#: fill differs from the background by under 20 levels.
FOREGROUND_THRESHOLDS = (40, 24, 14, 8)

#: Tried in this order for *connectors*, which can be much fainter than the nodes they join.
STROKE_THRESHOLDS = (40, 30, 24, 18, 14, 10, 8, 6)

#: A stroke layer covering more than this fraction of the page outside the nodes is not a set of
#: connectors, it is a flooded mask. Guards the low end of the sweep.
STROKE_SHARE_CAP = 0.05

#: Opening kernel that separates node fills from connectors. See the module docstring for the
#: comparison against 5, 9 and 11.
NODE_OPEN_KERNEL = 7

#: A node must be at least this many pixels in its smaller dimension and this large in area.
#: Below it, a component is a glyph or an arrowhead.
MIN_NODE_SIDE = 8
MIN_NODE_AREA = 200

#: A node fills most of its bounding rectangle. A curve does not.
MIN_NODE_FILL_RATIO = 0.5

#: Heights within this fraction of the median count as agreeing, for the sweep's
#: consistency score.
NODE_HEIGHT_TOLERANCE = 0.25

#: Node bounding boxes are grown by this much before being removed from the stroke layer. One
#: pixel, not three: the earlier value of 3 cut the fat part off any arrowhead whose tip touches
#: its target's border, which collapsed measured head thickness to line width on several charts.
#: Removing the node *shape* instead was tried and rejected - the opened mask under-covers the
#: node by a couple of pixels, so its own rim survived and became 84% of the stroke layer on
#: ``8267.jpeg``.
NODE_BLOCK_PAD = 1

#: A connector counts as attached to a node if it comes within this distance of the node's block.
#: Not zero, because a dotted connector's first dash can start short of the border.
ATTACH_GAP = 6

#: Line kernel length and orientation count for dash bridging. 9 was measured against 15, 21 and
#: 27: all of them fix fragmentation, and the longer ones then merge separate curves into
#: bundles - resolvable components fell from 206 at length 9 to 133 at length 27.
BRIDGE_LENGTH = 9
BRIDGE_ANGLES = 12

#: Components smaller than this are JPEG speckle, not connectors.
MIN_CONNECTOR_AREA = 8

#: Tracer geometry. The step is the radius of the ring of candidate next positions; the turn
#: limit is what makes the tracer carry straight on through a crossing rather than turn onto the
#: other curve.
TRACE_STEP = 3
TRACE_MAX_TURN = 55.0
TRACE_MAX_STEPS = 400
#: How much of the previous heading is kept when the tracer turns. Smoothing, so one noisy step
#: on a compressed edge cannot throw the trace off the curve.
TRACE_HEADING_INERTIA = 0.6

#: Radius of the disc in which ink is counted at each end of a connector, as a fraction of the
#: median node height. Wide enough to contain an arrowhead, narrow enough not to reach the
#: middle of a short connector.
DIRECTION_RADIUS_RATIO = 0.45

#: Ink ratio between the two ends below which direction is not considered established. 1.0 means
#: "any difference at all decides"; measured on dev, the ratio was 1.89 at the median when the
#: decision was right and 1.06 when it was wrong, so a margin above 1.0 trades recall for
#: precision. Kept at 1.15 so near-ties are reported as undirected rather than guessed.
MIN_DIRECTION_RATIO = 1.15


class NodeShape(StrEnum):
    """What the box looks like. Mermaid uses shape to carry meaning."""

    RECTANGLE = "rectangle"
    ROUNDED = "rounded"
    DIAMOND = "diamond"
    ELLIPSE = "ellipse"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DetectedNode:
    """One box, as found in the image."""

    node_id: str
    x: int
    y: int
    width: int
    height: int
    shape: NodeShape
    #: Fraction of the bounding rectangle the blob fills. A diagnostic, carried into the overlay.
    fill_ratio: float
    #: Read by OCR later; empty until then.
    text: str = ""
    text_confidence: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, px: int, py: int, *, margin: int = 0) -> bool:
        return (
            self.x - margin <= px <= self.x + self.width + margin
            and self.y - margin <= py <= self.y + self.height + margin
        )


@dataclass(frozen=True, slots=True)
class DetectedEdge:
    """One connector, with the evidence that produced it.

    Two different things can fail, so they are two different fields. ``ambiguous`` means the
    *endpoints* are not established - the connector was found but not what it joins. ``directed``
    means which end carries the arrowhead is established. An edge can be unambiguous and
    undirected, and such an edge is still useful: it says these two nodes are connected, which is
    a real fact about the chart, while refusing to claim a direction nobody could measure.
    """

    edge_id: str
    source_id: str | None
    target_id: str | None
    #: Where the connector meets the source and the target. Drawn on the overlay.
    source_point: tuple[int, int] | None = None
    target_point: tuple[int, int] | None = None
    #: ``"component"`` when the connector was its own component, ``"trace"`` when it was followed
    #: out of a bundle of crossing curves. A reviewer should look harder at traced edges.
    resolved_by: str = "component"
    #: Ink counted in a disc at each end: the direction evidence, in the units it was measured in.
    source_ink: int = 0
    target_ink: int = 0
    #: Whether the arrowhead end was established, and by what margin.
    directed: bool = False
    direction_ratio: float = 1.0
    ambiguous: bool = False
    reason: str = ""
    #: Text found beside the connector, e.g. "Yes". Filled by the OCR stage.
    label: str = ""
    confidence: float = 0.0

    @property
    def pair(self) -> tuple[str, str] | None:
        """``(source, target)`` when both ends are known, else ``None``."""
        if self.source_id and self.target_id:
            return (self.source_id, self.target_id)
        return None


@dataclass
class DiagramParse:
    """Everything recovered from one image, plus what could not be."""

    image_path: str
    width: int
    height: int
    background: tuple[int, int, int]
    threshold_used: int
    stroke_threshold: int = 0
    nodes: list[DetectedNode] = field(default_factory=list)
    edges: list[DetectedEdge] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: Kept for the overlay and for tests; not serialised.
    masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def connected_edges(self) -> list[DetectedEdge]:
        """Edges whose two endpoints are established, whatever their direction."""
        return [edge for edge in self.edges if not edge.ambiguous and edge.source_id and edge.target_id]

    @property
    def directed_edges(self) -> list[DetectedEdge]:
        """Edges safe to route with: endpoints *and* direction established."""
        return [edge for edge in self.connected_edges if edge.directed]

    @property
    def undirected_only_edges(self) -> list[DetectedEdge]:
        """Adjacency is known, direction is not. Real information, and not routable."""
        return [edge for edge in self.connected_edges if not edge.directed]

    @property
    def ambiguous_edges(self) -> list[DetectedEdge]:
        return [edge for edge in self.edges if edge.ambiguous or not (edge.source_id and edge.target_id)]

    def _as_labels(self, edges: list[DetectedEdge]) -> list[tuple[str, str]]:
        by_id = {node.node_id: node.text for node in self.nodes}
        return [(by_id.get(edge.source_id or "", ""), by_id.get(edge.target_id or "", "")) for edge in edges]

    @property
    def label_edges(self) -> list[tuple[str, str]]:
        """Directed edges as ``(source text, target text)`` pairs."""
        return self._as_labels(self.directed_edges)

    @property
    def connected_label_edges(self) -> list[tuple[str, str]]:
        """Every edge with both ends established, as label pairs.

        The order is still head-last where ink decided it, but for an edge whose direction was not
        established the order carries no claim and only the adjacency should be read from it.
        """
        return self._as_labels(self.connected_edges)

    @property
    def branch_node_ids(self) -> list[str]:
        """Nodes with more than one established outgoing edge. The decision points."""
        outgoing: dict[str, int] = {}
        for edge in self.directed_edges:
            if edge.source_id:
                outgoing[edge.source_id] = outgoing.get(edge.source_id, 0) + 1
        return [node_id for node_id, count in outgoing.items() if count > 1]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "background": list(self.background),
            "threshold_used": self.threshold_used,
            "stroke_threshold": self.stroke_threshold,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "box": list(node.box),
                    "shape": node.shape.value,
                    "fill_ratio": round(node.fill_ratio, 3),
                    "text": node.text,
                    "text_confidence": round(node.text_confidence, 3),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_point": list(edge.source_point) if edge.source_point else None,
                    "target_point": list(edge.target_point) if edge.target_point else None,
                    "resolved_by": edge.resolved_by,
                    "source_ink": edge.source_ink,
                    "target_ink": edge.target_ink,
                    "directed": edge.directed,
                    "direction_ratio": round(edge.direction_ratio, 3),
                    "ambiguous": edge.ambiguous,
                    "reason": edge.reason,
                    "label": edge.label,
                    "confidence": round(edge.confidence, 3),
                }
                for edge in self.edges
            ],
            "counts": {
                "nodes": self.node_count,
                "edges": self.edge_count,
                "connected": len(self.connected_edges),
                "directed": len(self.directed_edges),
                "undirected_only": len(self.undirected_only_edges),
                "ambiguous": len(self.ambiguous_edges),
                "branch_nodes": len(self.branch_node_ids),
            },
            "problems": self.problems,
        }


# --------------------------------------------------------------------------- #
# 1. Background and foreground
# --------------------------------------------------------------------------- #


def modal_background(image: np.ndarray) -> np.ndarray:
    """The most common colour, which for these renderings is the background.

    Computed by packing BGR into one integer and using :func:`numpy.unique`. A ``Counter`` over
    numpy tuples was the first attempt and was abandoned: it is orders of magnitude slower and
    stalled the measurement outright.
    """
    small = image[::3, ::3].reshape(-1, 3).astype(np.uint32)
    packed = (small[:, 0] << 16) | (small[:, 1] << 8) | small[:, 2]
    values, counts = np.unique(packed, return_counts=True)
    top = int(values[counts.argmax()])
    return np.array([(top >> 16) & 255, (top >> 8) & 255, top & 255], dtype=np.int16)


def foreground_mask(image: np.ndarray, background: np.ndarray, threshold: int) -> np.ndarray:
    """Everything that is not the background, by per-channel distance.

    ``max`` across channels rather than a sum, so a strong difference in one channel is enough -
    which is what distinguishes a light node fill from a light background of a different hue.
    """
    diff = np.abs(image.astype(np.int16) - background).max(axis=2)
    return (diff > threshold).astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# 2 and 3. Nodes: shapes, and choosing a threshold without ground truth
# --------------------------------------------------------------------------- #


#: A row of a node's blob counts as "full width" if it spans at least this fraction of the node's
#: width. Rows below it, at the top or bottom of the blob, are an arrowhead fused onto the node
#: rather than part of it. See :func:`_refine_box`.
BOX_FULL_WIDTH_RATIO = 0.6

#: Refinement may not remove more than this fraction of a blob. If it wants to, the blob is not a
#: node with something stuck to it and the original rectangle is kept.
BOX_MIN_KEPT_AREA = 0.4


def _longest_run(keep: np.ndarray) -> tuple[int, int]:
    """Start and end of the longest run of ``True``. ``(0, 0)`` when there is none."""
    best_start = best_end = 0
    best_length = 0
    start: int | None = None
    for index, value in enumerate([*keep.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best_length:
                best_length, best_start, best_end = index - start, start, index
            start = None
    return best_start, best_end


def _refine_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Trim anything fused onto a node off its bounding box.

    A solid arrowhead is about 11 pixels across, which is wider than the 7-pixel opening kernel, so
    it survives the opening. When it touches the node it joins the same component and the bounding
    rectangle grows by the length of the arrow. Measured on ``8240.jpeg``: the detected top edge
    sat 8 pixels above the node, which put three arrowheads *inside* the box, where the stroke
    layer then deleted them along with the node - and with the arrowheads gone, direction was
    decided by noise. That chart scored 0% on direction.

    Every node in this corpus is drawn as a rounded rectangle - all 13,068 of them use Mermaid's
    ``entity(label)`` form - so inside a node every row spans nearly the full width, and a row
    holding only an arrowhead spans a fraction of it. The refinement keeps the longest run of
    full-width rows, then does the same for columns.

    Returns ``(x, y, width, height)`` relative to the mask.
    """
    height, width = mask.shape
    binary = mask > 0
    row_counts = binary.sum(axis=1)
    if not row_counts.any():
        return (0, 0, width, height)
    # A high percentile rather than the maximum, so one bulging row cannot set the standard.
    row_width = float(np.percentile(row_counts[row_counts > 0], 80))
    y0, y1 = _longest_run(row_counts >= BOX_FULL_WIDTH_RATIO * row_width)
    if y1 <= y0:
        return (0, 0, width, height)

    column_counts = binary[y0:y1].sum(axis=0)
    if not column_counts.any():
        return (0, y0, width, y1 - y0)
    column_height = float(np.percentile(column_counts[column_counts > 0], 80))
    x0, x1 = _longest_run(column_counts >= BOX_FULL_WIDTH_RATIO * column_height)
    if x1 <= x0:
        return (0, y0, width, y1 - y0)

    if (x1 - x0) * (y1 - y0) < BOX_MIN_KEPT_AREA * width * height:
        # Refinement wants to throw most of the blob away, so the blob is not a node with an arrow
        # stuck to it and this heuristic does not apply.
        return (0, 0, width, height)
    return (x0, y0, x1 - x0, y1 - y0)


def _classify_shape(contour: np.ndarray, width: int, height: int, fill_ratio: float) -> NodeShape:
    """Rectangle, rounded, diamond or ellipse, from the contour's vertex count.

    ``approxPolyDP`` at 2% of the perimeter: a Mermaid rectangle approximates to four vertices,
    a diamond also to four but rotated, and a rounded box or stadium to more. The diamond is
    told apart by its vertices sitting near the edge midpoints rather than the corners.

    **This classifier is unmeasured.** Every node in the FlowLearn simulated corpus is a rounded
    rectangle, so the corpus cannot tell a correct diamond from an incorrect one and the shape
    field must not be relied on until there is a corpus that contains other shapes. It is kept
    because real procedure workflows do use a diamond for a decision, and it is reported in the
    overlay so a reviewer can see what it said rather than trusting it.
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return NodeShape.UNKNOWN
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)

    if vertices == 4:
        points = approx.reshape(-1, 2).astype(np.float32)
        centre = points.mean(axis=0)
        # For a diamond the vertices lie on the axes through the centre; for a rectangle they
        # sit diagonally. Compare how axis-aligned the offsets are.
        offsets = np.abs(points - centre)
        axis_aligned = np.mean(
            np.minimum(offsets[:, 0], offsets[:, 1]) / (np.maximum(offsets[:, 0], offsets[:, 1]) + 1e-6)
        )
        if axis_aligned < 0.35:
            return NodeShape.DIAMOND
        return NodeShape.RECTANGLE
    if vertices > 6 and fill_ratio > 0.85:
        # A near-perfect fill with many vertices is a rounded rectangle; an ellipse fills
        # about pi/4 of its box.
        return NodeShape.ROUNDED
    if vertices > 6 and 0.7 < fill_ratio <= 0.85:
        return NodeShape.ELLIPSE
    return NodeShape.ROUNDED if fill_ratio > 0.8 else NodeShape.UNKNOWN


def _boxes_at_threshold(
    image: np.ndarray, background: np.ndarray, threshold: int
) -> tuple[list[tuple[tuple[int, int, int, int], float, NodeShape]], np.ndarray]:
    """Node candidates at one threshold, plus the opened mask they came from."""
    foreground = foreground_mask(image, background, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (NODE_OPEN_KERNEL, NODE_OPEN_KERNEL))
    opened = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[tuple[int, int, int, int], float, NodeShape]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < MIN_NODE_AREA or min(w, h) < MIN_NODE_SIDE:
            continue

        # Trim any arrowhead fused onto the node before anything else uses this rectangle. The
        # box feeds the stroke layer, the OCR crop and the attachment map, so an inflated box
        # damages all three at once.
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [contour - np.array([[x, y]])], -1, 255, thickness=-1)
        dx, dy, dw, dh = _refine_box(blob)
        x, y, w, h = x + dx, y + dy, dw, dh
        if min(w, h) < MIN_NODE_SIDE:
            continue

        area = float((blob[dy : dy + dh, dx : dx + dw] > 0).sum())
        if area < MIN_NODE_AREA:
            continue
        fill_ratio = area / float(w * h) if w * h else 0.0
        if fill_ratio < MIN_NODE_FILL_RATIO:
            continue
        found.append(((int(x), int(y), int(w), int(h)), fill_ratio, _classify_shape(contour, w, h, fill_ratio)))
    return found, opened


def _height_consistency(boxes: list[tuple[tuple[int, int, int, int], float, NodeShape]]) -> float:
    """Share of boxes whose height agrees with the median.

    The selection signal for the threshold sweep, and it needs no ground truth: Mermaid lays out
    every node at a near-uniform height, so a threshold that has started merging nodes with
    strokes produces a set whose heights scatter.
    """
    if len(boxes) < 2:
        return 0.0
    heights = [box[0][3] for box in boxes]
    median = statistics.median(heights)
    if median <= 0:
        return 0.0
    return sum(1 for h in heights if abs(h - median) / median <= NODE_HEIGHT_TOLERANCE) / len(heights)


def detect_nodes(image: np.ndarray, background: np.ndarray) -> tuple[list[DetectedNode], int, np.ndarray]:
    """Find the boxes, sweeping thresholds until the set is self-consistent.

    Returns the nodes, the threshold that produced them, and the opened mask.
    """
    best_score = 0.0
    best: tuple[list, int, np.ndarray] = ([], FOREGROUND_THRESHOLDS[0], np.zeros(image.shape[:2], np.uint8))

    for threshold in FOREGROUND_THRESHOLDS:
        boxes, opened = _boxes_at_threshold(image, background, threshold)
        consistency = _height_consistency(boxes)
        # Count matters as well as consistency: two boxes agreeing perfectly is not better than
        # ten boxes agreeing well.
        score = consistency * min(len(boxes), 40)
        if score > best_score:
            best_score, best = score, (boxes, threshold, opened)
        if consistency >= 0.9 and len(boxes) >= 2:
            # Good enough, and lowering the threshold further only risks bleeding strokes into
            # the node mask.
            best = (boxes, threshold, opened)
            break

    boxes, threshold, opened = best
    # Ordered top-to-bottom then left-to-right, so ids are stable across runs and a human
    # reading the overlay meets them in a predictable order.
    boxes.sort(key=lambda item: (item[0][1], item[0][0]))
    nodes = [
        DetectedNode(
            node_id=f"n{index}",
            x=box[0],
            y=box[1],
            width=box[2],
            height=box[3],
            shape=shape,
            fill_ratio=fill_ratio,
        )
        for index, (box, fill_ratio, shape) in enumerate(boxes)
    ]
    return nodes, threshold, opened


# --------------------------------------------------------------------------- #
# 4. The stroke layer: where the nodes are not
# --------------------------------------------------------------------------- #


def node_block(nodes: list[DetectedNode], shape: tuple[int, int], pad: int = NODE_BLOCK_PAD) -> np.ndarray:
    """Filled bounding boxes of every node, padded, as one mask."""
    block = np.zeros(shape, np.uint8)
    for node in nodes:
        cv2.rectangle(
            block,
            (node.x - pad, node.y - pad),
            (node.x + node.width + pad, node.y + node.height + pad),
            255,
            thickness=-1,
        )
    return block


def owner_map(nodes: list[DetectedNode], shape: tuple[int, int], pad: int = NODE_BLOCK_PAD) -> np.ndarray:
    """Which node occupies each pixel, or -1. Built once; the alternative was testing every node
    for every pixel, which cost ten times as much for the same answer."""
    owner = np.full(shape, -1, dtype=np.int16)
    for index, node in enumerate(nodes):
        y0, y1 = max(node.y - pad, 0), min(node.y + node.height + pad, shape[0])
        x0, x1 = max(node.x - pad, 0), min(node.x + node.width + pad, shape[1])
        owner[y0:y1, x0:x1] = index
    return owner


def attachment_map(nodes: list[DetectedNode], shape: tuple[int, int], gap: int = ATTACH_GAP) -> np.ndarray:
    """The owner map widened by *gap*, so a connector stopping just short still counts as attached.

    Widened by repeated four-neighbour propagation rather than by dilating each node separately,
    so where two nodes are close the nearer one wins instead of the later one overwriting.
    """
    grown = owner_map(nodes, shape)
    for _ in range(gap):
        shifted = grown.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rolled = np.roll(grown, (dy, dx), axis=(0, 1))
            take = (shifted < 0) & (rolled >= 0)
            shifted[take] = rolled[take]
        grown = shifted
    return grown


def _line_kernel(length: int, degrees: float) -> np.ndarray:
    """A one-pixel-wide line through the centre of a square, at *degrees*."""
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), np.uint8)
    centre = size // 2
    radians = np.deg2rad(degrees)
    dx, dy = float(np.cos(radians)), float(np.sin(radians))
    cv2.line(
        kernel,
        (round(centre - dx * centre), round(centre - dy * centre)),
        (round(centre + dx * centre), round(centre + dy * centre)),
        1,
        1,
    )
    return kernel


_BRIDGE_KERNELS: tuple[np.ndarray, ...] = tuple(
    _line_kernel(BRIDGE_LENGTH, 180.0 * index / BRIDGE_ANGLES) for index in range(BRIDGE_ANGLES)
)


def bridge_dashes(strokes: np.ndarray) -> np.ndarray:
    """Join a dotted connector into one component without fattening it sideways.

    A closing with a thin line kernel spans a gap along that line's direction and leaves
    everything else alone, so unioning the result over 12 orientations reconnects dashes at any
    angle while keeping two parallel curves apart. See the module docstring for the measurement
    against square closing.
    """
    out = strokes.copy()
    for kernel in _BRIDGE_KERNELS:
        out = cv2.bitwise_or(out, cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel))
    return out


def _resolvable_count(bridged: np.ndarray, attachment: np.ndarray) -> int:
    """How many components join exactly two nodes. The stroke threshold's selection signal."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    total = 0
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < MIN_CONNECTOR_AREA:
            continue
        touched = {int(value) for value in np.unique(attachment[labels == index]) if value >= 0}
        if len(touched) == 2:
            total += 1
    return total


def stroke_layer(
    image: np.ndarray, background: np.ndarray, nodes: list[DetectedNode], attachment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """The connector layer, at whichever threshold resolves the most connectors.

    Returns the raw mask, the dash-bridged mask, and the threshold chosen. The raw mask is kept
    because direction is decided by counting ink, and bridging adds pixels that were never drawn.
    """
    outside = cv2.bitwise_not(node_block(nodes, image.shape[:2]))
    best: tuple[np.ndarray | None, np.ndarray | None, int, int] = (None, None, STROKE_THRESHOLDS[0], -1)
    for threshold in STROKE_THRESHOLDS:
        candidate = cv2.bitwise_and(foreground_mask(image, background, threshold), outside)
        if float((candidate > 0).mean()) > STROKE_SHARE_CAP:
            continue
        bridged = bridge_dashes(candidate)
        resolvable = _resolvable_count(bridged, attachment)
        if resolvable > best[3]:
            best = (candidate, bridged, threshold, resolvable)

    if best[0] is None or best[1] is None:
        # Every threshold floods. Take the strictest and let validation report the state.
        raw = cv2.bitwise_and(foreground_mask(image, background, STROKE_THRESHOLDS[0]), outside)
        return raw, bridge_dashes(raw), STROKE_THRESHOLDS[0]
    return best[0], best[1], best[2]


# --------------------------------------------------------------------------- #
# 5. Connectors: components, and tracing the ones that merged
# --------------------------------------------------------------------------- #

#: Offsets forming a ring of radius ``TRACE_STEP``, the tracer's candidate next positions.
_TRACE_RING: tuple[tuple[int, int], ...] = tuple(
    (dx, dy)
    for dx in range(-TRACE_STEP, TRACE_STEP + 1)
    for dy in range(-TRACE_STEP, TRACE_STEP + 1)
    if abs(float(np.hypot(dx, dy)) - TRACE_STEP) <= 0.75
)


def _contact_points(component: np.ndarray, attachment: np.ndarray, index: int) -> list[tuple[float, float]]:
    """Centroid of each separate place where this component meets node *index*."""
    contact = (component & (attachment == index)).astype(np.uint8)
    count, _labels, _stats, centroids = cv2.connectedComponentsWithStats(contact, connectivity=8)
    return [(float(centroids[i][0]), float(centroids[i][1])) for i in range(1, count)]


def _ink_near(point: tuple[float, float], radius: int, component: np.ndarray, strokes: np.ndarray) -> int:
    """Drawn pixels of this component within *radius* of *point*.

    Counted on the raw stroke mask, so bridged-in pixels cannot masquerade as an arrowhead.
    """
    height, width = component.shape
    cx, cy = round(float(point[0])), round(float(point[1]))
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, height)
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, width)
    window = component[y0:y1, x0:x1]
    if not window.any():
        return 0
    return int(((strokes[y0:y1, x0:x1] > 0) & window).sum())


def _trace_curve(
    bridged: np.ndarray, attachment: np.ndarray, start: tuple[float, float], heading: np.ndarray, from_index: int
) -> tuple[int, tuple[int, int]] | None:
    """Follow one curve out of a bundle, carrying straight on where curves cross.

    Returns the node index reached and the point at which it was reached, or ``None`` when the
    curve could not be followed - which is reported as ambiguity rather than filled in.
    """
    height, width = bridged.shape
    position = np.array([float(start[0]), float(start[1])])
    direction = heading.astype(float).copy()
    for _ in range(TRACE_MAX_STEPS):
        best: tuple[int, int, np.ndarray] | None = None
        best_angle = TRACE_MAX_TURN
        for dx, dy in _TRACE_RING:
            nx, ny = round(float(position[0]) + dx), round(float(position[1]) + dy)
            if not (0 <= nx < width and 0 <= ny < height) or not bridged[ny, nx]:
                continue
            step = np.array([float(dx), float(dy)])
            step /= float(np.linalg.norm(step))
            angle = float(np.degrees(np.arccos(np.clip(float(np.dot(step, direction)), -1.0, 1.0))))
            if angle < best_angle:
                best, best_angle = (nx, ny, step), angle
        if best is None:
            return None
        nx, ny, step = best
        position = np.array([float(nx), float(ny)])
        direction = direction * TRACE_HEADING_INERTIA + step * (1.0 - TRACE_HEADING_INERTIA)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction /= norm
        here = int(attachment[ny, nx])
        if here >= 0 and here != from_index:
            return here, (nx, ny)
    return None


def _direction_of(
    component: np.ndarray,
    strokes: np.ndarray,
    radius: int,
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[bool, int, int, float]:
    """Which end carries the arrowhead, by comparing ink at the two ends of one connector.

    Returns ``(second end is the head, ink at first, ink at second, ratio)``. The caller decides
    what to do when the ratio is too close to call.
    """
    ink_first = _ink_near(first, radius, component, strokes)
    ink_second = _ink_near(second, radius, component, strokes)
    larger, smaller = max(ink_first, ink_second), min(ink_first, ink_second)
    ratio = larger / smaller if smaller else float("inf") if larger else 1.0
    return ink_second > ink_first, ink_first, ink_second, ratio


def extract_edges(
    strokes: np.ndarray,
    bridged: np.ndarray,
    nodes: list[DetectedNode],
    attachment: np.ndarray,
) -> list[DetectedEdge]:
    """Turn the connector layer into edges, recording how each one was established."""
    if not nodes:
        return []
    radius = max(8, round(DIRECTION_RADIUS_RATIO * float(np.median([node.height for node in nodes]))))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    edges: list[DetectedEdge] = []

    def add(
        source: int | None,
        target: int | None,
        component: np.ndarray,
        first: tuple[float, float] | None,
        second: tuple[float, float] | None,
        resolved_by: str,
        reason: str = "",
    ) -> None:
        """Record one edge, deciding its direction from ink when both ends are known."""
        index = len(edges)
        if source is None or target is None or first is None or second is None:
            edges.append(
                DetectedEdge(
                    edge_id=f"e{index}",
                    source_id=nodes[source].node_id if source is not None else None,
                    target_id=nodes[target].node_id if target is not None else None,
                    source_point=(round(first[0]), round(first[1])) if first else None,
                    target_point=(round(second[0]), round(second[1])) if second else None,
                    resolved_by=resolved_by,
                    ambiguous=True,
                    reason=reason,
                )
            )
            return

        second_is_head, ink_first, ink_second, ratio = _direction_of(component, strokes, radius, first, second)
        established = ratio >= MIN_DIRECTION_RATIO
        if second_is_head:
            src, tgt, src_point, tgt_point = source, target, first, second
            src_ink, tgt_ink = ink_first, ink_second
        else:
            src, tgt, src_point, tgt_point = target, source, second, first
            src_ink, tgt_ink = ink_second, ink_first
        edges.append(
            DetectedEdge(
                edge_id=f"e{index}",
                source_id=nodes[src].node_id,
                target_id=nodes[tgt].node_id,
                source_point=(round(src_point[0]), round(src_point[1])),
                target_point=(round(tgt_point[0]), round(tgt_point[1])),
                resolved_by=resolved_by,
                source_ink=src_ink,
                target_ink=tgt_ink,
                directed=established,
                direction_ratio=min(ratio, 99.0),
                ambiguous=False,
                reason=""
                if established
                else (
                    f"both ends carry a similar amount of ink ({ink_first} vs {ink_second}), so which end "
                    f"has the arrowhead is not established"
                ),
                confidence=min(1.0, 0.5 + 0.1 * ratio) if established else 0.0,
            )
        )

    for component_index in range(1, count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) < MIN_CONNECTOR_AREA:
            continue
        component = labels == component_index
        touched = sorted({int(value) for value in np.unique(attachment[component]) if value >= 0})

        if len(touched) == 0:
            # Drawn, attached to nothing. Usually a glyph belonging to a node that was missed.
            continue
        if len(touched) == 1:
            only = touched[0]
            points = _contact_points(component, attachment, only)
            add(
                only,
                None,
                component,
                points[0] if points else None,
                None,
                "component",
                "this connector reaches only one node, so what it joins is not established",
            )
            continue
        if len(touched) == 2:
            first_points = _contact_points(component, attachment, touched[0])
            second_points = _contact_points(component, attachment, touched[1])
            if not first_points or not second_points:
                continue
            add(touched[0], touched[1], component, first_points[0], second_points[0], "component")
            continue

        # Three or more nodes: crossing or fanning curves. Follow each one.
        free = component & (attachment < 0)
        free_ys, free_xs = np.nonzero(free)
        seen: set[tuple[int, int]] = set()
        traced_any = False
        for node_index in touched:
            for contact in _contact_points(component, attachment, node_index):
                if free_ys.size == 0:
                    continue
                # Step just outside the attachment band and head away from the node.
                distances = (free_xs - contact[0]) ** 2 + (free_ys - contact[1]) ** 2
                nearest = int(distances.argmin())
                start = (float(free_xs[nearest]), float(free_ys[nearest]))
                heading = np.array([start[0] - contact[0], start[1] - contact[1]])
                norm = float(np.linalg.norm(heading))
                if norm < 1e-6:
                    continue
                result = _trace_curve(bridged, attachment, start, heading / norm, node_index)
                if result is None:
                    continue
                reached, arrival = result
                key = (min(node_index, reached), max(node_index, reached))
                if key in seen:
                    continue
                seen.add(key)
                traced_any = True
                other = _contact_points(component, attachment, reached)
                far = min(other, key=lambda p: (p[0] - arrival[0]) ** 2 + (p[1] - arrival[1]) ** 2) if other else None
                add(node_index, reached, component, contact, far, "trace")
        if not traced_any:
            names = ", ".join(nodes[i].node_id for i in touched)
            points = _contact_points(component, attachment, touched[0])
            add(
                touched[0],
                None,
                component,
                points[0] if points else None,
                None,
                "trace",
                f"curves touching {names} cross here and none could be followed through",
            )

    return edges


# --------------------------------------------------------------------------- #
# 6. Consistency validation
# --------------------------------------------------------------------------- #


def validate(parse: DiagramParse) -> list[str]:
    """Faults that would make this graph unsafe to route a case with.

    Cycles are **not** a fault. A troubleshooting chart that says "no? go back and retry" is a
    cycle, and rejecting it would reject the real thing.
    """
    problems: list[str] = []
    if not parse.nodes:
        problems.append("no nodes were detected")
    if not parse.edges:
        problems.append("no connectors were detected, so nothing is joined")

    ambiguous = parse.ambiguous_edges
    if ambiguous:
        problems.append(f"{len(ambiguous)} of {len(parse.edges)} connectors do not have both ends established")

    undirected = parse.undirected_only_edges
    if undirected:
        problems.append(f"{len(undirected)} connector(s) have no established direction")

    for edge in parse.connected_edges:
        if edge.source_id == edge.target_id:
            problems.append(f"edge {edge.edge_id} starts and ends at {edge.target_id}")

    # A node no connector touches is suspicious: Mermaid charts connect what they draw.
    touched = {edge.target_id for edge in parse.edges} | {edge.source_id for edge in parse.edges}
    isolated = [node.node_id for node in parse.nodes if node.node_id not in touched]
    if isolated and len(isolated) < len(parse.nodes):
        problems.append(f"{len(isolated)} node(s) touched by no connector: {', '.join(isolated[:4])}")

    unnamed = [node.node_id for node in parse.nodes if not node.text.strip()]
    if unnamed:
        problems.append(f"{len(unnamed)} node(s) have no readable text")

    return problems


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def parse_diagram(image_path: Path, *, read_text: bool = True) -> DiagramParse:
    """Recover nodes, connectors and directions from one flowchart image.

    ``read_text`` runs OCR per node. It is the slowest stage by far, so it can be switched off
    when only the topology is being measured.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return DiagramParse(
            image_path=str(image_path),
            width=0,
            height=0,
            background=(0, 0, 0),
            threshold_used=0,
            problems=[f"could not read {image_path}"],
        )

    background = modal_background(image)
    nodes, threshold, opened = detect_nodes(image, background)
    attachment = attachment_map(nodes, image.shape[:2])
    strokes, bridged, stroke_threshold = stroke_layer(image, background, nodes, attachment)
    edges = extract_edges(strokes, bridged, nodes, attachment)

    parse = DiagramParse(
        image_path=str(image_path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        background=tuple(int(v) for v in background),  # type: ignore[arg-type]
        threshold_used=threshold,
        stroke_threshold=stroke_threshold,
        nodes=nodes,
        edges=edges,
        masks={"nodes": opened, "strokes": strokes, "bridged": bridged},
    )

    if read_text:
        from agentpath_ingestion.corpora.diagram_ocr import read_node_text

        parse.nodes = read_node_text(image, parse.nodes)

    parse.problems = validate(parse)
    logger.debug(
        "diagram.parsed",
        image=image_path.name,
        nodes=len(parse.nodes),
        edges=len(parse.edges),
        directed=len(parse.directed_edges),
        threshold=threshold,
        stroke_threshold=stroke_threshold,
    )
    return parse


__all__ = [
    "ATTACH_GAP",
    "BRIDGE_ANGLES",
    "BRIDGE_LENGTH",
    "FOREGROUND_THRESHOLDS",
    "MIN_CONNECTOR_AREA",
    "MIN_DIRECTION_RATIO",
    "NODE_BLOCK_PAD",
    "NODE_OPEN_KERNEL",
    "STROKE_THRESHOLDS",
    "DetectedEdge",
    "DetectedNode",
    "DiagramParse",
    "NodeShape",
    "attachment_map",
    "bridge_dashes",
    "detect_nodes",
    "extract_edges",
    "foreground_mask",
    "modal_background",
    "node_block",
    "owner_map",
    "parse_diagram",
    "stroke_layer",
    "validate",
]

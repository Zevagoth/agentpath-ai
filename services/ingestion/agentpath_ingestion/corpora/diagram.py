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

4. **Arrowheads are the edge detector.** This is the load-bearing decision. Connector curves
   *cross* in a dense chart, so their connected components merge and counting components
   undercounts edges - the same failure the language model had. An arrowhead is a solid blob,
   so it survives an erosion that thins the curve away, and **one arrowhead is exactly one
   edge**. Counting heads is therefore robust to crossings in a way that counting curves is not.

5. **Direction comes from the arrowhead, not from geometry.** The head end is where the
   arrowhead is. The tail is found by walking the stroke mask away from the head.

6. **Association is geodesic, not euclidean.** The source node is the first node boundary
   reached by a breadth-first walk *along the stroke pixels* from the arrowhead. A straight-line
   nearest-node test would pick whichever box happens to sit close to the curve's middle, which
   on a curved connector is frequently the wrong one.

7. **Ambiguity is recorded, never resolved by guessing.** A head whose walk reaches two node
   boundaries at a similar distance, or reaches none, produces an edge marked ambiguous with
   the reason attached. The consumer shows the original image and says review is needed.

Loops are legal. Nothing here assumes a DAG, and a cycle is never a reason to reject a graph:
"no? go back and retry" is a real branch in a real troubleshooting chart.
"""

from __future__ import annotations

import statistics
from collections import deque
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

#: Tried in this order. The first producing a self-consistent box set wins. 40 handles the
#: common coloured backgrounds; 14 and 8 exist for white-on-white, where the node fill differs
#: from the background by under 20 levels.
FOREGROUND_THRESHOLDS = (40, 24, 14, 8)

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

#: Node regions are grown by this much before being subtracted from the stroke mask, so a
#: connector that touches a border does not inherit the node's pixels.
NODE_DILATION = 3

#: Window for the local fill density that separates arrowheads from lines. Wide enough that a
#: line's contribution is diluted, narrow enough that two nearby heads stay separate.
ARROWHEAD_WINDOW = 9

#: Fill fraction above which a window is arrowhead rather than line. A 3-pixel line fills about
#: 0.33 of a 9x9 window; a filled triangle fills most of it. Chosen in the gap, on dev.
ARROWHEAD_MIN_DENSITY = 0.5

#: Minimum blob area to count as an arrowhead, after the density filter.
MIN_ARROWHEAD_AREA = 6

#: How far the geodesic walk may travel along the stroke mask before giving up on finding a
#: source. Generous: a connector on a 1,500-px-wide chart can be long.
MAX_WALK_STEPS = 4000

#: When the two best source candidates are within this ratio of each other, which one the
#: curve actually came from is not established, and the edge is marked ambiguous.
SOURCE_MARGIN_RATIO = 1.35


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
    """One arrow, with the evidence that produced it.

    ``ambiguous`` is the honest outcome when the walk could not establish a source. Such an edge
    is still reported - the arrowhead is real - but it must not be used to route anything.
    """

    edge_id: str
    #: ``None`` when the walk found no unambiguous source.
    source_id: str | None
    target_id: str
    #: Where the arrowhead sits.
    head: tuple[int, int]
    #: Where the walk ended, i.e. the tail.
    tail: tuple[int, int] | None
    #: Geodesic length of the walk in pixels. Short is more trustworthy.
    walk_length: int
    arrowhead_area: int
    ambiguous: bool = False
    reason: str = ""
    #: Text found beside the arrow, e.g. "Yes". Filled by the OCR stage.
    label: str = ""
    confidence: float = 0.0


@dataclass
class DiagramParse:
    """Everything recovered from one image, plus what could not be."""

    image_path: str
    width: int
    height: int
    background: tuple[int, int, int]
    threshold_used: int
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
    def confident_edges(self) -> list[DetectedEdge]:
        return [edge for edge in self.edges if not edge.ambiguous and edge.source_id]

    @property
    def ambiguous_edges(self) -> list[DetectedEdge]:
        return [edge for edge in self.edges if edge.ambiguous or not edge.source_id]

    @property
    def label_edges(self) -> list[tuple[str, str]]:
        """Confident edges as ``(source text, target text)`` pairs."""
        by_id = {node.node_id: node.text for node in self.nodes}
        return [(by_id.get(edge.source_id or "", ""), by_id.get(edge.target_id, "")) for edge in self.confident_edges]

    @property
    def branch_node_ids(self) -> list[str]:
        """Nodes with more than one confident outgoing edge. The decision points."""
        outgoing: dict[str, int] = {}
        for edge in self.confident_edges:
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
                    "head": list(edge.head),
                    "tail": list(edge.tail) if edge.tail else None,
                    "walk_length": edge.walk_length,
                    "arrowhead_area": edge.arrowhead_area,
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
                "confident_edges": len(self.confident_edges),
                "ambiguous_edges": len(self.ambiguous_edges),
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


def _classify_shape(contour: np.ndarray, width: int, height: int, fill_ratio: float) -> NodeShape:
    """Rectangle, rounded, diamond or ellipse, from the contour's vertex count.

    ``approxPolyDP`` at 2% of the perimeter: a Mermaid rectangle approximates to four vertices,
    a diamond also to four but rotated, and a rounded box or stadium to more. The diamond is
    told apart by its vertices sitting near the edge midpoints rather than the corners.
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
        area = cv2.contourArea(contour)
        if area < MIN_NODE_AREA or min(w, h) < MIN_NODE_SIDE:
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
# 4 and 5. Connectors and arrowheads
# --------------------------------------------------------------------------- #


def stroke_mask(image: np.ndarray, background: np.ndarray, nodes: list[DetectedNode], threshold: int) -> np.ndarray:
    """Foreground with the node regions removed: what is left is connectors.

    Node boxes are dilated before subtraction so a connector that touches a border does not
    inherit the node's own pixels, which would fuse every arrow meeting that node into one
    component.
    """
    foreground = foreground_mask(image, background, threshold)
    occupied = np.zeros(image.shape[:2], np.uint8)
    for node in nodes:
        cv2.rectangle(
            occupied,
            (node.x - NODE_DILATION, node.y - NODE_DILATION),
            (node.x + node.width + NODE_DILATION, node.y + node.height + NODE_DILATION),
            255,
            thickness=-1,
        )
    return cv2.bitwise_and(foreground, cv2.bitwise_not(occupied))


def detect_arrowheads(strokes: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    """Find the solid triangles. Each one is exactly one edge.

    This is what makes the parser robust to crossing connectors: two curves that intersect merge
    into one connected component, so counting components undercounts, but each curve still ends
    in its own arrowhead.

    **Local fill density**, not erosion and not a distance transform. Both of those were measured
    on dev and rejected:

    * *Erosion* with a 3-pixel kernel leaves a 3-pixel line one pixel wide instead of removing
      it, so curve fragments were counted as heads. Measured: 33 heads found where 13 edges
      existed, and arrow-count accuracy of 10%.
    * *Distance transform* separates them in principle - a line's inscribed radius is about 2 and
      an arrowhead's about 4 - but the margin is too narrow to threshold reliably once JPEG
      compression has blurred a 22 KB image. Measured across 14 charts, the best cutoff was
      exact on some and off by a factor of two on others.

    Density has a much wider margin. In a 9x9 window a 3-pixel-wide line fills about a third;
    an arrowhead fills most of it. The threshold sits in a gap rather than on a knife edge.

    Returns ``(centroid, area)`` per head.
    """
    # Fraction of each window that is stroke. normalize=True makes the box filter an average, so
    # the result is directly a fill fraction in 0..1.
    density = cv2.boxFilter(
        (strokes > 0).astype(np.float32),
        ddepth=-1,
        ksize=(ARROWHEAD_WINDOW, ARROWHEAD_WINDOW),
        normalize=True,
    )
    peaks = ((density >= ARROWHEAD_MIN_DENSITY) & (strokes > 0)).astype(np.uint8) * 255
    # Close pinholes so one arrowhead is one component rather than a cluster.
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    heads: list[tuple[tuple[int, int], int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < MIN_ARROWHEAD_AREA:
            continue
        cx, cy = centroids[index]
        heads.append(((round(float(cx)), round(float(cy))), area))
    return heads


# --------------------------------------------------------------------------- #
# 6. Association: which nodes does this arrow join?
# --------------------------------------------------------------------------- #


def _nearest_node(nodes: list[DetectedNode], px: int, py: int) -> tuple[DetectedNode | None, float]:
    """The node whose border is closest to a point, with that distance."""
    best: DetectedNode | None = None
    best_distance = float("inf")
    for node in nodes:
        dx = max(node.x - px, 0, px - (node.x + node.width))
        dy = max(node.y - py, 0, py - (node.y + node.height))
        distance = float(np.hypot(dx, dy))
        if distance < best_distance:
            best, best_distance = node, distance
    return best, best_distance


def _walk_to_source(
    strokes: np.ndarray,
    nodes: list[DetectedNode],
    head: tuple[int, int],
    target: DetectedNode,
) -> tuple[DetectedNode | None, tuple[int, int] | None, int, str]:
    """Walk the connector away from the arrowhead until a different node is reached.

    Breadth-first over the mask, so distance is *along the curve* rather than across the image.
    A straight-line nearest-node test picks whichever box sits near the curve's middle, which on
    the bezier connectors these charts use is frequently not the source.

    The mask walked is the **full foreground**, with node boxes acting as barriers that are
    entered but never expanded from. The first attempt walked a mask with the node regions
    deleted, and it failed on most arrows: a connector routed close to a box was severed by that
    deletion, so the walk exhausted after a fifth of the curve and reported no source. Measured
    on ``9837.jpeg``: 83 reachable pixels out of a ~430-pixel curve.

    Returns ``(source, tail point, steps, reason)``. ``source`` is ``None`` when nothing was
    reached or when two candidates were too close to separate, and ``reason`` says which.
    """
    height, width = strokes.shape
    start_x, start_y = head
    if not (0 <= start_x < width and 0 <= start_y < height):
        return None, None, 0, "arrowhead lies outside the image"

    # Seed from the arrowhead's neighbourhood: its centroid can fall on a gap between the
    # triangle's pixels after erosion.
    seeds: list[tuple[int, int, int]] = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            sx, sy = start_x + dx, start_y + dy
            if 0 <= sx < width and 0 <= sy < height and strokes[sy, sx]:
                seeds.append((sx, sy, 0))
    if not seeds:
        return None, None, 0, "no stroke pixels next to the arrowhead"

    # A lookup of which node owns each pixel, built once. Checking every node for every visited
    # pixel was the first version and made the walk ten times more expensive for no benefit.
    owner = np.full(strokes.shape, -1, dtype=np.int16)
    for index, node in enumerate(nodes):
        y0 = max(node.y - NODE_DILATION, 0)
        y1 = min(node.y + node.height + NODE_DILATION, height)
        x0 = max(node.x - NODE_DILATION, 0)
        x1 = min(node.x + node.width + NODE_DILATION, width)
        owner[y0:y1, x0:x1] = index
    target_index = next((i for i, node in enumerate(nodes) if node.node_id == target.node_id), -1)

    visited = np.zeros(strokes.shape, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    for sx, sy, _ in seeds:
        visited[sy, sx] = True
        queue.append((sx, sy, 0))

    # Every node reached, with the number of steps it took.
    reached: dict[str, tuple[DetectedNode, int, tuple[int, int]]] = {}
    steps_taken = 0

    while queue and steps_taken < MAX_WALK_STEPS:
        x, y, steps = queue.popleft()
        steps_taken += 1

        here = int(owner[y, x])
        if here >= 0 and here != target_index:
            node = nodes[here]
            if node.node_id not in reached:
                reached[node.node_id] = (node, steps, (x, y))
            # A node is a destination, never a corridor: expanding through it would let the walk
            # emerge on the far side and claim a node the curve never touched.
            continue
        if here == target_index and steps > 0:
            # Back inside the box this arrow points at. Nothing useful lies beyond.
            continue

        # Stop early once something is found and the walk has gone far enough past it to see
        # whether a rival is equally close.
        if reached and steps > min(entry[1] for entry in reached.values()) * SOURCE_MARGIN_RATIO:
            break

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and strokes[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny, steps + 1))

    if not reached:
        return None, None, steps_taken, "the curve from this arrowhead reaches no other node"

    ordered = sorted(reached.values(), key=lambda entry: entry[1])
    source, steps, tail = ordered[0]
    if len(ordered) > 1:
        runner_up_steps = ordered[1][1]
        if steps > 0 and runner_up_steps / max(steps, 1) < SOURCE_MARGIN_RATIO:
            return (
                None,
                tail,
                steps,
                f"two possible sources at similar distance along the curve "
                f"({source.node_id} at {steps}, {ordered[1][0].node_id} at {runner_up_steps})",
            )
    return source, tail, steps, ""


def associate_edges(
    strokes: np.ndarray, nodes: list[DetectedNode], heads: list[tuple[tuple[int, int], int]]
) -> list[DetectedEdge]:
    """Turn arrowheads into directed edges by walking each one back to its source."""
    edges: list[DetectedEdge] = []
    for index, (head, area) in enumerate(heads):
        target, _distance = _nearest_node(nodes, head[0], head[1])
        if target is None:
            edges.append(
                DetectedEdge(
                    edge_id=f"e{index}",
                    source_id=None,
                    target_id="",
                    head=head,
                    tail=None,
                    walk_length=0,
                    arrowhead_area=area,
                    ambiguous=True,
                    reason="no node near this arrowhead",
                )
            )
            continue

        source, tail, steps, reason = _walk_to_source(strokes, nodes, head, target)
        # Confidence is deliberately simple and explainable: a short walk to a single
        # unambiguous source is trustworthy, a long one less so.
        confidence = 0.0 if source is None else max(0.35, min(1.0, 1.0 - steps / 3000.0))
        edges.append(
            DetectedEdge(
                edge_id=f"e{index}",
                source_id=source.node_id if source else None,
                target_id=target.node_id,
                head=head,
                tail=tail,
                walk_length=steps,
                arrowhead_area=area,
                ambiguous=source is None,
                reason=reason,
                confidence=confidence,
            )
        )
    return edges


# --------------------------------------------------------------------------- #
# 7. Consistency validation
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
        problems.append("no arrowheads were detected, so no direction is established")

    ambiguous = parse.ambiguous_edges
    if ambiguous:
        problems.append(f"{len(ambiguous)} of {len(parse.edges)} arrows have no established source")

    # An edge pointing at its own source is either a self-loop the renderer drew or a walk that
    # went wrong. Either way it cannot route anything.
    for edge in parse.confident_edges:
        if edge.source_id == edge.target_id:
            problems.append(f"edge {edge.edge_id} starts and ends at {edge.target_id}")

    # A node no arrow touches is suspicious: Mermaid charts connect what they draw.
    touched = {edge.target_id for edge in parse.edges} | {edge.source_id for edge in parse.edges if edge.source_id}
    isolated = [node.node_id for node in parse.nodes if node.node_id not in touched]
    if isolated and len(isolated) < len(parse.nodes):
        problems.append(f"{len(isolated)} node(s) touched by no arrow: {', '.join(isolated[:4])}")

    unnamed = [node.node_id for node in parse.nodes if not node.text.strip()]
    if unnamed:
        problems.append(f"{len(unnamed)} node(s) have no readable text")

    return problems


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def parse_diagram(image_path: Path, *, read_text: bool = True) -> DiagramParse:
    """Recover nodes, arrows and directions from one flowchart image.

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
    # Two different masks, for two different jobs. Arrowheads are found on the mask with node
    # fills removed, or a node's own fill would survive the erosion and be counted as a head.
    # The walk uses the full foreground, because deleting the nodes severs any curve routed
    # close to a box - see _walk_to_source.
    strokes = stroke_mask(image, background, nodes, threshold)
    walkable = foreground_mask(image, background, threshold)
    heads = detect_arrowheads(strokes)
    edges = associate_edges(walkable, nodes, heads)

    parse = DiagramParse(
        image_path=str(image_path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        background=tuple(int(v) for v in background),  # type: ignore[arg-type]
        threshold_used=threshold,
        nodes=nodes,
        edges=edges,
        masks={"nodes": opened, "strokes": strokes},
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
        confident=len(parse.confident_edges),
        threshold=threshold,
    )
    return parse


__all__ = [
    "ARROWHEAD_MIN_DENSITY",
    "ARROWHEAD_WINDOW",
    "FOREGROUND_THRESHOLDS",
    "MIN_ARROWHEAD_AREA",
    "NODE_OPEN_KERNEL",
    "SOURCE_MARGIN_RATIO",
    "DetectedEdge",
    "DetectedNode",
    "DiagramParse",
    "NodeShape",
    "associate_edges",
    "detect_arrowheads",
    "detect_nodes",
    "foreground_mask",
    "modal_background",
    "parse_diagram",
    "stroke_mask",
    "validate",
]

"""Follow connectors through a crossing by reading the drawing's medial axis.

Why this module exists
----------------------
Split by difficulty on the FlowLearn dev split, the geometric parser scores 93.8% directed edge F1
on the 126 charts where no two connectors cross, and 76.3% on the 176 where some do. Crossings are
most of the edge-detection gap. They are not the whole of the *path* gap - branch F1 is 80.7% and
path reachability 77.8% even on the crossing-free charts - so this module is not expected to fix
those, and they are diagnosed separately.

Two earlier attempts at crossings are worth knowing about, because both failed and this one is
shaped by why.

*Stepping along the mask, preferring to carry straight on.* Cheap, and it resolves a clean X. It
measured 81.0% precision against 96.4% for a connector that needed no tracing, because nothing
about it is global: each attachment traced independently, three attachments could all claim the
same fourth one, and a step taken through a thick blob wanders.

*Constraining those traces to a one-connector-per-attachment matching.* Raised traced precision and
lost more recall than it gained, because a trace that fails leaves its attachment with nothing to
pair with.

What is different here is that the crossing is first turned into a *graph* - arcs between junctions
- and the decision is then made over the whole graph at once: which arc continues into which is
settled per junction from tangents and curvature, and which attachment pairs with which is settled
by a global minimum-cost matching over all of them rather than greedily.

The pieces
----------
1. **Thinning** to a one-pixel medial axis (Zhang-Suen). Implemented here rather than taken from
   ``cv2.ximgproc``, which ships only in opencv-contrib; this project depends on the headless
   package deliberately, and adding contrib for one function is not worth the install size.
2. **Classifying** skeleton pixels by neighbour count into ends, path and junctions.
3. **Arcs**: what is left when junction pixels are removed, each ordered from one terminal to the
   other so a tangent can be measured at either end.
4. **Junction reading**: a junction whose incident arcs pair off into nearly-opposite couples is a
   crossing, and each couple is one connector passing through. A junction that does not pair off is
   a branch, where several connectors share a stem, and its branching is kept rather than being
   rewritten as two curves that happen to cross.
5. **Routing** from attachment to attachment, obeying those rules at every junction.
6. **Matching** the routes so each attachment is used once, by exact minimum-cost assignment.

Cycles, reciprocal pairs, several connectors on one node and curved connectors all fall out of this
rather than needing special cases: nothing here assumes a tree, a direction or a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: A spur shorter than this many pixels, ending in mid-air, is thinning debris rather than a
#: connector. A solid arrowhead thins to a short stub with one or two whiskers, and each whisker
#: would otherwise read as its own arc and turn the head into a junction.
SPUR_MAX_LENGTH = 6

#: How many pixels along an arc are used to measure its direction at one end. Long enough to be
#: steady on a compressed edge, short enough to describe the arc where it meets the junction.
TANGENT_WINDOW = 7

#: Two arc-ends continue into each other when their outward tangents are this close to opposite.
#: -0.55 is about 123 degrees or more of turn between them, which accepts a connector crossing at a
#: shallow angle while refusing a right-angle turn onto a different curve.
CONTINUATION_COSINE = -0.55

#: Added to a pairing's cost per radian of disagreement between the two arcs' bend. Two halves of
#: one smooth curve bend the same way; an arc from a different connector usually does not.
CURVATURE_WEIGHT = 0.8

#: Cost of leaving an attachment unmatched. Higher than any plausible route, so the matching only
#: gives up on an attachment when it truly cannot be routed.
UNMATCHED_COST = 100.0

#: Subtracted from a route's cost when exactly one of its two ends looks like an arrowhead. A
#: connector has one head and one tail, so a pairing that produces that is more likely right than
#: one joining two heads or two tails.
HEAD_EVIDENCE_BONUS = 1.5

#: Above this many attachments the exact matching is skipped for a greedy pass. 14 attachments is
#: 2^14 subsets, which is still milliseconds; no dev chart comes close.
EXACT_MATCH_LIMIT = 14

#: Two junctions joined by an arc no longer than this are one junction that thinning split in two.
#: Where two strokes several pixels wide cross, the medial axis of the overlap is not a point but a
#: short bar, so an X arrives as two three-way junctions with a stub between them - and a three-way
#: junction has an odd number of ends, so it can never pair off and every crossing was being read as
#: a branch. Merging them back is what makes the crossing test meaningful.
JUNCTION_MERGE_LENGTH = 6


# --------------------------------------------------------------------------- #
# 1. Thinning
# --------------------------------------------------------------------------- #


def _neighbours(padded: np.ndarray) -> list[np.ndarray]:
    """The eight neighbours of every pixel, clockwise from north: P2..P9 of Zhang-Suen."""
    return [
        padded[0:-2, 1:-1],  # P2 north
        padded[0:-2, 2:],  # P3 north-east
        padded[1:-1, 2:],  # P4 east
        padded[2:, 2:],  # P5 south-east
        padded[2:, 1:-1],  # P6 south
        padded[2:, 0:-2],  # P7 south-west
        padded[1:-1, 0:-2],  # P8 west
        padded[0:-2, 0:-2],  # P9 north-west
    ]


def thin(mask: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Reduce a mask to a one-pixel-wide medial axis, preserving connectivity.

    Zhang-Suen: two sub-iterations per pass, each deleting boundary pixels whose removal cannot
    break the shape apart or shorten a line's end. A pixel goes only if it has between two and six
    neighbours, exactly one 0-to-1 transition going round it - which is what guarantees it is not a
    connectivity bridge - and it is on the side of the boundary that this sub-iteration is peeling.

    Vectorised over the whole image: the textbook form tests one pixel at a time, which on a
    900x300 chart is tens of thousands of Python-level iterations per pass.
    """
    image = (mask > 0).astype(np.uint8)
    for _ in range(max_iterations):
        removed = 0
        for sub_iteration in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p = _neighbours(padded)
            filled = sum(int_array.astype(np.int16) for int_array in p)

            # Transitions from 0 to 1 walking P2..P9 and back to P2. Exactly one means the pixel is
            # on a simple boundary rather than joining two parts of the shape.
            sequence = [*p, p[0]]
            transitions = sum(
                ((sequence[index] == 0) & (sequence[index + 1] == 1)).astype(np.int16) for index in range(8)
            )

            candidate = (image == 1) & (filled >= 2) & (filled <= 6) & (transitions == 1)
            if sub_iteration == 0:
                candidate &= (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                candidate &= (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)

            count = int(candidate.sum())
            if count:
                image[candidate] = 0
                removed += count
        if removed == 0:
            break
    return image * 255


# --------------------------------------------------------------------------- #
# 2 and 3. The skeleton as a graph of arcs
# --------------------------------------------------------------------------- #

_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(slots=True)
class Arc:
    """One run of skeleton between two terminals, ordered from *first* to *last*.

    A terminal is either a junction cluster id, or ``None`` for a free end sitting in mid-air or at
    a node's border.
    """

    arc_id: int
    points: list[tuple[int, int]]
    first_junction: int | None
    last_junction: int | None

    @property
    def length(self) -> int:
        return len(self.points)

    def end_point(self, at_first: bool) -> tuple[int, int]:
        return self.points[0] if at_first else self.points[-1]

    def tangent(self, at_first: bool) -> np.ndarray:
        """Unit vector along the arc at the chosen end, as ``(dy, dx)``.

        It points from the arc's interior *towards* that end, so at an end that meets a junction it
        points into the junction. Two arcs that continue through a junction into each other
        therefore have opposing tangents there, which is what :func:`_pair_cost` tests.

        ``(dy, dx)`` and not ``(dx, dy)``: arc points are image coordinates in row-major order, and
        silently swapping them here would put the axes of every tangent, bend and dot product in
        this module at odds with the points they came from.
        """
        window = min(TANGENT_WINDOW, len(self.points) - 1)
        if window <= 0:
            return np.array([0.0, 0.0])
        end = self.points[0] if at_first else self.points[-1]
        inner = self.points[window] if at_first else self.points[-1 - window]
        vector = np.array([float(end[0] - inner[0]), float(end[1] - inner[1])])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-6 else np.array([0.0, 0.0])

    def bend(self, at_first: bool) -> float:
        """Signed turn per pixel near the chosen end, as a rough curvature.

        Signed, so two halves of one curve bending the same way can be told from an arc that bends
        the other way and merely happens to line up.
        """
        window = min(TANGENT_WINDOW, (len(self.points) - 1) // 2)
        if window <= 1:
            return 0.0
        points = self.points if at_first else list(reversed(self.points))
        near = np.array(points[window], dtype=float) - np.array(points[0], dtype=float)
        far = np.array(points[2 * window], dtype=float) - np.array(points[window], dtype=float)
        if np.linalg.norm(near) < 1e-6 or np.linalg.norm(far) < 1e-6:
            return 0.0
        cross = float(near[0] * far[1] - near[1] * far[0])
        dot = float(near[0] * far[0] + near[1] * far[1])
        return float(np.arctan2(cross, dot)) / window


@dataclass(slots=True)
class SkeletonGraph:
    """Arcs, and which arc-ends meet at each junction."""

    arcs: dict[int, Arc] = field(default_factory=dict)
    #: junction id -> centroid
    junctions: dict[int, tuple[float, float]] = field(default_factory=dict)
    #: junction id -> [(arc id, whether it is that arc's first end)]
    incident: dict[int, list[tuple[int, bool]]] = field(default_factory=dict)

    @property
    def free_ends(self) -> list[tuple[int, bool]]:
        """Arc-ends that meet no junction: the skeleton's tips."""
        ends: list[tuple[int, bool]] = []
        for arc in self.arcs.values():
            if arc.first_junction is None:
                ends.append((arc.arc_id, True))
            if arc.last_junction is None:
                ends.append((arc.arc_id, False))
        return ends


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad((skeleton > 0).astype(np.int16), 1, mode="constant")
    return sum(_neighbours(padded))


def _crossing_number(skeleton: np.ndarray) -> np.ndarray:
    """Number of separate neighbour groups around each pixel: 1 at a tip, 2 on a path, 3+ at a fork.

    This, and not the raw neighbour count, is what identifies a junction. A thinned diagonal runs as
    a staircase, and a staircase pixel has three neighbours that are all touching each other - one
    group, not three branches. Counting neighbours marked most of every diagonal as a junction, and
    the merge step downstream then fused a whole limb of a Y into one enormous junction, leaving
    single-pixel arcs and a stem nobody could identify.

    It is the same quantity as Zhang-Suen's ``A(P1)``: transitions from empty to filled walking the
    eight neighbours in order.
    """
    padded = np.pad((skeleton > 0).astype(np.int16), 1, mode="constant")
    ring = _neighbours(padded)
    sequence = [*ring, ring[0]]
    return sum(((sequence[index] == 0) & (sequence[index + 1] == 1)).astype(np.int16) for index in range(8))


def _order_run(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Put one run of skeleton pixels into path order.

    Starts at a pixel with a single neighbour inside the run - a tip - falling back to any pixel
    when the run is a closed loop, which happens on a self-returning connector.
    """
    if not points:
        return []
    degree = {point: sum(1 for dy, dx in _OFFSETS if (point[0] + dy, point[1] + dx) in points) for point in points}
    tips = [point for point, count in degree.items() if count <= 1]
    start = min(tips) if tips else min(points)

    ordered = [start]
    seen = {start}
    while True:
        current = ordered[-1]
        nxt = None
        for dy, dx in _OFFSETS:
            candidate = (current[0] + dy, current[1] + dx)
            if candidate in points and candidate not in seen:
                nxt = candidate
                break
        if nxt is None:
            break
        ordered.append(nxt)
        seen.add(nxt)
    return ordered


def _solid_block(skeleton: np.ndarray) -> np.ndarray:
    """Pixels belonging to a fully filled two-by-two square.

    In a thinned image such a square is the signature of a crossing that thinning could not reduce
    any further, and it is the one junction shape the crossing number cannot see: the four pixels are
    all neighbours of one another, so each of them has a single neighbour group.

    Measured on the three shapes this has to distinguish. A T has one pixel of crossing number 3 and
    no block; a Y likewise; an **X has no pixel of crossing number 3 at all** and one block. Taking
    either signal alone misses one of the three - and taking raw neighbour count instead marks 134
    pixels on the Y, because a thinned diagonal is a staircase whose every pixel has four neighbours.
    """
    body = (skeleton > 0).astype(np.uint8)
    padded = np.pad(body, ((0, 1), (0, 1)), mode="constant").astype(bool)
    square = padded[:-1, :-1] & padded[:-1, 1:] & padded[1:, :-1] & padded[1:, 1:]
    marked = np.zeros(body.shape, dtype=bool)
    ys, xs = np.nonzero(square)
    for dy in (0, 1):
        for dx in (0, 1):
            rows = np.clip(ys + dy, 0, body.shape[0] - 1)
            cols = np.clip(xs + dx, 0, body.shape[1] - 1)
            marked[rows, cols] = True
    return marked & (body > 0)


def _junction_seeds(skeleton: np.ndarray) -> np.ndarray:
    """Where the skeleton forks: three or more neighbour groups, or an unreducible block."""
    body = skeleton > 0
    return (body & ((_crossing_number(skeleton) >= 3) | _solid_block(skeleton))).astype(np.uint8)


_COLLAR = np.ones((3, 3), np.uint8)


def _cut_mask(skeleton: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """The junction pixels plus a one-pixel collar: what has to be removed to separate the arcs.

    Removing only the fork pixel does not separate anything. Its neighbours are all mutually
    eight-adjacent, so on a T the left arm, the right arm and the stem stay joined around the hole
    and the skeleton comes back as a single arc with no junction attached to it - the graph has no
    structure at all. Cutting with a collar separates them, at a cost of a pixel or two off each
    arc's end, which is nothing against arcs tens of pixels long.
    """
    return ((cv2.dilate(seeds, _COLLAR) > 0) & (skeleton > 0)).astype(np.uint8)


def _merge_split_junctions(skeleton: np.ndarray, merge_length: int) -> np.ndarray:
    """Grow the junction mask over the short stubs that split one crossing into two.

    Returns a mask of pixels to treat as junction: the forks themselves, plus any arc shorter than
    *merge_length* that runs between two of them.
    """
    body = skeleton > 0
    junction_mask = _junction_seeds(skeleton)
    cut = _cut_mask(skeleton, junction_mask)
    arc_mask = (body & (cut == 0)).astype(np.uint8)

    # Labelled from the *cut* mask, not from the bare seeds. The arcs were carved out with a collar,
    # so an arc's pixels sit two away from the seed they belong to; looking for seeds in their
    # neighbourhood found nothing and no merge ever happened. A 45-degree crossing came back as two
    # three-way branches with a five-pixel stub between them, and a three-way junction cannot pair.
    junction_count, junction_labels = cv2.connectedComponents(cut, connectivity=8)
    if junction_count <= 2:
        return junction_mask

    arc_count, arc_labels = cv2.connectedComponents(arc_mask, connectivity=8)
    for label in range(1, arc_count):
        ys, xs = np.nonzero(arc_labels == label)
        if ys.size == 0 or ys.size > merge_length:
            continue
        touching: set[int] = set()
        for y, x in zip(ys, xs, strict=True):
            for dy, dx in _OFFSETS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    found = int(junction_labels[ny, nx])
                    if found > 0:
                        touching.add(found)
        if len(touching) >= 2:
            junction_mask[ys, xs] = 1
    return junction_mask


def build_graph(skeleton: np.ndarray, merge_length: int = JUNCTION_MERGE_LENGTH) -> SkeletonGraph:
    """Split a skeleton into arcs between its junctions and tips."""
    body = skeleton > 0
    seeds = _merge_split_junctions(skeleton, merge_length)
    junction_mask = _cut_mask(skeleton, seeds)
    arc_mask = (body & (junction_mask == 0)).astype(np.uint8)

    junction_count, junction_labels = cv2.connectedComponents(junction_mask, connectivity=8)
    graph = SkeletonGraph()
    for label in range(1, junction_count):
        ys, xs = np.nonzero(junction_labels == label)
        graph.junctions[label] = (float(xs.mean()), float(ys.mean()))
        graph.incident[label] = []

    arc_count, arc_labels = cv2.connectedComponents(arc_mask, connectivity=8)
    for label in range(1, arc_count):
        ys, xs = np.nonzero(arc_labels == label)
        ordered = _order_run({(int(y), int(x)) for y, x in zip(ys, xs, strict=True)})
        if not ordered:
            continue
        arc = Arc(arc_id=label, points=ordered, first_junction=None, last_junction=None)
        for at_first in (True, False):
            y, x = arc.end_point(at_first)
            for dy, dx in _OFFSETS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    junction = int(junction_labels[ny, nx])
                    if junction > 0:
                        if at_first:
                            arc.first_junction = junction
                        else:
                            arc.last_junction = junction
                        graph.incident[junction].append((label, at_first))
                        break
        graph.arcs[label] = arc
    return graph


def prune_spurs(skeleton: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Remove short whiskers that thinning leaves on a thick blob.

    An arrowhead is a solid triangle, and thinning it leaves a stub with one or two whiskers a few
    pixels long. Left in place, each whisker is an arc and the head becomes a junction, so a
    connector arriving at a node looks like three curves meeting. Only whiskers that end in mid-air
    are removed, and *keep* marks places - the node attachment bands - where a free end is expected
    and must survive.
    """
    current = skeleton.copy()
    for _ in range(3):
        graph = build_graph(current)
        doomed: list[tuple[int, int]] = []
        for arc in graph.arcs.values():
            free_first = arc.first_junction is None
            free_last = arc.last_junction is None
            if free_first == free_last:
                # Either a whole isolated arc or a link between two junctions; neither is a whisker.
                continue
            if arc.length > SPUR_MAX_LENGTH:
                continue
            tip = arc.end_point(free_first)
            if keep is not None and keep[tip[0], tip[1]]:
                continue
            doomed.extend(arc.points)
        if not doomed:
            break
        for y, x in doomed:
            current[y, x] = 0
    return current


# --------------------------------------------------------------------------- #
# 4 and 5. Reading a junction, and routing through it
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Junction:
    """How traffic passes through one junction."""

    junction_id: int
    kind: str
    #: For a crossing: arc-end -> the arc-end it continues into.
    continues: dict[tuple[int, bool], tuple[int, bool]] = field(default_factory=dict)
    #: For a branch: the shared stem, and the ends that branch off it.
    stem: tuple[int, bool] | None = None
    branches: list[tuple[int, bool]] = field(default_factory=list)


def _pair_cost(graph: SkeletonGraph, left: tuple[int, bool], right: tuple[int, bool]) -> float | None:
    """Cost of treating two arc-ends as one connector passing through. ``None`` if they cannot be."""
    left_tangent = graph.arcs[left[0]].tangent(left[1])
    right_tangent = graph.arcs[right[0]].tangent(right[1])
    if np.linalg.norm(left_tangent) < 1e-6 or np.linalg.norm(right_tangent) < 1e-6:
        return None
    alignment = float(np.dot(left_tangent, right_tangent))
    if alignment > CONTINUATION_COSINE:
        return None
    # Straight through costs nothing; the more the pair has to kink, the more it costs.
    cost = 1.0 + alignment
    # Bend is measured outward at each end, so one smooth curve has bends of opposite sign.
    left_bend = graph.arcs[left[0]].bend(left[1])
    right_bend = graph.arcs[right[0]].bend(right[1])
    cost += CURVATURE_WEIGHT * abs(left_bend + right_bend)
    return cost


def _best_pairing(
    graph: SkeletonGraph, ends: list[tuple[int, bool]]
) -> tuple[dict[tuple[int, bool], tuple[int, bool]], float] | None:
    """The cheapest way to pair every arc-end at a junction with another. ``None`` if impossible."""
    if len(ends) % 2 != 0:
        return None

    costs: dict[tuple[int, int], float] = {}
    for first in range(len(ends)):
        for second in range(first + 1, len(ends)):
            cost = _pair_cost(graph, ends[first], ends[second])
            if cost is not None:
                costs[(first, second)] = cost

    best: tuple[list[tuple[int, int]], float] | None = None

    def recurse(remaining: tuple[int, ...], chosen: list[tuple[int, int]], total: float) -> None:
        nonlocal best
        if best is not None and total >= best[1]:
            return
        if not remaining:
            best = (list(chosen), total)
            return
        head, rest = remaining[0], remaining[1:]
        for index, other in enumerate(rest):
            key = (head, other) if head < other else (other, head)
            if key not in costs:
                continue
            chosen.append(key)
            recurse(rest[:index] + rest[index + 1 :], chosen, total + costs[key])
            chosen.pop()

    recurse(tuple(range(len(ends))), [], 0.0)
    if best is None:
        return None
    mapping: dict[tuple[int, bool], tuple[int, bool]] = {}
    for first, second in best[0]:
        mapping[ends[first]] = ends[second]
        mapping[ends[second]] = ends[first]
    return mapping, best[1]


def read_junctions(graph: SkeletonGraph) -> dict[int, Junction]:
    """Decide, for each junction, whether it is a crossing or a branch.

    A crossing is where two connectors happen to overlap: its arc-ends pair off into couples that
    each continue smoothly through, and traffic entering on one leaves on its partner. A branch is
    where connectors genuinely share a stem, which is what several arrows leaving one node looks
    like a few pixels out from the border; its branching is kept.

    The test is whether a complete pairing exists at all. If every end has a partner it continues
    into, the junction is a crossing; if some end has none - most obviously when there is an odd
    number of them - it is a branch, and the stem is the end that best opposes all the others.
    """
    result: dict[int, Junction] = {}
    for junction_id, ends in graph.incident.items():
        if len(ends) < 3:
            # Two ends meeting is not a decision; thinning sometimes leaves one.
            paired = _best_pairing(graph, ends)
            result[junction_id] = Junction(
                junction_id=junction_id,
                kind="crossing",
                continues=paired[0] if paired else {},
            )
            continue

        paired = _best_pairing(graph, ends)
        if paired is not None:
            result[junction_id] = Junction(junction_id=junction_id, kind="crossing", continues=paired[0])
            continue

        # No complete pairing: a genuine branch. The stem is the end whose direction most opposes
        # the others taken together.
        best_stem: tuple[int, bool] | None = None
        best_score = -2.0
        for candidate in ends:
            others = [graph.arcs[a].tangent(f) for a, f in ends if (a, f) != candidate]
            if not others:
                continue
            total = np.sum(others, axis=0)
            norm = float(np.linalg.norm(total))
            if norm < 1e-6:
                continue
            score = -float(np.dot(graph.arcs[candidate[0]].tangent(candidate[1]), total / norm))
            if score > best_score:
                best_score, best_stem = score, candidate
        result[junction_id] = Junction(
            junction_id=junction_id,
            kind="branch",
            stem=best_stem,
            branches=[end for end in ends if end != best_stem],
        )
    return result


def _other_end(arc: Arc, at_first: bool) -> tuple[int, bool]:
    return (arc.arc_id, not at_first)


def routes_between_ends(
    graph: SkeletonGraph,
    junctions: dict[int, Junction],
    start: tuple[int, bool],
    max_steps: int = 64,
) -> list[tuple[tuple[int, bool], float, int]]:
    """Every arc-end reachable from *start* by obeying the junction rules.

    Returns ``(end reached, accumulated cost, pixels travelled)``. A branch junction produces
    several results, one per branch, which is how two arrows sharing a stem both get found. Travel
    never revisits an arc in the same direction, so a cycle terminates.
    """
    found: list[tuple[tuple[int, bool], float, int]] = []
    stack: list[tuple[tuple[int, bool], float, int, frozenset[int]]] = [(start, 0.0, 0, frozenset())]

    while stack:
        entry, cost, travelled, visited = stack.pop()
        if len(visited) > max_steps:
            continue
        arc = graph.arcs.get(entry[0])
        if arc is None:
            continue
        # Traverse this arc from `entry` to its far end.
        exit_end = _other_end(arc, entry[1])
        travelled += arc.length
        visited = visited | {arc.arc_id}

        junction_id = arc.last_junction if entry[1] else arc.first_junction
        junction = junctions.get(junction_id) if junction_id is not None else None
        if junction is None:
            # Ran out of skeleton: a tip, which is where an attachment lives.
            found.append((exit_end, cost, travelled))
            continue

        # Which ends may traffic leave this junction on?
        departures: list[tuple[tuple[int, bool], float]] = []
        if junction.kind == "crossing":
            partner = junction.continues.get(exit_end)
            if partner is not None:
                departures.append((partner, _pair_cost(graph, exit_end, partner) or 0.0))
        elif exit_end == junction.stem:
            departures.extend((branch, 0.5) for branch in junction.branches)
        elif junction.stem is not None:
            departures.append((junction.stem, 0.5))

        # An arc already travelled is not offered again, so a loop terminates. The check is on
        # departures rather than on arrival, so a route that cannot go on is still reported as
        # having arrived rather than vanishing.
        onward = [(departure, extra) for departure, extra in departures if departure[0] not in visited]
        if not onward:
            found.append((exit_end, cost, travelled))
            continue
        for departure, extra in onward:
            stack.append((departure, cost + extra, travelled, visited))
    return found


# --------------------------------------------------------------------------- #
# 6. Global matching
# --------------------------------------------------------------------------- #


def match_pairs(count: int, costs: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    """Cheapest set of pairs covering as many items as possible, each item used at most once.

    Exact, by dynamic programming over subsets. Greedy nearest-first was what the previous two
    attempts used and it is what let three attachments claim the same fourth one; with a global
    optimum that cannot happen, because using an attachment twice is not a legal assignment.

    Leaving an item out costs :data:`UNMATCHED_COST`, so an attachment is abandoned only when it
    genuinely cannot be paired.
    """
    if count == 0:
        return []
    if count > EXACT_MATCH_LIMIT:
        # Far beyond anything this corpus draws; fall back rather than allocate 2^n.
        order = sorted(costs.items(), key=lambda item: item[1])
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for (first, second), _cost in order:
            if first in used or second in used:
                continue
            used.add(first)
            used.add(second)
            pairs.append((first, second))
        return pairs

    size = 1 << count
    best = np.full(size, np.inf)
    # Both the predecessor state and what was decided, because reconstructing the step from the
    # state alone gets it wrong: a skip consumes the lowest *unset* bit of the predecessor, which is
    # not the lowest set bit of the successor.
    parent: list[int] = [-1] * size
    decision: list[tuple[int, int] | None] = [None] * size
    best[0] = 0.0
    for state in range(size):
        if not np.isfinite(best[state]):
            continue
        # The lowest unused item must be dealt with, either paired or abandoned.
        first = next((index for index in range(count) if not state & (1 << index)), None)
        if first is None:
            continue
        skipped = state | (1 << first)
        if best[state] + UNMATCHED_COST < best[skipped]:
            best[skipped] = best[state] + UNMATCHED_COST
            parent[skipped] = state
            decision[skipped] = None
        for second in range(first + 1, count):
            if state & (1 << second):
                continue
            cost = costs.get((first, second))
            if cost is None:
                continue
            nxt = state | (1 << first) | (1 << second)
            if best[state] + cost < best[nxt]:
                best[nxt] = best[state] + cost
                parent[nxt] = state
                decision[nxt] = (first, second)

    pairs: list[tuple[int, int]] = []
    state = size - 1
    while state and parent[state] >= 0:
        picked = decision[state]
        if picked is not None:
            pairs.append(picked)
        state = parent[state]
    pairs.reverse()
    return pairs


__all__ = [
    "CONTINUATION_COSINE",
    "SPUR_MAX_LENGTH",
    "TANGENT_WINDOW",
    "Arc",
    "Junction",
    "SkeletonGraph",
    "build_graph",
    "match_pairs",
    "prune_spurs",
    "read_junctions",
    "routes_between_ends",
    "thin",
]

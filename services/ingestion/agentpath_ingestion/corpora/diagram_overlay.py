"""Draw what the parser saw, so a human can check it against the original.

This is the deliverable that makes the rest auditable. Every claim the parser makes is drawn on
top of the image it came from: the node boundary it found, the text it read, each connector, which
end carries the arrowhead, how that was decided, and how it was resolved.

Three outcomes are drawn differently on purpose, because they need different judgement:

* **orange, solid, with an arrow** - both ends and the direction established.
* **blue, solid, no arrow** - the two nodes are connected, but which end carries the arrowhead
  could not be measured. The adjacency is real; the direction is not claimed.
* **red, dashed** - the connector was found but what it joins was not established.

Traced edges are marked ``~`` because they came out of a crossing and deserve a closer look than
a connector that was its own component.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agentpath_ingestion.corpora.diagram import DiagramParse

#: BGR. Chosen to stay legible over any of the backgrounds this corpus uses, including white and
#: near-black, which is why everything is drawn with a contrasting outline as well as a fill.
NODE_COLOUR = (0, 200, 0)
NODE_TEXT_COLOUR = (0, 120, 0)
#: Orange. Written as BGR, which is the order OpenCV takes: (0, 165, 255) is orange and
#: (255, 165, 0) is azure. Worth stating because the earlier version had them the wrong way round,
#: so "orange=directed" in the legend drew blue and directed edges were indistinguishable from
#: undirected ones - the overlay was quietly useless for the one thing it existed to show.
DIRECTED_COLOUR = (0, 165, 255)
#: Purple, chosen to be unmistakable against orange rather than a shade of it.
UNDIRECTED_COLOUR = (200, 0, 200)
AMBIGUOUS_COLOUR = (0, 0, 255)
HEAD_COLOUR = (0, 215, 255)
TAIL_COLOUR = (180, 180, 180)
PANEL_BG = (20, 20, 20)
PANEL_FG = (240, 240, 240)

#: The overlay is rendered at this scale so small text stays readable when a reviewer opens it.
SCALE = 2

#: Height of the summary strip along the bottom.
PANEL_HEIGHT = 118


def _put(
    canvas: np.ndarray, text: str, origin: tuple[int, int], colour: tuple[int, int, int], scale: float = 0.4
) -> None:
    """Text with a dark halo, so it is readable over any background."""
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _dashed_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    dash: int = 8,
) -> None:
    """A dashed segment, used only for connectors whose ends are not established."""
    x1, y1 = start
    x2, y2 = end
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for offset in range(0, length, dash * 2):
        t0 = offset / length
        t1 = min((offset + dash) / length, 1.0)
        p0 = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
        p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        cv2.line(canvas, p0, p1, colour, 2, cv2.LINE_AA)


def render_overlay(parse: DiagramParse, image_path: Path | None = None) -> np.ndarray | None:
    """Build the debug image. Returns ``None`` when the source image cannot be read."""
    source = Path(image_path or parse.image_path)
    image = cv2.imread(str(source))
    if image is None:
        return None

    canvas = cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    # Dim the original so the annotations sit clearly on top of it while the drawing underneath
    # is still comparable to the real thing.
    canvas = cv2.addWeighted(canvas, 0.55, np.full_like(canvas, 255), 0.45, 0)

    for node in parse.nodes:
        x, y = node.x * SCALE, node.y * SCALE
        w, h = node.width * SCALE, node.height * SCALE
        cv2.rectangle(canvas, (x, y), (x + w, y + h), NODE_COLOUR, 2, cv2.LINE_AA)
        _put(canvas, f"{node.node_id} [{node.shape.value[:4]}]", (x + 2, max(y - 4, 10)), NODE_COLOUR, 0.42)
        if node.text:
            confidence = f" {node.text_confidence:.2f}" if node.text_confidence else ""
            _put(canvas, f'"{node.text[:30]}"{confidence}', (x + 2, y + h + 14), NODE_TEXT_COLOUR, 0.38)

    for edge in parse.edges:
        if edge.ambiguous or not (edge.source_id and edge.target_id):
            colour = AMBIGUOUS_COLOUR
        elif edge.directed:
            colour = DIRECTED_COLOUR
        else:
            colour = UNDIRECTED_COLOUR

        tail = (edge.source_point[0] * SCALE, edge.source_point[1] * SCALE) if edge.source_point else None
        head = (edge.target_point[0] * SCALE, edge.target_point[1] * SCALE) if edge.target_point else None

        if head is not None:
            # The head end is the direct evidence for the direction, so it is marked distinctly.
            cv2.circle(canvas, head, 6, HEAD_COLOUR if edge.directed else colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, head, 6, (0, 0, 0), 1, cv2.LINE_AA)
        if tail is not None:
            cv2.circle(canvas, tail, 4, TAIL_COLOUR, -1, cv2.LINE_AA)
            cv2.circle(canvas, tail, 4, (0, 0, 0), 1, cv2.LINE_AA)

        if tail is not None and head is not None:
            if edge.ambiguous:
                _dashed_line(canvas, tail, head, colour)
            elif edge.directed:
                cv2.arrowedLine(canvas, tail, head, colour, 2, cv2.LINE_AA, tipLength=0.08)
            else:
                cv2.line(canvas, tail, head, colour, 2, cv2.LINE_AA)

        anchor = head or tail
        if anchor is None:
            continue
        mark = "~" if edge.resolved_by == "trace" else ""
        description = f"{edge.edge_id}{mark}: {edge.source_id or '?'}->{edge.target_id or '?'}"
        if edge.label:
            description += f" [{edge.label}]"
        if edge.source_id and edge.target_id:
            description += f" ink {edge.source_ink}/{edge.target_ink}"
            description += f" x{edge.direction_ratio:.2f}" if edge.directed else " UNDIRECTED"
        _put(canvas, description, (anchor[0] + 9, anchor[1] - 6), colour, 0.38)

    panel = np.full((PANEL_HEIGHT, canvas.shape[1], 3), PANEL_BG, dtype=np.uint8)
    traced = sum(1 for edge in parse.edges if edge.resolved_by == "trace")
    lines = [
        f"{source.name}   {parse.width}x{parse.height}   bg={parse.background}   "
        f"node threshold={parse.threshold_used}   stroke threshold={parse.stroke_threshold}",
        f"nodes={parse.node_count}   connectors={parse.edge_count}   directed={len(parse.directed_edges)}   "
        f"undirected={len(parse.undirected_only_edges)}   unresolved={len(parse.ambiguous_edges)}   "
        f"traced={traced}   branch nodes={len(parse.branch_node_ids)}",
        "green=node   orange+arrow=direction established   purple=connected, direction not established",
        "RED DASHED=ends not established   yellow dot=arrowhead end   grey dot=tail end   ~=traced out of a crossing",
    ]
    if parse.problems:
        lines.append("problems: " + "; ".join(parse.problems)[: max(canvas.shape[1] // 7, 40)])
    for index, line in enumerate(lines[:5]):
        cv2.putText(panel, line, (8, 18 + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, PANEL_FG, 1, cv2.LINE_AA)

    return np.vstack([canvas, panel])


def write_overlay(parse: DiagramParse, target: Path, image_path: Path | None = None) -> Path | None:
    """Render and save. Returns the path written, or ``None`` when rendering failed."""
    overlay = render_overlay(parse, image_path)
    if overlay is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), overlay)
    return target


__all__ = ["PANEL_HEIGHT", "SCALE", "render_overlay", "write_overlay"]

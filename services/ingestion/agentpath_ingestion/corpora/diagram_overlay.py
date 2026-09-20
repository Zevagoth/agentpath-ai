"""Draw what the parser saw, so a human can check it against the original.

This is the deliverable that makes the rest auditable. Every claim the parser makes is drawn on
top of the image it came from: the node boundary it found, the text it read, each arrowhead, the
direction it inferred, the source it walked back to, and the confidence.

Ambiguity is drawn differently from confidence on purpose. An edge whose source could not be
established is dashed and red, so a reviewer's eye goes to the thing that needs judgement rather
than to the 90% that is fine.
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
EDGE_COLOUR = (255, 140, 0)
AMBIGUOUS_COLOUR = (0, 0, 255)
HEAD_COLOUR = (0, 215, 255)
PANEL_BG = (20, 20, 20)
PANEL_FG = (240, 240, 240)

#: The overlay is rendered at this scale so small text stays readable when a reviewer opens it.
SCALE = 2

#: Height of the summary strip along the bottom.
PANEL_HEIGHT = 96


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
    """A dashed segment, used only for edges whose source is not established."""
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

    by_id = {node.node_id: node for node in parse.nodes}

    for node in parse.nodes:
        x, y = node.x * SCALE, node.y * SCALE
        w, h = node.width * SCALE, node.height * SCALE
        cv2.rectangle(canvas, (x, y), (x + w, y + h), NODE_COLOUR, 2, cv2.LINE_AA)
        label = f"{node.node_id} [{node.shape.value[:4]}]"
        _put(canvas, label, (x + 2, max(y - 4, 10)), NODE_COLOUR, 0.42)
        if node.text:
            confidence = f" {node.text_confidence:.2f}" if node.text_confidence else ""
            _put(canvas, f'"{node.text[:30]}"{confidence}', (x + 2, y + h + 14), NODE_TEXT_COLOUR, 0.38)

    for edge in parse.edges:
        head = (edge.head[0] * SCALE, edge.head[1] * SCALE)
        ambiguous = edge.ambiguous or not edge.source_id
        colour = AMBIGUOUS_COLOUR if ambiguous else EDGE_COLOUR

        # The arrowhead itself: a filled circle, because its position is the direct evidence for
        # the edge's existence and for which end is the head.
        cv2.circle(canvas, head, 6, HEAD_COLOUR, -1, cv2.LINE_AA)
        cv2.circle(canvas, head, 6, (0, 0, 0), 1, cv2.LINE_AA)

        if edge.tail is not None:
            tail = (edge.tail[0] * SCALE, edge.tail[1] * SCALE)
            if ambiguous:
                _dashed_line(canvas, tail, head, colour)
            else:
                cv2.arrowedLine(canvas, tail, head, colour, 2, cv2.LINE_AA, tipLength=0.08)

        source_node = by_id.get(edge.source_id or "")
        target_node = by_id.get(edge.target_id)
        description = (
            f"{edge.edge_id}: {source_node.node_id if source_node else '?'}"
            f"->{target_node.node_id if target_node else '?'}"
        )
        if edge.label:
            description += f" [{edge.label}]"
        if not ambiguous:
            description += f" c={edge.confidence:.2f}"
        _put(canvas, description, (head[0] + 9, head[1] - 6), colour, 0.38)

    panel = np.full((PANEL_HEIGHT, canvas.shape[1], 3), PANEL_BG, dtype=np.uint8)
    lines = [
        f"{source.name}   {parse.width}x{parse.height}   bg={parse.background}   threshold={parse.threshold_used}",
        f"nodes={parse.node_count}   arrows={parse.edge_count}   "
        f"confident={len(parse.confident_edges)}   ambiguous={len(parse.ambiguous_edges)}   "
        f"branch nodes={len(parse.branch_node_ids)}",
        "green=node  orange=directed arrow  RED DASHED=source not established  yellow dot=arrowhead",
    ]
    if parse.problems:
        lines.append("problems: " + "; ".join(parse.problems)[: canvas.shape[1] // 7])
    for index, line in enumerate(lines[:4]):
        cv2.putText(panel, line, (8, 20 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PANEL_FG, 1, cv2.LINE_AA)

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

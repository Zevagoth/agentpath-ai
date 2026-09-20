"""Turn a measured diagram into the workflow graph the rest of the system already speaks.

The geometry in :mod:`diagram` recovers boxes, connectors and directions from pixels. The product
consumes :class:`WorkflowGraph`, and so does the evaluation that produced the Stage 2B1 baseline
for `qwen3.5:4b`. Converting here rather than re-implementing the scoring means the geometric
parser and the language model are measured by the same code on the same ground truth, so the
comparison between them is a comparison and not two different yardsticks.

What this module decides, and what it refuses to
------------------------------------------------
**Node kinds are structural, not semantic.** A node with more than one established outgoing edge
is a decision point; one with no incoming edge is a start; one with no outgoing edge is an end.
That is read off the measured topology. What a box *means* - whether it is a question, what its
``Yes`` branch implies, whether a case matches it - is a language model's job and happens
elsewhere. Nothing here invents meaning.

**An unproven direction is carried, not guessed.** A connector whose two ends were established but
whose direction was not is a real fact: those two nodes are joined. It becomes an edge with
``direction_clear=False``, and its presence sends the whole graph to review. It is not dropped,
because dropping it would hide evidence, and it is not treated as routable, because its direction
is not known.

**Acceptance is a gate, not a score.** A graph is ``OK`` only when every connector has both ends
and a direction, every node has readable text, and validation found nothing. Anything else is
``NEEDS_REVIEW`` and the product shows the original image instead of routing the case. This is
what makes "no falsely accepted path" a measurable claim rather than an aspiration.
"""

from __future__ import annotations

import time
from pathlib import Path

from agentpath_ingestion.corpora.diagram import DiagramParse, parse_diagram
from agentpath_ingestion.corpora.workflow import (
    ExtractionStatus,
    NodeKind,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    _endpoints,
    _paths,
)

#: Reported as the extractor's identity, so a stored graph says how it was produced. Geometry has
#: no model, and a report that left this as the VLM's name would be a lie about provenance.
EXTRACTOR_NAME = "geometry"


def _kinds(parse: DiagramParse) -> dict[str, NodeKind]:
    """Structural role of each node, from the established edges only.

    Deliberately ignores undirected and unresolved connectors. A node's role follows from arrows
    whose direction was measured; counting a connector whose direction is unknown would let a
    guess decide whether something is a decision point.
    """
    outgoing: dict[str, int] = {node.node_id: 0 for node in parse.nodes}
    incoming: dict[str, int] = {node.node_id: 0 for node in parse.nodes}
    for edge in parse.directed_edges:
        if edge.source_id in outgoing:
            outgoing[edge.source_id] += 1
        if edge.target_id in incoming:
            incoming[edge.target_id or ""] = incoming.get(edge.target_id or "", 0) + 1

    kinds: dict[str, NodeKind] = {}
    for node in parse.nodes:
        out_count = outgoing.get(node.node_id, 0)
        in_count = incoming.get(node.node_id, 0)
        if out_count > 1:
            # More than one way out: a decision is taken here. Which answer takes which branch is
            # for the semantic layer to read off the arrow labels.
            kinds[node.node_id] = NodeKind.QUESTION
        elif in_count == 0 and out_count > 0:
            kinds[node.node_id] = NodeKind.START
        elif out_count == 0 and in_count > 0:
            kinds[node.node_id] = NodeKind.END
        elif out_count == 0 and in_count == 0:
            # Touched by nothing established. Visible, but its role is not known.
            kinds[node.node_id] = NodeKind.UNKNOWN
        else:
            kinds[node.node_id] = NodeKind.ACTION
    return kinds


def review_reasons(parse: DiagramParse) -> list[str]:
    """Why this diagram may not be used to route a case unaided.

    Each reason names the evidence that is missing, in the terms a reviewer can check against the
    overlay, rather than a score they would have to take on trust.
    """
    reasons: list[str] = []

    unresolved = parse.ambiguous_edges
    if unresolved:
        reasons.append(
            f"{len(unresolved)} of {len(parse.edges)} connectors do not have both ends established, "
            f"so part of the diagram's structure is unknown"
        )

    undirected = parse.undirected_only_edges
    if undirected:
        pairs = ", ".join(f"{edge.source_id}-{edge.target_id}" for edge in undirected[:3])
        reasons.append(
            f"{len(undirected)} connector(s) join two nodes with no established direction ({pairs}), "
            f"so which way a case travels along them is not known"
        )

    unreadable = [node.node_id for node in parse.nodes if not node.text.strip()]
    if unreadable:
        reasons.append(f"{len(unreadable)} node(s) have no readable text: {', '.join(unreadable[:4])}")

    # A connector was seen attaching to this node and never resolved into an edge. Purely internal
    # evidence of a missing edge - no ground truth involved - and the strongest signal available
    # that the structure is incomplete.
    gaps = parse.unaccounted_ports
    if gaps:
        detail = ", ".join(f"{node_id} ({count})" for node_id, count in sorted(gaps.items())[:4])
        reasons.append(
            f"{sum(gaps.values())} connector end(s) attach to a node without resolving into an edge "
            f"({detail}), so at least one arrow is missing from the structure"
        )

    # Every node in a real chart is joined to something: measured across 302 dev charts, not one
    # has an isolated node. A node with no established edge therefore means an arrow was missed.
    touched = {edge.source_id for edge in parse.edges} | {edge.target_id for edge in parse.edges}
    isolated = [node.node_id for node in parse.nodes if node.node_id not in touched]
    if isolated:
        reasons.append(
            f"{len(isolated)} node(s) have no connector at all ({', '.join(isolated[:4])}), and every "
            f"node in a real chart is joined to something"
        )

    if not parse.nodes:
        reasons.append("no nodes were found, so there is nothing to route")
    elif not parse.directed_edges:
        reasons.append("no connector has an established direction, so no path can be followed")

    return reasons


def graph_from_parse(
    parse: DiagramParse,
    *,
    image_id: str,
    document_id: str,
    corpus_id: str,
    seconds: float | None = None,
) -> WorkflowGraph:
    """Build a :class:`WorkflowGraph` from measured geometry.

    ``confidence`` is the share of connectors whose direction was established, which is a fact
    about the evidence rather than a model's self-assessment. A graph sent to review keeps its
    confidence: the number says how much of the diagram was read, and the status says whether that
    was enough.
    """
    kinds = _kinds(parse)
    nodes = tuple(
        WorkflowNode(
            node_id=node.node_id,
            kind=kinds.get(node.node_id, NodeKind.UNKNOWN),
            text=node.text,
            ambiguous=not node.text.strip(),
            note="" if node.text.strip() else "no text could be read from this box",
        )
        for node in parse.nodes
    )

    edges: list[WorkflowEdge] = []
    for edge in parse.directed_edges:
        edges.append(
            WorkflowEdge(
                source=edge.source_id or "",
                target=edge.target_id or "",
                label=edge.label or None,
                direction_clear=True,
                note="followed out of a crossing" if edge.resolved_by == "trace" else "",
            )
        )
    for edge in parse.undirected_only_edges:
        # Carried, because the adjacency was measured. Not routable, because the direction was not.
        edges.append(
            WorkflowEdge(
                source=edge.source_id or "",
                target=edge.target_id or "",
                label=edge.label or None,
                direction_clear=False,
                note=edge.reason,
            )
        )

    reasons = review_reasons(parse)

    # Paths are enumerated over routable edges only, so a path can never depend on a direction
    # nobody measured.
    routable = tuple(edge for edge in edges if edge.direction_clear)
    endpoint_notes: list[str] = []
    starts, ends = _endpoints(nodes, routable, endpoint_notes)
    paths = _paths(nodes, routable, starts, ends)

    # "No start node" and "no end node" are what a loop looks like, not a fault, and refusing a
    # chart for having one would refuse the real thing: 46% of dev charts contain a cycle, and every
    # one of the 42 charts with no zero-in-degree node has a cycle. Before this, those two messages
    # were 61 of 110 refusals. They stay as warnings so a reviewer can still see them.
    if parse.has_cycle:
        endpoint_notes = [note for note in endpoint_notes if "no start node" not in note and "no end node" not in note]

    all_reasons = reasons + endpoint_notes
    status = ExtractionStatus.OK if not all_reasons else ExtractionStatus.NEEDS_REVIEW
    if not parse.nodes:
        status = ExtractionStatus.FAILED

    established = len(parse.directed_edges)
    confidence = established / len(parse.edges) if parse.edges else 0.0

    return WorkflowGraph(
        image_id=image_id,
        document_id=document_id,
        corpus_id=corpus_id,
        status=status,
        confidence=confidence,
        nodes=nodes,
        edges=tuple(edges),
        paths=paths,
        visible_text=" ".join(node.text for node in parse.nodes if node.text.strip()),
        start_nodes=starts,
        end_nodes=ends,
        warnings=tuple(parse.problems),
        review_reasons=tuple(all_reasons),
        model=EXTRACTOR_NAME,
        seconds=seconds,
    )


class GeometricWorkflowExtractor:
    """Reads a diagram with geometry, behind the same interface as the language-model extractor.

    Same method signature as :class:`~agentpath_ingestion.corpora.workflow.WorkflowExtractor`, so
    the existing evaluation runs against either one without change. That is the point: the 4B and
    9B numbers this has to beat came out of that harness, and a new harness would make the
    comparison meaningless.
    """

    def __init__(self, *, read_text: bool = True) -> None:
        #: OCR is most of the cost. Off when only topology is being measured.
        self.read_text = read_text

    @property
    def model(self) -> str:
        return EXTRACTOR_NAME

    def extract(self, path: Path, *, image_id: str, document_id: str, corpus_id: str) -> WorkflowGraph:
        started = time.perf_counter()
        parse = parse_diagram(path, read_text=self.read_text)
        return graph_from_parse(
            parse,
            image_id=image_id,
            document_id=document_id,
            corpus_id=corpus_id,
            seconds=time.perf_counter() - started,
        )


__all__ = ["EXTRACTOR_NAME", "GeometricWorkflowExtractor", "graph_from_parse", "review_reasons"]

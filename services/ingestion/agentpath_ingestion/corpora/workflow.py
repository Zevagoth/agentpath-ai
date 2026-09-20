"""Turn a decision flowchart into something searchable, or refuse to.

The product problem: a workflow diagram asks "did the customer make contact?", "supplier
or hotel?", "does the booking exist?", and routes the agent down a branch. Handing them
the picture makes them trace it under time pressure. Understanding it once, at ingestion,
lets the system say "your case is path B, here are the three steps".

The hard rule, and the reason most of this module is about refusing
--------------------------------------------------------------------
**The image is the source of truth. The model's reading of it is not.** Everything
extracted here carries the image id and the document it came from, and the console shows
the original alongside, because an agent acting on a misread arrow is worse off than an
agent reading the diagram themselves.

So nothing is invented. A node that is not visible is not created. An arrow whose
direction is unclear is not given one. When the model's answer is incomplete, contradicts
itself, or references something it did not extract, the result is
:attr:`ExtractionStatus.NEEDS_REVIEW` with the reason attached and the original image
shown - not a plausible-looking graph.

The checks below are structural, and they are the point of this module: a graph that
passes them is still only as good as the model's eyesight, but a graph that fails them is
*certainly* wrong, and saying so is cheap.

Status on this corpus
---------------------
MRAMG-Manual contains no decision flowcharts. 138 of its 2,599 images were examined -
every image whose surrounding text asks a question and offers yes/no, every image with
flow vocabulary at 0.25 MP or larger, the 15 largest images, and 60 drawn at random from
those the prefilter rejected. Three were labelled ``workflow_diagram`` by the model and
all three were inspected by hand: a symptom-by-check lookup matrix, a keycap assembly
sequence, and a numbered CPU-install sequence. None branches.

This module is therefore written and tested but has no corpus to be measured against, and
accuracy numbers for it are not claimed. See ``docs/IMAGE_ROLES.md``.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

WORKFLOWS_NAME = "workflows.jsonl"

#: Below this the extraction is reported but marked for review regardless of structure.
#: A model that is unsure about a decision tree is a model whose tree an agent should not
#: act on without looking at the picture.
MIN_TRUSTED_CONFIDENCE = 0.6

#: A flowchart with one node is not a flowchart, and neither is one with no edges.
MIN_NODES = 2
MIN_EDGES = 1


class NodeKind(StrEnum):
    """What a box in the diagram does."""

    START = "start"
    QUESTION = "question"
    """Has outgoing edges labelled with the answers that take them."""
    ACTION = "action"
    CONDITION = "condition"
    END = "end"
    UNKNOWN = "unknown"
    """The box is visible but what it does is not. Forces review."""


class ExtractionStatus(StrEnum):
    """Whether the extracted graph may be used to route a case."""

    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    """Structurally broken, low confidence, or ambiguous. The image is shown instead."""
    FAILED = "failed"
    """Nothing usable came back at all."""


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One box, exactly as it appears."""

    node_id: str
    kind: NodeKind
    #: The box's own words, verbatim. Never paraphrased - an agent compares this against
    #: the picture.
    text: str
    #: For a question node, the options its outgoing edges are labelled with.
    options: tuple[str, ...] = ()
    #: For a condition node, the condition as written.
    condition: str | None = None
    #: For an action node, what the agent should do.
    action: str | None = None
    ambiguous: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "text": self.text,
            "options": list(self.options),
            "condition": self.condition,
            "action": self.action,
            "ambiguous": self.ambiguous,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class WorkflowEdge:
    """One arrow. Direction is required and is never guessed.

    ``direction_clear`` exists because an unlabelled line between two boxes is common in
    scanned diagrams, and an edge pointed the wrong way sends an agent down the opposite
    branch. When it is false the whole extraction needs review.
    """

    source: str
    target: str
    #: The answer or condition written on the arrow, when there is one.
    label: str | None = None
    direction_clear: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "direction_clear": self.direction_clear,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class WorkflowPath:
    """One route from a start node to an end node, with what it takes to get there."""

    path_id: str
    node_ids: tuple[str, ...]
    #: The edge labels along the way: the conditions that select this path.
    conditions: tuple[str, ...]
    #: The action nodes on this path, in order. What the agent is finally told to do.
    actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "node_ids": list(self.node_ids),
            "conditions": list(self.conditions),
            "actions": list(self.actions),
        }


@dataclass(frozen=True, slots=True)
class WorkflowGraph:
    """A whole extracted diagram, and whether it can be trusted."""

    #: Where it came from. Both required: a routing decision has to be traceable to a
    #: picture a human can open.
    image_id: str
    document_id: str
    corpus_id: str
    status: ExtractionStatus
    confidence: float
    nodes: tuple[WorkflowNode, ...] = ()
    edges: tuple[WorkflowEdge, ...] = ()
    paths: tuple[WorkflowPath, ...] = ()
    #: Every word the model could read, for the searchable index.
    visible_text: str = ""
    start_nodes: tuple[str, ...] = ()
    end_nodes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    model: str | None = None
    seconds: float | None = None

    @property
    def usable(self) -> bool:
        """Whether a case may be routed with this graph without a human first looking."""
        return self.status is ExtractionStatus.OK

    @property
    def search_text(self) -> str:
        """Everything about this diagram that a text query could match.

        The node text, the questions, the options and the conditions, so a case
        description can find the diagram and then the path inside it. Built from extracted
        content only - nothing here is generated prose.
        """
        parts: list[str] = [self.visible_text]
        for node in self.nodes:
            parts.extend(filter(None, (node.text, node.condition, node.action, " ".join(node.options))))
        for edge in self.edges:
            if edge.label:
                parts.append(edge.label)
        return " ".join(part for part in parts if part)

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "document_id": self.document_id,
            "corpus_id": self.corpus_id,
            "status": self.status.value,
            "confidence": round(self.confidence, 3),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "paths": [path.to_dict() for path in self.paths],
            "visible_text": self.visible_text,
            "start_nodes": list(self.start_nodes),
            "end_nodes": list(self.end_nodes),
            "warnings": list(self.warnings),
            "review_reasons": list(self.review_reasons),
            "model": self.model,
            "seconds": round(self.seconds, 2) if self.seconds is not None else None,
        }


EXTRACT_PROMPT = """You are reading one decision flowchart from a customer-service or
product manual. Extract its structure exactly as drawn.

Absolute rules:
- Extract ONLY what is visibly drawn. Do not add a node, an arrow or a condition that is
  not in the picture, and do not complete a diagram you think is missing something.
- If you cannot tell which way an arrow points, set "direction_clear" to false for that
  edge. Do not guess a direction.
- If a box's text is unreadable or its purpose is unclear, mark that node
  "ambiguous": true and say why in its "note".
- Copy text verbatim. Do not paraphrase, translate or tidy it.
- If this figure is not actually a branching decision flowchart - if it is a table, a
  parts diagram, or a linear list of steps - return {"not_a_workflow": true, "reason": "..."}.

Reply with JSON only and nothing else:
{
  "confidence": 0.0 to 1.0,
  "visible_text": "all text you can read in the figure",
  "nodes": [
    {"node_id": "n1", "kind": "start|question|action|condition|end|unknown",
     "text": "the box's words verbatim", "options": ["Yes", "No"],
     "condition": null, "action": null, "ambiguous": false, "note": ""}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "label": "Yes", "direction_clear": true, "note": ""}
  ]
}

Use the node_id values you invented in "nodes" for every edge endpoint. Every edge must
reference nodes you listed."""


class WorkflowExtractor:
    """Extracts a decision graph from one image, then tries hard to distrust it."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:4b",
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def extract(self, path: Path, *, image_id: str, document_id: str, corpus_id: str) -> WorkflowGraph:
        """Read the diagram at *path*. Never raises; a failure is a status."""
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": EXTRACT_PROMPT,
                        "images": [base64.b64encode(path.read_bytes()).decode("ascii")],
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0.0, "num_ctx": 16384, "num_predict": 2000},
                    },
                )
                response.raise_for_status()
                raw = str((response.json() or {}).get("response") or "")
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return self._failed(image_id, document_id, corpus_id, f"extraction call failed: {exc}", started)

        seconds = time.perf_counter() - started
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return self._failed(image_id, document_id, corpus_id, "model did not return JSON", started)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return self._failed(image_id, document_id, corpus_id, f"unparseable JSON: {exc}", started)
        if not isinstance(parsed, dict):
            return self._failed(image_id, document_id, corpus_id, "JSON was not an object", started)

        if parsed.get("not_a_workflow"):
            return WorkflowGraph(
                image_id=image_id,
                document_id=document_id,
                corpus_id=corpus_id,
                status=ExtractionStatus.NEEDS_REVIEW,
                confidence=0.0,
                visible_text=str(parsed.get("visible_text") or "")[:4000],
                review_reasons=(f"the model says this is not a branching workflow: {parsed.get('reason')}",),
                model=self._model,
                seconds=round(seconds, 2),
            )

        return build_graph(
            parsed,
            image_id=image_id,
            document_id=document_id,
            corpus_id=corpus_id,
            model=self._model,
            seconds=seconds,
        )

    def _failed(self, image_id: str, document_id: str, corpus_id: str, reason: str, started: float) -> WorkflowGraph:
        return WorkflowGraph(
            image_id=image_id,
            document_id=document_id,
            corpus_id=corpus_id,
            status=ExtractionStatus.FAILED,
            confidence=0.0,
            review_reasons=(reason,),
            model=self._model,
            seconds=round(time.perf_counter() - started, 2),
        )


def build_graph(
    parsed: dict[str, object],
    *,
    image_id: str,
    document_id: str,
    corpus_id: str,
    model: str | None = None,
    seconds: float | None = None,
) -> WorkflowGraph:
    """Assemble and validate a graph from a model answer.

    Separate from the HTTP call so every validation rule is testable against a handwritten
    answer, with no model and no network. That matters more here than anywhere else in the
    corpus pipeline: these rules are the only thing standing between a misread arrow and an
    agent acting on it.
    """
    warnings: list[str] = []
    review: list[str] = []

    nodes = _nodes_from(parsed, warnings)
    known = {node.node_id for node in nodes}
    edges = _edges_from(parsed, known, warnings, review)

    confidence = parsed.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, int | float) else 0.0

    # -- structural checks, cheapest and most decisive first ---------------- #

    if len(nodes) < MIN_NODES:
        review.append(f"only {len(nodes)} node(s) extracted; a decision diagram needs at least {MIN_NODES}")
    if len(edges) < MIN_EDGES:
        review.append(f"only {len(edges)} edge(s) extracted; without arrows there is no routing")

    if any(not edge.direction_clear for edge in edges):
        unclear = sum(1 for edge in edges if not edge.direction_clear)
        review.append(f"{unclear} arrow direction(s) unclear; a reversed arrow sends the agent down the wrong branch")

    if any(node.ambiguous for node in nodes):
        names = [node.node_id for node in nodes if node.ambiguous][:5]
        review.append(f"ambiguous node(s): {', '.join(names)}")

    if any(node.kind is NodeKind.UNKNOWN for node in nodes):
        review.append("at least one node's purpose could not be determined")

    # A question with no labelled way out cannot route anything.
    for node in nodes:
        if node.kind is NodeKind.QUESTION:
            outgoing = [edge for edge in edges if edge.source == node.node_id]
            if len(outgoing) < 2:
                review.append(f"question node {node.node_id!r} has {len(outgoing)} outgoing edge(s); a branch needs 2")
            elif not all(edge.label for edge in outgoing):
                review.append(
                    f"question node {node.node_id!r} has an unlabelled branch, so the answers cannot be told apart"
                )

    if confidence_value < MIN_TRUSTED_CONFIDENCE:
        review.append(f"model confidence {confidence_value:.2f} is below {MIN_TRUSTED_CONFIDENCE}")

    # No branch at all means this is a linear sequence, which is the single most common
    # thing to be mistaken for a workflow. Measured on MRAMG-Manual: all three images the
    # classifier called workflow diagrams were linear or tabular.
    branching = [node for node in nodes if sum(1 for edge in edges if edge.source == node.node_id) >= 2]
    if not branching:
        review.append("no node has more than one outgoing edge, so nothing branches; this is a linear sequence")

    starts, ends = _endpoints(nodes, edges, review)
    paths = _paths(nodes, edges, starts, ends) if not review else ()
    if not review and not paths:
        review.append("no start-to-end path could be traced through the extracted graph")

    status = ExtractionStatus.NEEDS_REVIEW if review else ExtractionStatus.OK
    return WorkflowGraph(
        image_id=image_id,
        document_id=document_id,
        corpus_id=corpus_id,
        status=status,
        confidence=confidence_value,
        nodes=nodes,
        edges=edges,
        paths=paths,
        visible_text=str(parsed.get("visible_text") or "")[:4000],
        start_nodes=starts,
        end_nodes=ends,
        warnings=tuple(warnings),
        review_reasons=tuple(review),
        model=model,
        seconds=seconds,
    )


def _nodes_from(parsed: dict[str, object], warnings: list[str]) -> tuple[WorkflowNode, ...]:
    raw_nodes = parsed.get("nodes")
    if not isinstance(raw_nodes, list):
        warnings.append("no node list in the answer")
        return ()
    nodes: list[WorkflowNode] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            warnings.append(f"node {index} is not an object")
            continue
        node_id = str(item.get("node_id") or f"n{index}")
        if node_id in seen:
            warnings.append(f"duplicate node id {node_id!r}; the later one was dropped")
            continue
        seen.add(node_id)
        raw_kind = str(item.get("kind") or "").strip().lower()
        try:
            kind = NodeKind(raw_kind)
        except ValueError:
            warnings.append(f"node {node_id!r} has unknown kind {raw_kind!r}")
            kind = NodeKind.UNKNOWN
        options = item.get("options")
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                kind=kind,
                text=str(item.get("text") or "").strip(),
                options=tuple(str(option) for option in options) if isinstance(options, list) else (),
                condition=str(item["condition"]) if item.get("condition") else None,
                action=str(item["action"]) if item.get("action") else None,
                ambiguous=bool(item.get("ambiguous")),
                note=str(item.get("note") or ""),
            )
        )
    return tuple(nodes)


def _edges_from(
    parsed: dict[str, object], known: set[str], warnings: list[str], review: list[str]
) -> tuple[WorkflowEdge, ...]:
    raw_edges = parsed.get("edges")
    if not isinstance(raw_edges, list):
        warnings.append("no edge list in the answer")
        return ()
    edges: list[WorkflowEdge] = []
    for index, item in enumerate(raw_edges):
        if not isinstance(item, dict):
            warnings.append(f"edge {index} is not an object")
            continue
        source = str(item.get("source") or "")
        target = str(item.get("target") or "")
        # An edge naming a node that was not extracted is the clearest possible sign the
        # answer is partly invented, so it is dropped and it forces review.
        if source not in known or target not in known:
            review.append(f"edge {source!r} -> {target!r} references a node that was not extracted")
            continue
        if source == target:
            review.append(f"edge from {source!r} to itself")
            continue
        edges.append(
            WorkflowEdge(
                source=source,
                target=target,
                label=str(item["label"]) if item.get("label") else None,
                direction_clear=bool(item.get("direction_clear", True)),
                note=str(item.get("note") or ""),
            )
        )
    return tuple(edges)


def _endpoints(
    nodes: tuple[WorkflowNode, ...], edges: tuple[WorkflowEdge, ...], review: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Start and end nodes, preferring the declared kinds and falling back to degree."""
    declared_starts = tuple(node.node_id for node in nodes if node.kind is NodeKind.START)
    declared_ends = tuple(node.node_id for node in nodes if node.kind is NodeKind.END)

    incoming = {node.node_id: 0 for node in nodes}
    outgoing = {node.node_id: 0 for node in nodes}
    for edge in edges:
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
        incoming[edge.target] = incoming.get(edge.target, 0) + 1

    starts = declared_starts or tuple(node_id for node_id, count in incoming.items() if count == 0)
    ends = declared_ends or tuple(node_id for node_id, count in outgoing.items() if count == 0)

    if not starts:
        review.append("no start node: every node has an incoming arrow, so the diagram has no entry point")
    if not ends:
        review.append("no end node: every node has an outgoing arrow, so no path terminates")
    return starts, ends


#: Guard against a cyclic graph turning path enumeration into a hang, and against a
#: diagram so large that listing its paths is useless to a reader anyway.
MAX_PATHS = 64
MAX_PATH_LENGTH = 32


def _paths(
    nodes: tuple[WorkflowNode, ...],
    edges: tuple[WorkflowEdge, ...],
    starts: tuple[str, ...],
    ends: tuple[str, ...],
) -> tuple[WorkflowPath, ...]:
    """Every start-to-end route, with the conditions that select it.

    Depth-first with the visited set carried down the branch, so a cycle cannot loop
    forever. A cycle is legal in a real troubleshooting chart ("no? go back and retry"),
    so it is traversed once rather than treated as an error.
    """
    by_id = {node.node_id: node for node in nodes}
    outgoing: dict[str, list[WorkflowEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge.source, []).append(edge)

    found: list[WorkflowPath] = []
    end_set = set(ends)

    def walk(node_id: str, trail: list[str], labels: list[str], visited: frozenset[str]) -> None:
        if len(found) >= MAX_PATHS or len(trail) > MAX_PATH_LENGTH:
            return
        if node_id in end_set or node_id not in outgoing:
            actions = tuple(
                text
                for step in trail
                if (node := by_id.get(step)) is not None
                and (text := node.action or (node.text if node.kind is NodeKind.ACTION else None))
            )
            found.append(
                WorkflowPath(
                    path_id=f"p{len(found) + 1:02d}",
                    node_ids=tuple(trail),
                    conditions=tuple(labels),
                    actions=actions,
                )
            )
            return
        for edge in outgoing[node_id]:
            if edge.target in visited:
                continue
            walk(
                edge.target,
                [*trail, edge.target],
                [*labels, edge.label] if edge.label else list(labels),
                visited | {edge.target},
            )

    for start in starts:
        walk(start, [start], [], frozenset({start}))
    return tuple(found)


def write_workflows(graphs: list[WorkflowGraph], target: Path) -> Path:
    from agentpath_ingestion.corpora.base import write_jsonl

    path = target / WORKFLOWS_NAME
    write_jsonl(path, [graph.to_dict() for graph in graphs])
    return path


__all__ = [
    "EXTRACT_PROMPT",
    "MAX_PATHS",
    "MAX_PATH_LENGTH",
    "MIN_EDGES",
    "MIN_NODES",
    "MIN_TRUSTED_CONFIDENCE",
    "WORKFLOWS_NAME",
    "ExtractionStatus",
    "NodeKind",
    "WorkflowEdge",
    "WorkflowExtractor",
    "WorkflowGraph",
    "WorkflowNode",
    "WorkflowPath",
    "build_graph",
    "write_workflows",
]

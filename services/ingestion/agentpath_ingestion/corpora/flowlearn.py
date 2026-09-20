"""FlowLearn: a corpus of flowcharts that comes with the structure written down.

Why this corpus exists in this project
--------------------------------------
MRAMG-Manual contains no decision workflows - 138 of its 2,599 images were examined and
the three the classifier flagged turned out to be a lookup matrix and two linear assembly
sequences (``docs/IMAGE_ROLES.md``). So the workflow extractor built in Task 9B had no
corpus to be measured against, and an unmeasured extractor is an unsupported claim.

FlowLearn's simulated subset supplies exactly the ground truth the five structural metrics
need, and supplies it *as data* rather than as something we would have had to annotate
ourselves:

``Flowchart-to-Mermaid``      the whole graph: nodes, edges and their direction
``Num_Nodes`` / ``Num_Arrows``  counts, for detection accuracy without alignment
``Arrow_AtoB``                a true and a false ordered pair - arrow *direction*
``Arrow_betweenAB``           a true and a false unordered pair - edge *existence*
``ocr``                       the text visible in the figure
``Flowchart-to-Description``  the graph in prose, for path reasoning

Licence, and the limits it puts on this repository
--------------------------------------------------
**CC BY-NC-SA 4.0.** Read before changing anything here.

* Used for **non-commercial evaluation and research only**. If AgentPath ever becomes a
  commercial project this corpus must be removed or separately licensed from the authors.
* Nothing from it is committed: images, annotations and any derived extraction stay under
  ``data/external/``, which is gitignored. Only aggregate metrics reach the repository,
  and a metric is a fact about our system's performance rather than an adaptation of the
  dataset.
* **Not used for training or fine-tuning**, and not shipped in the product or any install
  package. This module reads it, scores against it, and nothing else.
* Attribution travels with the corpus record and appears in the evaluation report.

Pan, Zhang, Caragea, Dragut and Latecki, *FlowLearn: Evaluating Large Vision-Language
Models on Flowchart Understanding*, ECAI 2024. arXiv:2407.05183.
https://huggingface.co/datasets/jopan/FlowLearn

Why only a slice, and why the simulated half
--------------------------------------------
The full dataset is 3.39 GB. The scientific half was inspected first and rejected for this
purpose: its annotations are OCR boxes, true/false statements and captions - no node, edge
or direction ground truth at all, so it can support one of the six ordered metrics and not
the other five. The simulated half carries the Mermaid source, which is what makes the
structural metrics computable.

A deterministic sample of the simulated test split is taken rather than all 2,000 images,
because 600 is already far more than the metrics need and a smaller honest sample is worth
more than a larger one nobody re-runs. The selection is seeded and recorded, so "which 600"
is reproducible.

This corpus is deliberately **not** indexed into RAGFlow. There is no retrieval task here -
the question is whether a diagram can be read, not whether it can be found - so it gets its
own corpus id and its own directory and never touches the procedures or MRAMG datasets.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)

CORPUS_ID = "flowlearn-sim"
CORPUS_TITLE = "FlowLearn - SimFlowchart (word), test split"
CORPUS_LICENCE = "CC-BY-NC-SA-4.0"
CORPUS_ATTRIBUTION = (
    "FlowLearn (jopan/FlowLearn on Hugging Face), SimFlowchart word test split. "
    "Pan, Zhang, Caragea, Dragut and Latecki, 'FlowLearn: Evaluating Large Vision-Language "
    "Models on Flowchart Understanding', ECAI 2024, arXiv:2407.05183. "
    "Licensed CC BY-NC-SA 4.0 and used here for non-commercial evaluation only: not committed, "
    "not used for training or fine-tuning, and not shipped in any product."
)
SOURCE_URL = "https://huggingface.co/datasets/jopan/FlowLearn"

_BASE = "https://huggingface.co/datasets/jopan/FlowLearn/resolve/main/SimFlowchart"
VQA_TEST_URL = f"{_BASE}/word/VQA/test.json"
IMAGE_URL = f"{_BASE}/images/mermaid_word/jpeg"

#: How many of the 2,000 test flowcharts to take. 600 splits into 300 dev and 300 held-out,
#: which puts the standard error on a proportion at about 2.9 points - tight enough to tell
#: 60% from 75% without downloading data nobody will look at. Recorded because changing it
#: changes every number.
SAMPLE_SIZE = 600

#: Seeds the sample. Separate from the dev/held-out seed on purpose: *which* flowcharts were
#: downloaded and *how they were split* are two decisions, and conflating them would mean a
#: larger download silently reshuffled the split.
SAMPLE_SEED = 20260920

#: Refuse a download that is wildly larger than expected rather than filling the disk. The
#: measured average is about 22 KB per flowchart.
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


class FlowLearnError(RuntimeError):
    """Something about the corpus on disk is wrong in a way that invalidates metrics."""


@dataclass(frozen=True, slots=True)
class MermaidGraph:
    """The ground-truth graph, parsed from the Mermaid source the corpus ships.

    Parsed rather than trusted as a string, so node and edge accuracy are computed against
    a structure instead of against text similarity.
    """

    #: ``entity0`` -> its label, in declaration order.
    nodes: dict[str, str] = field(default_factory=dict)
    #: ``(source, target)`` pairs, direction as drawn.
    edges: tuple[tuple[str, str], ...] = ()
    #: Direction keyword, e.g. ``TB``.
    orientation: str = ""

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.nodes.values())

    @property
    def label_edges(self) -> tuple[tuple[str, str], ...]:
        """Edges as label pairs, which is the only form a model's answer can be scored in.

        A model has no way to know the corpus called a box ``entity3``; it can only report
        the words in it. So alignment happens on labels.
        """
        return tuple(
            (self.nodes[source], self.nodes[target])
            for source, target in self.edges
            if source in self.nodes and target in self.nodes
        )

    @property
    def branch_nodes(self) -> tuple[str, ...]:
        """Labels of nodes with more than one outgoing edge - the decision points."""
        outgoing: dict[str, int] = {}
        for source, _target in self.edges:
            outgoing[source] = outgoing.get(source, 0) + 1
        return tuple(self.nodes[node] for node, count in outgoing.items() if count > 1 and node in self.nodes)


#: ``entity0(some label)`` or ``entity0{some label}`` or ``entity0[some label]``.
_NODE = re.compile(r"^\s*(entity\d+)\s*[([{]\s*(.*?)\s*[)\]}]\s*$")
#: ``entity0 --> entity1`` with any of Mermaid's arrow bodies, optionally labelled.
_EDGE = re.compile(r"^\s*(entity\d+)\s*[-=.]+[->]*\s*(?:\|(.*?)\|)?\s*(entity\d+)\s*$")
_ORIENT = re.compile(r"^\s*flowchart\s+(\w+)\s*$", re.IGNORECASE)


def parse_mermaid(source: str) -> MermaidGraph:
    """Parse the corpus's Mermaid into nodes and directed edges.

    Deliberately strict: a line that is neither a recognised node nor a recognised edge is
    ignored rather than guessed at. If this parser ever silently dropped edges the structural
    metrics would flatter the model, so :func:`verify_parse` checks the parse against the
    corpus's own ``Num_Nodes`` and ``Num_Arrows`` counts and the evaluation refuses any
    flowchart where they disagree.
    """
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    orientation = ""

    cleaned = source.replace("```mermaid", "").replace("```", "")
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        orient = _ORIENT.match(stripped)
        if orient:
            orientation = orient.group(1).upper()
            continue
        node = _NODE.match(stripped)
        if node:
            nodes[node.group(1)] = node.group(2).strip()
            continue
        edge = _EDGE.match(stripped)
        if edge:
            edges.append((edge.group(1), edge.group(3)))
    return MermaidGraph(nodes=nodes, edges=tuple(edges), orientation=orientation)


def verify_parse(graph: MermaidGraph, *, expected_nodes: int, expected_arrows: int) -> list[str]:
    """Reasons this flowchart's ground truth cannot be trusted, if any.

    Run over every flowchart before any model sees it. A disagreement here is a bug in our
    parser, not in the model, and scoring against a mis-parsed graph would produce numbers
    that say nothing.
    """
    problems: list[str] = []
    if len(graph.nodes) != expected_nodes:
        problems.append(f"parsed {len(graph.nodes)} nodes but the corpus says {expected_nodes}")
    if len(graph.edges) != expected_arrows:
        problems.append(f"parsed {len(graph.edges)} edges but the corpus says {expected_arrows}")
    if len(set(graph.labels)) != len(graph.labels):
        problems.append("two nodes share a label, so a model's answer cannot be aligned unambiguously")
    for source, target in graph.edges:
        if source not in graph.nodes or target not in graph.nodes:
            problems.append(f"edge {source}->{target} names a node that was not declared")
            break
    return problems


@dataclass(frozen=True, slots=True)
class FlowchartRecord:
    """One flowchart: the image, and everything the corpus says about it."""

    flowchart_id: str
    image_path: Path
    #: The ground-truth graph.
    graph: MermaidGraph
    mermaid_source: str
    expected_nodes: int
    expected_arrows: int
    #: Text visible in the figure, per the corpus.
    ocr_text: str
    #: The graph in prose, for path reasoning.
    description: str
    #: ``{"true": {"a": ..., "b": ...}, "false": {...}}`` - an ordered pair that is drawn and
    #: one that is not. This is how arrow direction is scored.
    arrow_a_to_b: dict[str, dict[str, str]] = field(default_factory=dict)
    #: The same shape for undirected adjacency.
    arrow_between: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Filled by :func:`load_corpus` when the ground truth failed verification.
    ground_truth_problems: tuple[str, ...] = ()

    @property
    def trustworthy(self) -> bool:
        return not self.ground_truth_problems

    @property
    def byte_size(self) -> int:
        try:
            return self.image_path.stat().st_size
        except OSError:
            return 0


@dataclass
class FetchReport:
    corpus_id: str = CORPUS_ID
    images: int = 0
    downloaded_bytes: int = 0
    reused_bytes: int = 0
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return self.downloaded_bytes + self.reused_bytes


def default_root(repo_root: Path) -> Path:
    return repo_root / "data" / "external" / "flowlearn"


def selected_ids(annotations: dict[str, dict], *, sample: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    """Which flowcharts this project uses, reproducibly.

    Sorted before shuffling so the choice does not depend on the order the JSON happened to
    be written in, exactly as the dev/held-out split does.
    """
    keys = sorted(annotations)
    random.Random(f"{seed}:{CORPUS_ID}").shuffle(keys)
    return keys[:sample]


def fetch(root: Path, *, sample: int = SAMPLE_SIZE, force: bool = False) -> FetchReport:
    """Download the annotations and the sampled images. Idempotent and resumable.

    A second run downloads nothing and says so. An interrupted run resumes, because each
    image is a separate small file and presence on disk is the resume marker.
    """
    started = time.perf_counter()
    report = FetchReport()
    raw = root / "raw"
    images_dir = raw / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = raw / "vqa_test.json"
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        if force or not annotations_path.is_file():
            response = client.get(VQA_TEST_URL)
            response.raise_for_status()
            annotations_path.write_bytes(response.content)
            report.downloaded_bytes += len(response.content)
            logger.info("flowlearn.fetched_annotations", bytes=len(response.content))
        else:
            report.reused_bytes += annotations_path.stat().st_size

        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        wanted = selected_ids(annotations, sample=sample)

        for flowchart_id in wanted:
            target = images_dir / flowchart_id
            if target.is_file() and not force:
                report.reused_bytes += target.stat().st_size
                report.images += 1
                continue
            if report.downloaded_bytes > MAX_TOTAL_BYTES:
                report.problems.append(f"stopped: download exceeded {MAX_TOTAL_BYTES} bytes")
                break
            try:
                response = client.get(f"{IMAGE_URL}/{flowchart_id}")
                response.raise_for_status()
            except httpx.HTTPError as exc:
                report.problems.append(f"{flowchart_id}: {exc}")
                continue
            if len(response.content) > MAX_IMAGE_BYTES:
                report.problems.append(f"{flowchart_id}: {len(response.content)} bytes is implausibly large; skipped")
                continue
            target.write_bytes(response.content)
            report.downloaded_bytes += len(response.content)
            report.images += 1

    report.seconds = round(time.perf_counter() - started, 1)
    logger.info(
        "flowlearn.fetched",
        images=report.images,
        downloaded=report.downloaded_bytes,
        reused=report.reused_bytes,
        seconds=report.seconds,
        problems=len(report.problems),
    )
    return report


def load_corpus(root: Path, *, sample: int = SAMPLE_SIZE) -> tuple[FlowchartRecord, ...]:
    """Read what is on disk, verifying every flowchart's ground truth before use.

    A flowchart whose Mermaid does not parse to the counts the corpus states is kept but
    flagged, and the evaluation excludes it with the reason reported. Silently scoring
    against ground truth we mis-parsed would be the worst available outcome: the numbers
    would look like model performance.
    """
    raw = root / "raw"
    annotations_path = raw / "vqa_test.json"
    if not annotations_path.is_file():
        raise FlowLearnError(f"{annotations_path} is missing. Run: make flowlearn-fetch")

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    wanted = selected_ids(annotations, sample=sample)

    records: list[FlowchartRecord] = []
    for flowchart_id in wanted:
        image_path = raw / "images" / flowchart_id
        if not image_path.is_file():
            continue
        entry = annotations[flowchart_id]
        mermaid = str(entry.get("Flowchart-to-Mermaid") or "")
        graph = parse_mermaid(mermaid)
        expected_nodes = int(entry.get("Num_Nodes") or 0)
        expected_arrows = int(entry.get("Num_Arrows") or 0)
        problems = verify_parse(graph, expected_nodes=expected_nodes, expected_arrows=expected_arrows)
        records.append(
            FlowchartRecord(
                flowchart_id=flowchart_id,
                image_path=image_path,
                graph=graph,
                mermaid_source=mermaid,
                expected_nodes=expected_nodes,
                expected_arrows=expected_arrows,
                ocr_text=str(entry.get("ocr") or ""),
                description=str(entry.get("Flowchart-to-Description") or ""),
                arrow_a_to_b=entry.get("Arrow_AtoB") or {},
                arrow_between=entry.get("Arrow_betweenAB") or {},
                ground_truth_problems=tuple(problems),
            )
        )

    logger.info(
        "flowlearn.loaded",
        flowcharts=len(records),
        trustworthy=sum(1 for record in records if record.trustworthy),
    )
    return tuple(records)


def corpus_fingerprint(records: tuple[FlowchartRecord, ...]) -> str:
    """A hash of which flowcharts are in play, so a report names its own inputs."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.flowchart_id):
        digest.update(record.flowchart_id.encode("utf-8"))
    return digest.hexdigest()[:16]


__all__ = [
    "CORPUS_ATTRIBUTION",
    "CORPUS_ID",
    "CORPUS_LICENCE",
    "CORPUS_TITLE",
    "SAMPLE_SEED",
    "SAMPLE_SIZE",
    "SOURCE_URL",
    "FetchReport",
    "FlowLearnError",
    "FlowchartRecord",
    "MermaidGraph",
    "corpus_fingerprint",
    "default_root",
    "fetch",
    "load_corpus",
    "parse_mermaid",
    "selected_ids",
    "verify_parse",
]

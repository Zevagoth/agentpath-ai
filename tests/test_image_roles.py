"""Image roles, workflow extraction, and the gallery that hides nothing.

Three things are under test and they fail in different ways, so they are tested apart.

**Classification** must never overstate what it knows. An image no model examined is
recorded as unexamined, not as "checked and not a diagram", and a model that answers
outside the schema is corrected with the deviation kept.

**Workflow extraction** must refuse more readily than it accepts. Most of these tests feed
it a broken or invented graph and assert that it declines rather than producing something
plausible. That ratio is deliberate: a misread arrow acted on by an agent is worse than no
extraction at all.

**The gallery** must be complete. Its whole purpose is that no correct figure is hidden by
a ranking, so the tests assert completeness and order rather than relevance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentpath.corpus import KNOWN_IMAGE_ROLES, Corpus, CorpusCatalogue
from agentpath.corpus_retrieval.gallery import (
    MAX_GALLERY_IMAGES,
    build_gallery,
    gallery_diagnostics,
)
from agentpath.corpus_retrieval.ports import CorpusPassage, DocumentHit, OffsetSource
from agentpath_ingestion.corpora.base import CorpusBundle, CorpusDocument, CorpusImage, slice_chunks
from agentpath_ingestion.corpora.export import export_bundle
from agentpath_ingestion.corpora.roles import (
    MIN_DIAGRAM_AREA,
    ROLES_NAME,
    ImageRole,
    ImageRoleRecord,
    RoleMethod,
    RoleReport,
    VisionClassifier,
    _role_from_answer,
    classify_bundle,
    is_diagram_candidate,
    jpeg_dimensions,
    write_roles,
)
from agentpath_ingestion.corpora.workflow import (
    MIN_TRUSTED_CONFIDENCE,
    ExtractionStatus,
    NodeKind,
    build_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# A real 1x1 PNG and a real 8x8 JPEG, so parsers meet real bytes.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d4944415478da63f8ffff3f0005fe02fea735d1c2"
    "0000000049454e44ae426082"
)


def _jpeg(width: int, height: int) -> bytes:
    """A minimal JPEG carrying real SOF dimensions. Not decodable, but parseable."""
    import struct

    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + sof + b"\xff\xd9"


# --------------------------------------------------------------------------- #
# Geometry without an image library
# --------------------------------------------------------------------------- #


def test_jpeg_dimensions_are_read_from_the_header(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(_jpeg(1494, 1953))

    assert jpeg_dimensions(path) == (1494, 1953)


def test_a_non_jpeg_reports_unknown_rather_than_raising(tmp_path: Path) -> None:
    """86 of the corpus's 2,599 images are not parseable here. They are simply not
    prefiltered on size, which is a lost signal and not an error."""
    path = tmp_path / "a.png"
    path.write_bytes(PNG)

    assert jpeg_dimensions(path) is None


def test_a_missing_file_reports_unknown(tmp_path: Path) -> None:
    assert jpeg_dimensions(tmp_path / "gone.jpg") is None


def test_a_truncated_jpeg_does_not_hang_or_raise(tmp_path: Path) -> None:
    path = tmp_path / "cut.jpg"
    path.write_bytes(b"\xff\xd8\xff\xc0\x00")

    assert jpeg_dimensions(path) is None


# --------------------------------------------------------------------------- #
# The prefilter: who gets looked at
# --------------------------------------------------------------------------- #


def _image(image_id: str = "i1", offset: int = 0) -> CorpusImage:
    return CorpusImage(image_id=image_id, document_id="d1", position_index=0, char_offset=offset)


def test_a_question_with_both_branches_is_a_candidate_whatever_its_size() -> None:
    """The strongest available text signal, and the only one strong enough alone."""
    candidate, why = is_diagram_candidate(_image(), "Did the customer make contact? Yes / No", None)

    assert candidate
    assert "question" in why


def test_flow_vocabulary_alone_is_not_enough() -> None:
    """ "Troubleshoot" sits beside tables constantly. On the real corpus 155 windows
    mention flow vocabulary and only one image turned out to be a diagram."""
    candidate, _ = is_diagram_candidate(_image(), "See the troubleshooting chart below.", (400, 300))

    assert not candidate


def test_flow_vocabulary_with_a_page_sized_figure_is_a_candidate() -> None:
    large = (1500, 1900)
    candidate, why = is_diagram_candidate(_image(), "See the troubleshooting chart below.", large)

    assert candidate
    assert "MP" in why


def test_a_large_figure_with_no_vocabulary_is_not_a_candidate() -> None:
    """Otherwise every product photo in the corpus would be sent to the model."""
    candidate, why = is_diagram_candidate(_image(), "The appliance is shown below.", (2000, 2000))

    assert not candidate
    assert "MP" in why, "the size is still reported, it is just not sufficient"


def test_the_size_floor_is_where_it_says_it_is() -> None:
    text = "Refer to the diagram."
    just_under = is_diagram_candidate(_image(), text, (500, 499))[0]
    just_over = is_diagram_candidate(_image(), text, (500, 500))[0]

    assert MIN_DIAGRAM_AREA == 250_000
    assert not just_under
    assert just_over


# --------------------------------------------------------------------------- #
# Reading the model's answer
# --------------------------------------------------------------------------- #


def test_a_role_outside_the_schema_becomes_other_and_is_recorded() -> None:
    """Measured: over 78 images this model answered "table" nine times and
    "product_photo" once, despite the prompt offering three values."""
    role, warnings = _role_from_answer({"role": "table"})

    assert role is ImageRole.OTHER
    assert warnings
    assert "table" in warnings[0]


def test_a_known_role_passes_through_without_a_warning() -> None:
    role, warnings = _role_from_answer({"role": "ui_screenshot"})

    assert role is ImageRole.UI_SCREENSHOT
    assert warnings == []


def test_a_missing_role_becomes_other() -> None:
    role, warnings = _role_from_answer({})

    assert role is ImageRole.OTHER
    assert warnings


def test_the_reader_and_the_writer_agree_on_the_role_vocabulary() -> None:
    """A role the writer can emit but the reader does not know would reach the UI as an
    unrecognised string."""
    assert {role.value for role in ImageRole} == set(KNOWN_IMAGE_ROLES)


# --------------------------------------------------------------------------- #
# Classifying a bundle
# --------------------------------------------------------------------------- #


class FakeClassifier:
    """A stand-in for the vision model, answering from a script."""

    model = "fake-vision"

    def __init__(self, answers: dict[str, dict] | None = None, *, available: bool = True) -> None:
        self._answers = answers or {}
        self._available = available
        self.asked: list[str] = []

    def available(self) -> tuple[bool, str]:
        return self._available, "fake" if self._available else "no vision capability"

    def classify(self, path: Path) -> tuple[dict, float, list[str]]:
        self.asked.append(path.stem)
        return self._answers.get(path.stem, {}), 0.1, []


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[CorpusBundle, Path]:
    """A document with one page-sized figure beside flow vocabulary, and one small one."""
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "big.jpg").write_bytes(_jpeg(1500, 1900))
    (images_root / "small.jpg").write_bytes(_jpeg(300, 200))

    text = (
        "# Troubleshooting\n\n"
        "Refer to the troubleshooting chart below for the procedure.\n\n"
        "<PIC>\n\n"
        "The control panel is shown here for reference only.\n\n"
        "<PIC>\n\n"
        "End of section.\n"
    )
    offsets = [index for index in range(len(text)) if text.startswith("<PIC>", index)]
    document = CorpusDocument(
        document_id="d1",
        title="Troubleshooting",
        text=text,
        source_id="demo-d1",
        images=(
            CorpusImage(image_id="big", document_id="d1", position_index=0, char_offset=offsets[0]),
            CorpusImage(image_id="small", document_id="d1", position_index=1, char_offset=offsets[1]),
        ),
    )
    return (
        CorpusBundle(
            corpus_id="demo",
            title="Demo",
            attribution="Demo",
            licence="CC0-1.0",
            documents=(document,),
            chunks=slice_chunks(document, target_chars=200, overlap_chars=20),
            manifest={"image_retrieval_mode": "context-window"},
        ),
        images_root,
    )


def test_only_prefilter_candidates_reach_the_model(bundle: tuple[CorpusBundle, Path]) -> None:
    """The cost argument, asserted: 2,599 images produced 67 candidates on the real
    corpus, because a UI screenshot and an "other" figure display identically."""
    corpus_bundle, images_root = bundle
    classifier = FakeClassifier({"big": {"role": "workflow_diagram", "confidence": 0.9, "has_branches": True}})

    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]

    assert classifier.asked == ["big"]
    assert report.candidates == 1
    assert report.examined == 1


def test_an_unexamined_image_says_so_rather_than_claiming_it_is_not_a_diagram(
    bundle: tuple[CorpusBundle, Path],
) -> None:
    """The distinction the UI must not lose."""
    corpus_bundle, images_root = bundle
    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=None)

    small = next(record for record in report.records if record.image_id == "small")
    assert small.role is ImageRole.OTHER
    assert small.method is RoleMethod.PREFILTER
    assert small.confidence == 0.0
    assert not small.examined
    assert any("unchecked" in warning for warning in small.warnings)


def test_a_workflow_label_without_branches_is_downgraded(bundle: tuple[CorpusBundle, Path]) -> None:
    """The guard that mattered most in practice.

    On the real corpus every image the model first called a workflow_diagram reported no
    branches, and all three turned out to be a lookup matrix or a linear assembly
    sequence. Believing the label over the model's own evidence would have fed three
    non-workflows into the extraction pipeline.
    """
    corpus_bundle, images_root = bundle
    classifier = FakeClassifier(
        {"big": {"role": "workflow_diagram", "confidence": 0.95, "has_arrows": True, "has_branches": False}}
    )

    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]

    big = next(record for record in report.records if record.image_id == "big")
    assert big.role is ImageRole.OTHER
    assert any("no branches" in warning for warning in big.warnings)
    assert report.workflows == []


def test_a_workflow_with_branches_survives(bundle: tuple[CorpusBundle, Path]) -> None:
    corpus_bundle, images_root = bundle
    classifier = FakeClassifier(
        {"big": {"role": "workflow_diagram", "confidence": 0.9, "has_arrows": True, "has_branches": True}}
    )

    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]

    assert [record.image_id for record in report.workflows] == ["big"]
    assert report.workflows[0].method is RoleMethod.VISION_MODEL


def test_an_unavailable_model_leaves_everything_unexamined_and_says_why(
    bundle: tuple[CorpusBundle, Path],
) -> None:
    """A text-only model answers this prompt fluently and entirely from imagination, so
    the capability is checked rather than assumed."""
    corpus_bundle, images_root = bundle
    classifier = FakeClassifier(available=False)

    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]

    assert classifier.asked == []
    assert report.problems
    assert "unavailable" in report.problems[0]
    assert all(not record.examined for record in report.records)


def test_a_model_that_gives_no_usable_answer_leaves_the_image_unexamined(
    bundle: tuple[CorpusBundle, Path],
) -> None:
    corpus_bundle, images_root = bundle
    classifier = FakeClassifier({"big": {}})

    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]

    big = next(record for record in report.records if record.image_id == "big")
    assert big.method is RoleMethod.PREFILTER
    assert "no usable answer" in big.reason


def test_the_role_file_round_trips_into_the_api_reader(tmp_path: Path, bundle: tuple[CorpusBundle, Path]) -> None:
    """The writer and the reader meet at a file, so the contract is worth asserting."""
    corpus_bundle, images_root = bundle
    build = tmp_path / "demo" / "build"
    export_bundle(corpus_bundle, build, images_root=images_root)

    classifier = FakeClassifier(
        {"big": {"role": "workflow_diagram", "confidence": 0.9, "has_arrows": True, "has_branches": True}}
    )
    report = classify_bundle(corpus_bundle, images_root=images_root, classifier=classifier)  # type: ignore[arg-type]
    write_roles(report, build)

    catalogue = CorpusCatalogue.load(tmp_path)
    corpus = catalogue.corpus("demo")
    assert corpus is not None
    assert catalogue.problems == ()
    big = corpus.image("big")
    assert big is not None
    assert big.role == "workflow_diagram"
    assert big.role_method == "vision_model"
    assert big.is_workflow_diagram
    assert big.role_examined
    small = corpus.image("small")
    assert small is not None
    assert small.role == "other"
    assert not small.role_examined


def test_a_role_file_naming_an_unknown_image_is_reported_not_invented(
    tmp_path: Path, bundle: tuple[CorpusBundle, Path]
) -> None:
    corpus_bundle, images_root = bundle
    build = tmp_path / "demo" / "build"
    export_bundle(corpus_bundle, build, images_root=images_root)
    report = RoleReport(
        corpus_id="demo",
        records=[
            ImageRoleRecord(
                image_id="ghost",
                document_id="d1",
                role=ImageRole.WORKFLOW_DIAGRAM,
                method=RoleMethod.VISION_MODEL,
                confidence=1.0,
                reason="stale",
            )
        ],
    )
    write_roles(report, build)

    catalogue = CorpusCatalogue.load(tmp_path)
    corpus = catalogue.corpus("demo")

    assert corpus is not None
    assert corpus.image("ghost") is None
    assert any("ghost" in problem for problem in catalogue.problems)


def test_an_unknown_role_in_the_file_falls_back_to_other(tmp_path: Path, bundle: tuple[CorpusBundle, Path]) -> None:
    corpus_bundle, images_root = bundle
    build = tmp_path / "demo" / "build"
    export_bundle(corpus_bundle, build, images_root=images_root)
    (build / ROLES_NAME).write_text(
        json.dumps({"image_id": "big", "document_id": "d1", "role": "invented", "method": "vision_model"}) + "\n",
        encoding="utf-8",
    )

    catalogue = CorpusCatalogue.load(tmp_path)
    corpus = catalogue.corpus("demo")

    assert corpus is not None
    big = corpus.image("big")
    assert big is not None and big.role == "other"
    assert any("invented" in problem for problem in catalogue.problems)


def test_a_bundle_with_no_role_file_behaves_exactly_as_before(
    tmp_path: Path, bundle: tuple[CorpusBundle, Path]
) -> None:
    corpus_bundle, images_root = bundle
    export_bundle(corpus_bundle, tmp_path / "demo" / "build", images_root=images_root)

    corpus = CorpusCatalogue.load(tmp_path).corpus("demo")

    assert corpus is not None
    assert all(record.role == "other" for record in corpus.images.values())
    assert all(not record.role_examined for record in corpus.images.values())


def test_the_real_vision_classifier_reports_an_unreachable_host_rather_than_raising() -> None:
    ok, detail = VisionClassifier(base_url="http://127.0.0.1:1", model="nope").available()

    assert ok is False
    assert "cannot reach" in detail


# --------------------------------------------------------------------------- #
# Workflow extraction: mostly about refusing
# --------------------------------------------------------------------------- #


def _answer(**overrides: object) -> dict:
    """A well-formed two-branch decision graph, for tests to break one field at a time."""
    base: dict = {
        "confidence": 0.9,
        "visible_text": "Did the customer make contact?",
        "nodes": [
            {"node_id": "n1", "kind": "start", "text": "Start"},
            {"node_id": "n2", "kind": "question", "text": "Did the customer make contact?", "options": ["Yes", "No"]},
            {"node_id": "n3", "kind": "action", "text": "Follow path A", "action": "Follow path A"},
            {"node_id": "n4", "kind": "action", "text": "Follow path B", "action": "Follow path B"},
            {"node_id": "n5", "kind": "end", "text": "End"},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "direction_clear": True},
            {"source": "n2", "target": "n3", "label": "Yes", "direction_clear": True},
            {"source": "n2", "target": "n4", "label": "No", "direction_clear": True},
            {"source": "n3", "target": "n5", "direction_clear": True},
            {"source": "n4", "target": "n5", "direction_clear": True},
        ],
    }
    base.update(overrides)
    return base


def _graph(**overrides: object):
    return build_graph(_answer(**overrides), image_id="i1", document_id="d1", corpus_id="demo")


def test_a_clean_branching_graph_is_usable() -> None:
    graph = _graph()

    assert graph.status is ExtractionStatus.OK
    assert graph.usable
    assert graph.review_reasons == ()
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 5


def test_every_extraction_carries_the_image_and_document_it_came_from() -> None:
    """The hard rule: a routing decision has to be traceable to a picture a human can open."""
    graph = _graph()

    assert graph.image_id == "i1"
    assert graph.document_id == "d1"
    assert graph.corpus_id == "demo"


def test_both_paths_are_traced_with_the_conditions_that_select_them() -> None:
    graph = _graph()

    assert len(graph.paths) == 2
    conditions = {path.conditions for path in graph.paths}
    assert ("Yes",) in conditions
    assert ("No",) in conditions
    actions = {path.actions for path in graph.paths}
    assert ("Follow path A",) in actions
    assert ("Follow path B",) in actions


def test_an_unclear_arrow_direction_forces_review() -> None:
    """A reversed arrow sends the agent down the opposite branch, so it is never guessed."""
    answer = _answer()
    answer["edges"][1]["direction_clear"] = False

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert not graph.usable
    assert any("direction" in reason for reason in graph.review_reasons)


def test_an_ambiguous_node_forces_review() -> None:
    answer = _answer()
    answer["nodes"][1]["ambiguous"] = True

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("ambiguous" in reason for reason in graph.review_reasons)


def test_an_edge_naming_a_node_that_was_not_extracted_is_dropped_and_forces_review() -> None:
    """The clearest sign an answer is partly invented."""
    answer = _answer()
    answer["edges"].append({"source": "n2", "target": "n99", "label": "Maybe"})

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert all(edge.target != "n99" for edge in graph.edges)
    assert any("n99" in reason for reason in graph.review_reasons)


def test_a_linear_sequence_is_refused_because_nothing_branches() -> None:
    """The single most common thing mistaken for a workflow.

    This is exactly what the model produced for the one MRAMG-Manual image it called a
    workflow diagram: it read all the text correctly and then chained eight symptoms into
    n1 -> n2 -> ... -> n8, when the real figure is an 8x9 matrix.
    """
    answer = _answer(
        nodes=[{"node_id": f"n{i}", "kind": "start", "text": f"Symptom {i}"} for i in range(1, 9)],
        edges=[{"source": f"n{i}", "target": f"n{i + 1}", "direction_clear": True} for i in range(1, 8)],
    )

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("branches" in reason for reason in graph.review_reasons)
    assert graph.paths == ()


def test_a_question_with_one_way_out_is_refused() -> None:
    answer = _answer()
    answer["edges"] = [edge for edge in answer["edges"] if not (edge["source"] == "n2" and edge["target"] == "n4")]

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("outgoing edge" in reason for reason in graph.review_reasons)


def test_a_question_whose_branches_are_unlabelled_is_refused() -> None:
    """Two ways out with no labels cannot tell the answers apart."""
    answer = _answer()
    for edge in answer["edges"]:
        edge.pop("label", None)

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("unlabelled" in reason for reason in graph.review_reasons)


def test_low_confidence_forces_review_however_clean_the_structure() -> None:
    graph = _graph(confidence=MIN_TRUSTED_CONFIDENCE - 0.01)

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("confidence" in reason for reason in graph.review_reasons)


def test_a_graph_with_too_few_nodes_is_refused() -> None:
    graph = _graph(
        nodes=[{"node_id": "n1", "kind": "start", "text": "Only"}],
        edges=[],
    )

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any("node" in reason for reason in graph.review_reasons)


def test_an_unknown_node_kind_forces_review_and_is_recorded() -> None:
    answer = _answer()
    answer["nodes"][2]["kind"] = "mystery"

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert any(node.kind is NodeKind.UNKNOWN for node in graph.nodes)
    assert any("mystery" in warning for warning in graph.warnings)


def test_a_self_loop_is_dropped_and_forces_review() -> None:
    answer = _answer()
    answer["edges"].append({"source": "n2", "target": "n2", "label": "Retry"})

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert all(edge.source != edge.target for edge in graph.edges)


def test_a_duplicate_node_id_keeps_the_first_and_warns() -> None:
    answer = _answer()
    answer["nodes"].append({"node_id": "n2", "kind": "action", "text": "Impostor"})

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert sum(1 for node in graph.nodes if node.node_id == "n2") == 1
    assert any("duplicate" in warning for warning in graph.warnings)


def test_a_cycle_is_traversed_once_rather_than_hanging() -> None:
    """A loop is legal in a real troubleshooting chart: "no? go back and retry"."""
    answer = _answer()
    answer["edges"].append({"source": "n4", "target": "n2", "label": "Retry", "direction_clear": True})

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.paths or graph.review_reasons


def test_node_text_is_kept_verbatim() -> None:
    """An agent compares this against the picture, so it is never tidied."""
    answer = _answer()
    answer["nodes"][1]["text"] = "  Is the booking   PRESENT?  "

    graph = build_graph(answer, image_id="i1", document_id="d1", corpus_id="demo")

    node = next(node for node in graph.nodes if node.node_id == "n2")
    assert node.text == "Is the booking   PRESENT?"


def test_the_search_text_carries_the_questions_options_and_conditions() -> None:
    """This is what lets a written case description find the diagram and then the path."""
    graph = _graph()
    text = graph.search_text

    assert "Did the customer make contact?" in text
    assert "Yes" in text and "No" in text
    assert "Follow path A" in text
    assert "Follow path B" in text


def test_a_model_saying_this_is_not_a_workflow_is_believed() -> None:
    graph = build_graph(
        {"not_a_workflow": True, "reason": "it is a table"},
        image_id="i1",
        document_id="d1",
        corpus_id="demo",
    )

    # build_graph does not special-case this; the extractor does. Asserted here so the
    # empty answer still comes back as review rather than as a usable empty graph.
    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert not graph.usable


def test_an_empty_answer_is_refused() -> None:
    graph = build_graph({}, image_id="i1", document_id="d1", corpus_id="demo")

    assert graph.status is ExtractionStatus.NEEDS_REVIEW
    assert graph.nodes == ()


def test_the_graph_serialises_with_its_sources_and_reasons() -> None:
    payload = _graph(confidence=0.2).to_dict()

    assert payload["image_id"] == "i1"
    assert payload["document_id"] == "d1"
    assert payload["status"] == "needs_review"
    assert payload["review_reasons"]
    assert isinstance(payload["nodes"], list)


# --------------------------------------------------------------------------- #
# The gallery: completeness, not relevance
# --------------------------------------------------------------------------- #


@pytest.fixture
def gallery_corpus(tmp_path: Path) -> Corpus:
    """A document with 30 figures, so pagination has something to paginate."""
    images_root = tmp_path / "images"
    images_root.mkdir()

    body = "Step text that runs on for a while to separate the figures. "
    pieces: list[str] = ["# Procedure\n\n"]
    for index in range(30):
        pieces.append(f"{body}Step {index}.\n\n<PIC>\n\n")
    text = "".join(pieces)
    offsets = [index for index in range(len(text)) if text.startswith("<PIC>", index)]

    images = []
    for index, offset in enumerate(offsets):
        image_id = f"fig{index:02d}"
        (images_root / f"{image_id}.png").write_bytes(PNG)
        images.append(
            CorpusImage(
                image_id=image_id,
                document_id="d1",
                position_index=index,
                char_offset=offset,
                source_url=f"https://cdn.example.test/{image_id}.png" if index % 2 == 0 else None,
            )
        )

    document = CorpusDocument(document_id="d1", title="Procedure", text=text, source_id="demo-d1", images=tuple(images))
    bundle = CorpusBundle(
        corpus_id="demo",
        title="Demo",
        attribution="Demo",
        licence="CC0-1.0",
        documents=(document,),
        chunks=slice_chunks(document, target_chars=400, overlap_chars=40),
        manifest={"image_retrieval_mode": "context-window"},
    )
    export_bundle(bundle, tmp_path / "demo" / "build", images_root=images_root)
    corpus = CorpusCatalogue.load(tmp_path).corpus("demo")
    assert corpus is not None
    return corpus


def _hit(corpus: Corpus, document_id: str = "d1", rank: int = 1) -> DocumentHit:
    chunk = next(iter(corpus.chunks.values()))
    return DocumentHit(
        document_id=document_id,
        title=corpus.documents[document_id].title,
        score=1.0,
        passages=(
            CorpusPassage(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                corpus_id=corpus.corpus_id,
                text=chunk.text,
                score=1.0,
                offset_source=OffsetSource.BUNDLE,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                image_ids=chunk.image_ids,
            ),
        ),
        rank=rank,
    )


def test_the_gallery_holds_every_figure_of_the_document(gallery_corpus: Corpus) -> None:
    """Recall 1.0 by construction. This is the whole point."""
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)

    assert gallery.documents[0].total == 30
    assert len(gallery.documents[0].images) == 30
    assert set(gallery.image_ids) == set(gallery_corpus.images)


def test_the_gallery_is_in_document_order(gallery_corpus: Corpus) -> None:
    """The order figures appear in the manual is the order of the steps they illustrate."""
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)
    item = gallery.documents[0]

    assert item.ordered
    assert [image.position_index for image in item.images] == list(range(30))


def test_a_ranked_selection_does_not_shrink_the_gallery(gallery_corpus: Corpus) -> None:
    """The failure this exists to prevent: on the held-out split the ranking put the wanted
    figure outside the top five for 87 questions while having already found the right
    document in 74 of them."""
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], selected_image_ids=["fig03"], page_size=100)

    assert len(gallery.documents[0].images) == 30
    selected = [image for image in gallery.documents[0].images if image.selected]
    assert [image.image_id for image in selected] == ["fig03"]


def test_pagination_pages_rather_than_filters(gallery_corpus: Corpus) -> None:
    first = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page=1, page_size=12).documents[0]
    second = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page=2, page_size=12).documents[0]
    third = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page=3, page_size=12).documents[0]

    assert first.total == second.total == third.total == 30
    assert [len(page.images) for page in (first, second, third)] == [12, 12, 6]
    assert first.pages == 3
    assert first.truncated and third.truncated
    seen = [image.image_id for page in (first, second, third) for image in page.images]
    assert len(seen) == 30
    assert len(set(seen)) == 30, "no figure appears on two pages"


def test_a_page_past_the_end_is_empty_but_still_reports_the_total(gallery_corpus: Corpus) -> None:
    item = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page=99, page_size=12).documents[0]

    assert item.images == ()
    assert item.total == 30


def test_each_figure_is_linked_to_the_passage_that_contains_it(gallery_corpus: Corpus) -> None:
    """The nearest thing to a step link this corpus supports: there are no numbered steps,
    so a figure is linked to the passage whose character span contains it."""
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)

    linked = [image for image in gallery.documents[0].images if image.passage_id]
    assert linked, "at least some figures should resolve to a passage"
    for image in linked:
        chunk = gallery_corpus.chunk(image.passage_id)
        assert chunk is not None
        assert chunk.char_start <= image.char_offset < chunk.char_end


def test_a_figure_with_no_passage_link_is_still_shown(gallery_corpus: Corpus) -> None:
    """An unlinked correct figure is worth more than a hidden one."""
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)

    assert len(gallery.documents[0].images) == 30


def test_the_matched_passage_is_marked(gallery_corpus: Corpus) -> None:
    hit = _hit(gallery_corpus)
    matched_chunk = hit.passages[0].chunk_id

    gallery = build_gallery(gallery_corpus, [hit], page_size=100)

    marked = [image for image in gallery.documents[0].images if image.in_matched_passage]
    assert all(image.passage_id == matched_chunk for image in marked)


def test_every_gallery_figure_carries_a_url_and_its_position(gallery_corpus: Corpus) -> None:
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)

    for image in gallery.documents[0].images:
        assert image.url == f"/api/corpus-images/demo/{image.image_id}"
        trail = image.provenance
        assert trail.image_id == image.image_id
        assert trail.position_index == image.position_index
        assert trail.char_offset == image.char_offset


def test_a_figure_with_no_upstream_original_reports_none(gallery_corpus: Corpus) -> None:
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)
    by_id = {image.image_id: image for image in gallery.documents[0].images}

    assert by_id["fig00"].provenance.source_url is not None
    assert by_id["fig01"].provenance.source_url is None


def test_a_document_the_corpus_does_not_have_is_reported_not_invented(gallery_corpus: Corpus) -> None:
    ghost = DocumentHit(document_id="nope", title="Ghost", score=1.0, passages=(), rank=1)

    gallery = build_gallery(gallery_corpus, [ghost])

    assert gallery.documents == ()
    assert any("nope" in warning for warning in gallery.warnings)


def test_the_hard_cap_reports_what_it_omitted(gallery_corpus: Corpus) -> None:
    """The only way a figure can be missing, and it is counted so it cannot be mistaken
    for filtering."""
    hits = [_hit(gallery_corpus, rank=index + 1) for index in range(30)]

    gallery = build_gallery(gallery_corpus, hits, page_size=30)

    assert gallery.shown_images <= MAX_GALLERY_IMAGES
    assert gallery.omitted_by_cap > 0
    assert any("cap" in warning for warning in gallery.warnings)


def test_the_diagnostics_report_order_and_roles(gallery_corpus: Corpus) -> None:
    gallery = build_gallery(gallery_corpus, [_hit(gallery_corpus)], page_size=100)

    stats = gallery_diagnostics(gallery)

    assert stats["images_total"] == 30
    assert stats["images_shown"] == 30
    assert stats["all_pages_ordered"] is True
    assert stats["roles"] == {"other": 30}
    assert stats["role_examined_by_a_model"] == 0


def test_an_empty_result_yields_an_empty_gallery(gallery_corpus: Corpus) -> None:
    gallery = build_gallery(gallery_corpus, [])

    assert gallery.documents == ()
    assert gallery.total_images == 0
    assert gallery.warnings == ()


def test_a_figure_reused_at_two_slots_appears_once_and_order_holds(tmp_path: Path) -> None:
    """A real MRAMG-Manual fact, not a hypothetical.

    2,607 ``<PIC>`` slots carry 2,599 distinct ids, so five documents place the same image
    at two positions. The registry holds one record per id with its first position, so
    walking the raw slot list yields it again at a position already passed and the sequence
    runs backwards. Measured before the fix: 4 of 40 documents non-monotonic, and the
    gallery's "document order" claim was false for them.
    """
    images_root = tmp_path / "images"
    images_root.mkdir()
    for name in ("a", "b", "c"):
        (images_root / f"{name}.png").write_bytes(PNG)

    body = "Some step text here to separate the markers. "
    text = f"# Doc\n\n{body}\n\n<PIC>\n\n{body}\n\n<PIC>\n\n{body}\n\n<PIC>\n\n{body}\n\n<PIC>\n"
    offsets = [index for index in range(len(text)) if text.startswith("<PIC>", index)]

    document = CorpusDocument(
        document_id="d1",
        title="Doc",
        text=text,
        source_id="demo-d1",
        images=(
            CorpusImage(image_id="a", document_id="d1", position_index=0, char_offset=offsets[0]),
            CorpusImage(image_id="b", document_id="d1", position_index=1, char_offset=offsets[1]),
            CorpusImage(image_id="c", document_id="d1", position_index=2, char_offset=offsets[2]),
            # The reuse: "a" again, at the fourth slot.
            CorpusImage(image_id="a", document_id="d1", position_index=3, char_offset=offsets[3]),
        ),
    )
    bundle = CorpusBundle(
        corpus_id="demo",
        title="Demo",
        attribution="Demo",
        licence="CC0-1.0",
        documents=(document,),
        chunks=slice_chunks(document, target_chars=200, overlap_chars=20),
        manifest={"image_retrieval_mode": "context-window"},
    )
    export_bundle(bundle, tmp_path / "demo" / "build", images_root=images_root)
    corpus = CorpusCatalogue.load(tmp_path).corpus("demo")
    assert corpus is not None

    figures = corpus.images_for("d1")
    ids = [record.image_id for record in figures]
    positions = [record.position_index for record in figures]

    assert ids == ["a", "b", "c"], "the reused figure appears once"
    assert positions == sorted(positions), "and the sequence never runs backwards"

    gallery = build_gallery(corpus, [_hit(corpus)], page_size=100)
    assert gallery.documents[0].ordered

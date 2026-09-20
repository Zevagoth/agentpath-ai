"""Every figure of the documents that won, in order, with nothing hidden by a ranking.

This exists because ranked image selection is the wrong tool for one of the two kinds of
figure a manual contains, and the Stage 2B1 evaluation is what showed it.

A UI screenshot with an arrow on it does not need to be *chosen*. Once the right procedure
is found, all of its screenshots are relevant and their order is the order of the steps.
Ranking them can only do harm: on the held-out split, context-window ranking put the
wanted figure outside the top five for 87 questions, and in 74 of those it had already
found the right document. A ranking that discards a correct image from a correct document
is strictly worse than showing the document's images.

So the two live side by side and mean different things:

``images``  - the ranked *selection*. Which figure does this question appear to be about?
              Precision matters; it may be empty; it is what Image Recall is measured on.
``gallery`` - the *complete set* for the documents retrieved, in document order. Recall is
              1.0 by construction. Nothing is filtered, ever.

The gallery is the answer to "show me what this procedure looks like". The selection is
the answer to "which picture is my question about". Presenting only the second was the
design mistake; presenting only the first would hide the reasoning.

Workflow diagrams are marked, not hidden. A diagram whose structure was extracted is
flagged so the console can offer the extracted path alongside it, and a diagram whose
extraction needs review is still shown as a picture, because the picture is the source of
truth either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentpath.corpus import Corpus, CorpusImageRecord
from agentpath.corpus_retrieval.ports import DocumentHit
from agentpath.provenance import ImageProvenance

logger = structlog.get_logger(__name__)

#: Images per page when a document has too many to show at once. A page, not a cut: every
#: image stays in the result and ``total`` says how many there are. MRAMG-Manual's busiest
#: document holds 273 figures, which is a scrolling problem and not a relevance problem.
DEFAULT_PAGE_SIZE = 24

#: Refuse to assemble more than this in one response. A guard against a pathological
#: request, not a filter: the cap is reported when it bites.
MAX_GALLERY_IMAGES = 600


@dataclass(frozen=True, slots=True)
class GalleryImage:
    """One figure, with where it sits and what is known about it."""

    image_id: str
    document_id: str
    #: 0-based ordinal among its document's figures. The display order.
    position_index: int
    char_offset: int
    url: str
    source_url: str | None
    caption: str | None
    role: str
    role_method: str
    role_confidence: float
    #: True when this figure's placeholder falls inside a passage the retrieval matched,
    #: which is the only step link this corpus supports. See :func:`build_gallery`.
    in_matched_passage: bool = False
    #: The chunk whose span contains this figure, when one does. The nearest thing to a
    #: "step" for a corpus of prose manuals.
    passage_id: str | None = None
    section: str | None = None
    #: True when the figure was also chosen by the ranked selection, so the UI can mark it
    #: without the two lists having to be diffed client-side.
    selected: bool = False

    @property
    def provenance(self) -> ImageProvenance:
        return ImageProvenance(
            image_id=self.image_id,
            served_url=self.url,
            source_url=self.source_url,
            document_id=self.document_id,
            position_index=self.position_index,
            char_offset=self.char_offset,
            caption=self.caption,
        )


@dataclass(frozen=True, slots=True)
class DocumentGallery:
    """Every figure of one document, in order, paginated but never filtered."""

    document_id: str
    title: str
    rank: int
    #: Total figures in this document, regardless of what this page holds.
    total: int
    images: tuple[GalleryImage, ...] = ()
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE
    #: Set when the document has more figures than this page shows.
    truncated: bool = False

    @property
    def pages(self) -> int:
        if self.page_size <= 0:
            return 1
        return max(1, -(-self.total // self.page_size))

    @property
    def ordered(self) -> bool:
        """Whether this page's figures are in document order. Asserted by the evaluation."""
        indexes = [image.position_index for image in self.images]
        return indexes == sorted(indexes)


@dataclass(frozen=True, slots=True)
class Gallery:
    """The complete figure set for one search, document by document."""

    corpus_id: str
    documents: tuple[DocumentGallery, ...] = ()
    #: Figures omitted only by the hard cap, never by relevance.
    omitted_by_cap: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_images(self) -> int:
        return sum(item.total for item in self.documents)

    @property
    def shown_images(self) -> int:
        return sum(len(item.images) for item in self.documents)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(image.image_id for item in self.documents for image in item.images)


def build_gallery(
    corpus: Corpus,
    documents: Sequence[DocumentHit],
    *,
    selected_image_ids: Sequence[str] = (),
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Gallery:
    """Assemble the complete figure set for *documents*, best document first.

    Order is the corpus's own: ``position_index``, which is which ``<PIC>`` marker the
    figure fills. That is the order a reader of the manual meets them in, and therefore the
    order of the steps they illustrate.

    Step linking is honest about what this corpus supports. There are no numbered steps to
    link to - these are prose manuals, not the procedure YAML of Stage 2A - so a figure is
    linked to the *passage* whose character span contains it, and to that passage's section
    heading. When the retrieval matched that passage, ``in_matched_passage`` says so. When
    no such link exists the figure is still shown, with the link fields left empty, because
    an unlinked correct figure is worth more than a hidden one.
    """
    selected = set(selected_image_ids)
    galleries: list[DocumentGallery] = []
    warnings: list[str] = []
    omitted = 0
    budget = MAX_GALLERY_IMAGES

    for hit in documents:
        record = corpus.document(hit.document_id)
        if record is None:
            warnings.append(f"document {hit.document_id!r} is not in corpus {corpus.corpus_id!r}")
            continue

        figures = corpus.images_for(hit.document_id)
        matched = _matched_passages(hit)
        containing = _containing_chunks(corpus, hit.document_id)

        start = max(0, (page - 1) * page_size) if page_size > 0 else 0
        window = figures[start : start + page_size] if page_size > 0 else figures

        if budget <= 0:
            omitted += len(window)
            continue
        if len(window) > budget:
            omitted += len(window) - budget
            window = window[:budget]
        budget -= len(window)

        images = tuple(_gallery_image(figure, hit.document_id, containing, matched, selected) for figure in window)
        galleries.append(
            DocumentGallery(
                document_id=hit.document_id,
                title=hit.title or record.title,
                rank=hit.rank,
                total=len(figures),
                images=images,
                page=max(1, page),
                page_size=page_size,
                truncated=len(figures) > len(images),
            )
        )

    if omitted:
        warnings.append(
            f"{omitted} figure(s) omitted by the {MAX_GALLERY_IMAGES}-image cap on one response, "
            "not by relevance; request a later page to see them"
        )

    logger.debug(
        "corpus.gallery_built",
        corpus=corpus.corpus_id,
        documents=len(galleries),
        images=sum(len(item.images) for item in galleries),
        omitted=omitted,
    )
    return Gallery(
        corpus_id=corpus.corpus_id,
        documents=tuple(galleries),
        omitted_by_cap=omitted,
        warnings=tuple(warnings),
    )


def _gallery_image(
    figure: CorpusImageRecord,
    document_id: str,
    containing: list[tuple[int, int, str, str | None]],
    matched: set[str],
    selected: set[str],
) -> GalleryImage:
    passage_id: str | None = None
    section: str | None = None
    for start, end, chunk_id, chunk_section in containing:
        if start <= figure.char_offset < end:
            passage_id = chunk_id
            section = chunk_section
            break

    return GalleryImage(
        image_id=figure.image_id,
        document_id=document_id,
        position_index=figure.position_index,
        char_offset=figure.char_offset,
        url=figure.served_url,
        source_url=figure.source_url,
        caption=figure.caption,
        role=figure.role,
        role_method=figure.role_method,
        role_confidence=figure.role_confidence,
        in_matched_passage=passage_id in matched if passage_id else False,
        passage_id=passage_id,
        section=section,
        selected=figure.image_id in selected,
    )


def _matched_passages(hit: DocumentHit) -> set[str]:
    """Character spans the retrieval actually matched, as bundle chunk ids when known.

    The hybrid path's chunk ids are RAGFlow's, not ours, so they will not appear in the
    bundle. The offsets are what carry over, and :func:`_containing_chunks` resolves a
    figure to a bundle chunk by offset - so the link works for both methods without either
    knowing about the other.
    """
    return {passage.chunk_id for passage in hit.passages if passage.chunk_id}


def _containing_chunks(corpus: Corpus, document_id: str) -> list[tuple[int, int, str, str | None]]:
    """``(start, end, chunk_id, section)`` for the document's chunks, in order."""
    return sorted(
        (
            (chunk.char_start, chunk.char_end, chunk.chunk_id, chunk.section)
            for chunk in corpus.chunks.values()
            if chunk.document_id == document_id
        ),
        key=lambda row: row[0],
    )


def gallery_diagnostics(gallery: Gallery) -> dict[str, Any]:
    """Numbers the evaluation checks, computed once here so both sides agree."""
    roles: dict[str, int] = {}
    examined = 0
    linked = 0
    for item in gallery.documents:
        for image in item.images:
            roles[image.role] = roles.get(image.role, 0) + 1
            if image.role_method != "prefilter":
                examined += 1
            if image.passage_id:
                linked += 1
    return {
        "documents": len(gallery.documents),
        "images_total": gallery.total_images,
        "images_shown": gallery.shown_images,
        "omitted_by_cap": gallery.omitted_by_cap,
        "all_pages_ordered": all(item.ordered for item in gallery.documents),
        "roles": dict(sorted(roles.items())),
        "role_examined_by_a_model": examined,
        "linked_to_a_passage": linked,
    }


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_GALLERY_IMAGES",
    "DocumentGallery",
    "Gallery",
    "GalleryImage",
    "build_gallery",
    "gallery_diagnostics",
]

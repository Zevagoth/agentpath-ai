"""The provenance contract, weighted towards what it refuses.

A provenance link is the one place in this system where a third-party string is
handed to a browser as something to open. The value of this module is therefore not
the links it builds but the ones it will not build, so most of what follows is
rejection.
"""

from __future__ import annotations

import pytest

from agentpath.provenance import (
    DANGEROUS_URL_SCHEMES,
    ImageProvenance,
    Provenance,
    SourceKind,
    SourceLocator,
    build_image_provenance,
    build_provenance,
    safe_external_url,
    safe_source_id,
)

# --------------------------------------------------------------------------- #
# URLs it refuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hostile",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "  javascript:alert(1)  ",
        "jAvAsCrIpT:alert(document.cookie)",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "blob:https://example.com/abc",
        "about:blank",
        # A link into the serving machine's filesystem is not provenance.
        "file:///etc/passwd",
        "file://localhost/etc/shadow",
    ],
)
def test_dangerous_schemes_are_refused(hostile: str) -> None:
    assert safe_external_url(hostile) is None


def test_every_scheme_on_the_dangerous_list_is_actually_refused() -> None:
    """The list is not decoration: each entry is exercised."""
    for scheme in DANGEROUS_URL_SCHEMES:
        assert safe_external_url(f"{scheme}:whatever") is None


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "   ",
        "not a url",
        "://missing-scheme",
        # No host: these are how a scheme-only value parses.
        "http://",
        "https://",
        "http:///etc/passwd",
        # A newline in an href can close the attribute and open another.
        "https://example.com/a\nonmouseover=alert(1)",
        "https://example.com/a\r\nSet-Cookie: x",
        "https://example.com/\tx",
        "https://example.com/\x00",
    ],
)
def test_malformed_urls_are_refused(malformed: str) -> None:
    assert safe_external_url(malformed) is None


def test_credentials_in_a_citation_are_refused() -> None:
    """A citation has no business carrying a password."""
    assert safe_external_url("https://user:secret@example.com/doc") is None
    assert safe_external_url("https://user@example.com/doc") is None


def test_an_absurdly_long_url_is_refused() -> None:
    assert safe_external_url("https://example.com/" + "a" * 4000) is None


@pytest.mark.parametrize("wrong_type", [None, 42, 3.5, b"https://example.com", ["https://x.com"], {}])
def test_non_strings_are_refused(wrong_type: object) -> None:
    assert safe_external_url(wrong_type) is None


@pytest.mark.parametrize(
    "good",
    [
        "https://example.com",
        "https://example.com/manual.pdf",
        "http://127.0.0.1:8000/doc",
        "https://cdn-mineru.openxlab.org.cn/model-mineru/prod/11ccff.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/X.jpg/800px-X.jpg",
        "https://example.com/a?b=c&d=e#frag",
    ],
)
def test_plain_http_and_https_are_accepted_unchanged(good: str) -> None:
    assert safe_external_url(good) == good


def test_a_url_is_never_repaired() -> None:
    """A value that needs repairing is not one to vouch for."""
    assert safe_external_url("example.com/doc") is None
    assert safe_external_url("//example.com/doc") is None


# --------------------------------------------------------------------------- #
# Local ids that cannot become paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "traversal",
    [
        "..",
        "../..",
        "../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "/etc/passwd",
        "etc/passwd",
        "a/b",
        "a\\b",
        "..\\..\\windows\\system32",
        "C:\\Windows",
        "\\\\server\\share",
        "doc.md",
        "NW-HTL-014.md",
        ".hidden",
        "a:b",
        "a\x00b",
        "a b",
        "",
        "-leading-dash",
        "_leading-underscore",
    ],
)
def test_a_source_id_can_never_look_like_a_path(traversal: str) -> None:
    """Shape alone rules out every path form, before any registry lookup."""
    assert safe_source_id(traversal) is None


@pytest.mark.parametrize("good", ["NW-HTL-014", "manual-50000", "Manual04_0", "a", "A1_b-2"])
def test_flat_opaque_ids_are_accepted(good: str) -> None:
    assert safe_source_id(good) == good


def test_a_source_id_is_length_bounded() -> None:
    assert safe_source_id("a" * 128) is not None
    assert safe_source_id("a" * 129) is None


# --------------------------------------------------------------------------- #
# A missing link is stated, not invented
# --------------------------------------------------------------------------- #


def test_no_source_yields_no_link_and_a_reason() -> None:
    provenance = build_provenance(corpus_id="mramg-manual", document_id="50000")

    assert provenance.kind is SourceKind.NONE
    assert provenance.link() is None
    assert provenance.link_absent_reason is not None
    assert "no original source" in provenance.link_absent_reason.lower()
    assert provenance.opens_at_position is False


def test_a_refused_url_leaves_no_link_rather_than_a_broken_one() -> None:
    """Ingestion must not fail on a hostile URL, and must not keep it either."""
    provenance = build_provenance(corpus_id="mramg-manual", source_url="javascript:alert(1)")

    assert provenance.source_url is None
    assert provenance.link() is None
    assert provenance.link_absent_reason is not None


def test_a_refused_source_id_leaves_no_link() -> None:
    provenance = build_provenance(corpus_id="synthetic-procedures", source_id="../../etc/passwd")

    assert provenance.source_id is None
    assert provenance.link() is None


# --------------------------------------------------------------------------- #
# Opening at the position, not at page one
# --------------------------------------------------------------------------- #


def test_a_local_link_opens_at_the_character_offset() -> None:
    provenance = build_provenance(
        corpus_id="mramg-manual",
        source_id="manual-50000",
        locator=SourceLocator(char_start=4820, char_end=5120),
    )

    assert provenance.link() == "/api/sources/manual-50000#char=4820"
    assert provenance.opens_at_position is True


def test_an_explicit_anchor_wins_over_an_offset() -> None:
    provenance = build_provenance(
        corpus_id="synthetic-procedures",
        source_id="NW-HTL-014",
        locator=SourceLocator(anchor="step-5", char_start=99, section="## Steps > ### Step 5"),
    )

    assert provenance.link() == "/api/sources/NW-HTL-014#step-5"


def test_a_local_link_without_a_locator_opens_the_document_and_says_so() -> None:
    provenance = build_provenance(corpus_id="mramg-manual", source_id="manual-50000")

    assert provenance.link() == "/api/sources/manual-50000"
    # No position was recorded, so the link is honest about landing at the top.
    assert provenance.opens_at_position is False


def test_an_external_link_gets_a_fragment_only_when_the_corpus_gave_an_anchor() -> None:
    """A fragment we invented would be a guess about someone else's document."""
    with_anchor = build_provenance(
        corpus_id="mramg-manual",
        source_url="https://example.com/manual",
        locator=SourceLocator(anchor="section-3"),
    )
    assert with_anchor.link() == "https://example.com/manual#section-3"
    assert with_anchor.opens_at_position is True

    offsets_only = build_provenance(
        corpus_id="mramg-manual",
        source_url="https://example.com/manual",
        locator=SourceLocator(char_start=4820),
    )
    assert offsets_only.link() == "https://example.com/manual"
    assert offsets_only.opens_at_position is False


def test_an_existing_fragment_on_an_external_url_is_left_alone() -> None:
    provenance = build_provenance(
        corpus_id="mramg-manual",
        source_url="https://example.com/manual#page=4",
        locator=SourceLocator(anchor="section-3"),
    )
    assert provenance.link() == "https://example.com/manual#page=4"


def test_the_local_prefix_is_injectable_so_the_route_can_move() -> None:
    provenance = build_provenance(corpus_id="mramg-manual", source_id="manual-50000")
    assert provenance.link(local_prefix="/api/v2/docs") == "/api/v2/docs/manual-50000"


# --------------------------------------------------------------------------- #
# One contract, two corpora
# --------------------------------------------------------------------------- #


def test_a_procedure_entry_carries_version_and_effective_date() -> None:
    provenance = build_provenance(
        corpus_id="synthetic-procedures",
        dataset_id="ds-1",
        document_id="doc-1",
        document_title="Property unable to accommodate a confirmed guest",
        procedure_id="NW-HTL-014",
        procedure_version="2.1",
        effective_date="2026-03-01",
        step_id="NW-HTL-014#S05",
        source_id="NW-HTL-014",
        locator=SourceLocator(section="## Steps > ### Step 5", anchor="step-5"),
    )

    assert provenance.citation == "NW-HTL-014 v2.1 · NW-HTL-014#S05"
    assert provenance.locator.summary.startswith("## Steps")
    assert provenance.kind is SourceKind.LOCAL_DOCUMENT


def test_an_external_document_entry_needs_no_version_to_be_citable() -> None:
    """MRAMG has no versions or effective dates, and must still cite cleanly."""
    provenance = build_provenance(
        corpus_id="mramg-manual",
        dataset_id="ds-2",
        document_id="50000",
        document_title="225B Operator's manual",
        chunk_id="50000#c3",
        source_id="manual-50000",
        locator=SourceLocator(char_start=1200, char_end=1900),
    )

    assert provenance.procedure_version is None
    assert provenance.effective_date is None
    assert provenance.citation == "mramg-manual · 225B Operator's manual · 50000#c3"
    assert "chars 1200-1900" in provenance.locator.summary


def test_a_locator_with_nothing_set_says_so_rather_than_implying_precision() -> None:
    empty = SourceLocator()
    assert empty.is_precise is False
    assert empty.summary == "position not recorded"
    assert empty.fragment() == ""


# --------------------------------------------------------------------------- #
# Image provenance
# --------------------------------------------------------------------------- #


def test_an_image_records_where_it_sits_in_its_document() -> None:
    image = build_image_provenance(
        image_id="Manual04_0",
        served_url="/api/corpus-images/mramg-manual/Manual04_0",
        source_url="https://cdn-mineru.openxlab.org.cn/model-mineru/prod/11ccff.jpg",
        document_id="50000",
        position_index=0,
        char_offset=0,
    )

    assert image.has_original is True
    assert image.position_summary == "image 1 in the document, at character 0"


def test_an_image_with_no_upstream_url_says_so() -> None:
    """299 of MRAMG-Manual's 2,599 images have no recorded URL."""
    image = build_image_provenance(image_id="Manual04_9", document_id="50000", position_index=9)

    assert image.source_url is None
    assert image.has_original is False
    assert image.position_summary == "image 10 in the document"


def test_an_image_url_is_validated_like_any_other() -> None:
    image = build_image_provenance(image_id="x", source_url="javascript:alert(1)")
    assert image.source_url is None


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t ", None])
def test_a_blank_caption_reads_as_absent(blank: str | None) -> None:
    """Every MRAMG-Manual caption is "", and that must not read as a caption."""
    assert build_image_provenance(image_id="x", caption=blank).caption is None


def test_a_real_caption_survives_unchanged() -> None:
    assert build_image_provenance(image_id="x", caption="A diagram").caption == "A diagram"


def test_images_ride_on_the_provenance_they_belong_to() -> None:
    provenance = build_provenance(
        corpus_id="mramg-manual",
        document_id="50000",
        images=(
            build_image_provenance(image_id="Manual04_0", document_id="50000", position_index=0),
            build_image_provenance(image_id="Manual04_1", document_id="50000", position_index=1),
        ),
    )

    assert [image.image_id for image in provenance.images] == ["Manual04_0", "Manual04_1"]


def test_the_contract_is_frozen() -> None:
    """Nothing downstream may edit a provenance trail it was handed."""
    provenance = build_provenance(corpus_id="mramg-manual")
    with pytest.raises((AttributeError, TypeError)):
        provenance.corpus_id = "synthetic-procedures"  # type: ignore[misc]

    image = ImageProvenance(image_id="x")
    with pytest.raises((AttributeError, TypeError)):
        image.image_id = "y"  # type: ignore[misc]

    assert isinstance(Provenance(corpus_id="x"), Provenance)

"""Tests for language detection, normalization, the lexicon, and the pipeline.

These tests encode the product problem directly: an agent typing "hotel has no
room", "overboking", "nonref", or "الفندق ما عنده غرف" must end up with a query
that can reach the right procedure. Each case below is one of those situations.
"""

from __future__ import annotations

import pytest

from agentpath.domain import Language
from agentpath.query.language import arabic_ratio, detect_language
from agentpath.query.lexicon import LexiconError, load_lexicon
from agentpath.query.normalize import normalize, strip_hyphens, tokenize
from agentpath.query.pipeline import MAX_EXPANSION_TERMS, QueryPreprocessor


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon()


@pytest.fixture(scope="module")
def preprocessor(lexicon):
    return QueryPreprocessor(lexicon)


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hotel has no room", Language.ENGLISH),
        ("The property cannot accommodate the guest", Language.ENGLISH),
        ("الفندق ما عنده غرف", Language.ARABIC),
        ("العميل بدو يلغي الحجز", Language.ARABIC),
        ("overbooking بالفندق", Language.MIXED),
        ("الفندق overbooking والضيف واصل", Language.MIXED),
        # A bare procedure reference is Latin script; English is the useful call
        # because the corpus pivot is English and only ar/mixed flip the UI to RTL.
        ("NW-HTL-014", Language.ENGLISH),
        ("", Language.UNKNOWN),
        ("   ", Language.UNKNOWN),
        ("12345", Language.UNKNOWN),
        ("---", Language.UNKNOWN),
    ],
)
def test_detect_language(text: str, expected: Language) -> None:
    assert detect_language(text) == expected


def test_arabic_ratio_ignores_digits_and_punctuation() -> None:
    assert arabic_ratio("NW-HTL-014") == 0.0
    assert arabic_ratio("الفندق 2026") == 1.0
    assert 0.0 < arabic_ratio("hotel الفندق") < 1.0


def test_reference_number_does_not_dilute_arabic_detection() -> None:
    assert detect_language("NW-HTL-014 الفندق ما عنده غرف") == Language.ARABIC


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hotel  has NO room!! ", "hotel has no room"),
        ("HOTEL HAS NO ROOM", "hotel has no room"),
        ("non-refundable", "non-refundable"),
        ("Hôtel", "hotel"),
        ("NW-HTL-014", "nw-htl-014"),
    ],
)
def test_normalize_english(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("إلغاء الحجز", "الغاء الحجز"),
        ("الغاء الحجز", "الغاء الحجز"),
        ("إلغــاء", "الغاء"),
        ("إِلْغَاء", "الغاء"),
        ("مسترد", "مسترد"),
        ("غرفة", "غرفه"),
        ("الفندق ٢٠٢٦", "الفندق 2026"),
        ("مُكرَّر", "مكرر"),
    ],
)
def test_normalize_arabic_folds_orthographic_variation(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_arabic_spelling_variants_converge() -> None:
    """The point of normalization: four spellings, one key."""
    variants = ["إلغاء", "الغاء", "إلغــاء", "إِلْغَاء"]
    assert len({normalize(variant) for variant in variants}) == 1


def test_tokenize_keeps_hyphenated_terms_whole() -> None:
    assert tokenize(normalize("non-refundable booking")) == ["non-refundable", "booking"]


def test_strip_hyphens_produces_the_spaced_variant() -> None:
    assert strip_hyphens("non-refundable") == "non refundable"


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #


def test_lexicon_loads_with_concepts_and_hints(lexicon) -> None:
    assert lexicon.version
    assert len(lexicon.concepts) >= 7
    assert set(lexicon.product_line_hints) >= {"hotels", "flights", "cancellations", "refunds", "escalations"}


def test_lexicon_missing_file_raises() -> None:
    from pathlib import Path

    with pytest.raises(LexiconError, match="not found"):
        load_lexicon(Path("/nonexistent/lexicon.yaml"))


@pytest.mark.parametrize(
    ("typo", "corrected"),
    [
        ("overboking", "overbooking"),
        ("cancelation", "cancellation"),
        ("refnud", "refund"),
        ("duplicat", "duplicate"),
        ("accomodate", "accommodate"),
        ("hotle", "hotel"),
    ],
)
def test_lexicon_corrects_misspellings(lexicon, typo: str, corrected: str) -> None:
    text, corrections = lexicon.correct(normalize(f"{typo} issue"))
    assert corrected in text
    assert (typo, corrected) in corrections


@pytest.mark.parametrize(
    ("shorthand", "expanded"),
    [
        ("nonref", "non-refundable"),
        ("cxl", "cancellation"),
        ("dup", "duplicate"),
        ("bkg", "booking"),
        ("pax", "guest"),
    ],
)
def test_lexicon_expands_abbreviations(lexicon, shorthand: str, expanded: str) -> None:
    text, corrections = lexicon.correct(normalize(f"{shorthand} case"))
    assert expanded in text
    assert any(after == expanded for _, after in corrections)


def test_multiword_abbreviation_wins_over_single_token(lexicon) -> None:
    """"non ref" must become non-refundable, not "non refund"."""
    text, _ = lexicon.correct(normalize("non ref booking"))
    assert "non-refundable" in text


@pytest.mark.parametrize(
    ("query", "expected_concept"),
    [
        ("hotel has no room", "overbooking"),
        ("the hotel is full and turned the guest away", "overbooking"),
        ("property cannot find the booking", "booking_not_found"),
        ("customer booked twice", "duplicate_booking"),
        ("non-refundable rate cancellation", "non_refundable"),
        ("where is my refund", "delayed_refund"),
        ("الفندق ما عنده غرف", "overbooking"),
        ("ما لاقو الحجز", "booking_not_found"),
        ("حجز مكرر", "duplicate_booking"),
        ("الاسترداد متاخر", "delayed_refund"),
        ("غير قابل للاستراد", "non_refundable"),
    ],
)
def test_lexicon_matches_concepts_in_both_languages(lexicon, query: str, expected_concept: str) -> None:
    matches = lexicon.match_concepts(normalize(query))
    assert expected_concept in {match.concept_id for match in matches}


def test_single_token_concepts_do_not_match_inside_longer_words(lexicon) -> None:
    """"ref" inside "reference" must not be read as "refund"."""
    matches = lexicon.match_concepts(normalize("check the reference number"))
    assert "delayed_refund" not in {match.concept_id for match in matches}


def test_english_equivalents_are_produced_for_arabic_concepts(lexicon) -> None:
    matches = lexicon.match_concepts(normalize("الفندق ما عنده غرف"))
    equivalents = lexicon.english_equivalents(matches)
    assert equivalents
    assert any("overbooking" in term for term in equivalents)


@pytest.mark.parametrize(
    ("query", "expected_line"),
    [
        ("hotel has no room tonight", "hotels"),
        ("guest at reception with no booking", "hotels"),
        ("where is my refund", "refunds"),
        ("customer booked twice", "cancellations"),
    ],
)
def test_product_line_classification(lexicon, query: str, expected_line: str) -> None:
    normalized = normalize(query)
    matches = lexicon.match_concepts(normalized)
    assert lexicon.classify_product_line(normalized, matches) == expected_line


def test_product_line_is_none_when_there_is_no_signal(lexicon) -> None:
    normalized = normalize("please help me with this case")
    matches = lexicon.match_concepts(normalized)
    assert lexicon.classify_product_line(normalized, matches) is None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def test_pipeline_preserves_the_original_query(preprocessor) -> None:
    prepared = preprocessor.prepare("  Hotel has NO room  ")
    assert prepared.original == "Hotel has NO room"
    assert prepared.search_text.startswith("Hotel has NO room")


def test_pipeline_adds_english_equivalents_for_an_arabic_query(preprocessor) -> None:
    prepared = preprocessor.prepare("الفندق ما عنده غرف والضيف واصل")

    assert prepared.language == Language.ARABIC
    assert prepared.original in prepared.search_text
    assert prepared.expansions, "an Arabic query needs English terms to reach the lexical half"
    assert any("overbooking" in term for term in prepared.expansions)
    assert "overbooking" in prepared.scenario_tags
    assert prepared.detected_product_line == "hotels"


def test_pipeline_corrects_typos_and_reports_them(preprocessor) -> None:
    prepared = preprocessor.prepare("overboking at the hotle")
    assert ("overboking", "overbooking") in prepared.corrections
    assert ("hotle", "hotel") in prepared.corrections
    assert "overbooking" in prepared.normalized


def test_pipeline_tags_scenarios_for_an_english_query(preprocessor) -> None:
    prepared = preprocessor.prepare("guest is at the property and the hotel cannot host them")
    assert {"overbooking", "guest_at_property"} & set(prepared.scenario_tags)


def test_explicit_product_line_overrides_detection(preprocessor) -> None:
    prepared = preprocessor.prepare("where is my refund", product_line="refunds")
    assert prepared.requested_product_line == "refunds"
    assert prepared.effective_product_line == "refunds"

    prepared = preprocessor.prepare("hotel has no room", product_line="refunds")
    assert prepared.detected_product_line == "hotels"
    assert prepared.effective_product_line == "refunds"


def test_expansions_never_repeat_terms_already_in_the_query(preprocessor) -> None:
    prepared = preprocessor.prepare("overbooking")
    assert "overbooking" not in prepared.expansions


def test_expansion_count_is_capped(preprocessor) -> None:
    prepared = preprocessor.prepare(
        "overbooking duplicate booking non-refundable refund delayed escalate supervisor guest at reception"
    )
    assert len(prepared.expansions) <= MAX_EXPANSION_TERMS


def test_empty_query_is_flagged_not_crashed(preprocessor) -> None:
    prepared = preprocessor.prepare("   ")
    assert prepared.is_empty
    assert prepared.language == Language.UNKNOWN


def test_procedure_id_query_survives_normalization(preprocessor) -> None:
    """Searching by reference number must keep the number intact."""
    prepared = preprocessor.prepare("NW-HTL-014")
    assert "nw-htl-014" in prepared.normalized
    assert "NW-HTL-014" in prepared.search_text


def test_preparation_is_deterministic(preprocessor) -> None:
    first = preprocessor.prepare("الفندق ما عنده غرف")
    second = preprocessor.prepare("الفندق ما عنده غرف")
    assert first == second

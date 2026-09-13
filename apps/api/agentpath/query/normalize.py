"""Text normalization for Arabic and English customer-service queries.

Normalization is applied to the query *and* to every lexicon entry, so the two
sides always meet in the same shape. It is intentionally lossless in meaning and
lossy only in orthography.

Arabic needs this more than English: the same word arrives as "إلغاء", "الغاء",
"إلغــاء" or "إِلْغَاء" depending on keyboard, habit and copy-paste source. Without
folding those together, lexical retrieval misses matches a human would call
identical.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat, sukun, shadda, superscript alef and Quranic annotation marks.
_ARABIC_DIACRITICS = re.compile(
    "["
    "\u064b-\u0652"  # fathatan .. sukun
    "\u0653-\u0655"  # maddah, hamza above/below
    "\u0670"  # superscript alef
    "\u06d6-\u06ed"  # Quranic annotation signs
    "]"
)

_TATWEEL = "\u0640"

#: Alef forms -> bare alef; ya/alef-maqsura -> ya; ta marbuta -> ha;
#: hamza-on-waw -> waw. These are the folds Arabic search engines apply by
#: convention, and they map every spelling agents actually type onto one form.
_ARABIC_CHAR_FOLDS = str.maketrans(
    {
        "\u0622": "\u0627",  # آ
        "\u0623": "\u0627",  # أ
        "\u0625": "\u0627",  # إ
        "\u0671": "\u0627",  # ٱ
        "\u0649": "\u064a",  # ى
        "\u0629": "\u0647",  # ة
        "\u0624": "\u0648",  # ؤ
        "\u0626": "\u064a",  # ئ
        "\u06a9": "\u0643",  # ک (Persian kaf)
        "\u06cc": "\u064a",  # ی (Persian ya)
    }
)

#: Arabic-Indic and Eastern Arabic-Indic digits -> ASCII.
_DIGIT_FOLDS = str.maketrans(
    {
        **{chr(0x0660 + index): str(index) for index in range(10)},
        **{chr(0x06F0 + index): str(index) for index in range(10)},
    }
)

#: Punctuation that carries no retrieval signal. Hyphen and slash survive
#: because "non-refundable" and "check-in" are single concepts, and procedure
#: IDs look like NW-HTL-014.
_PUNCTUATION = re.compile(r"[^\w\s\-/]", re.UNICODE)

_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Fold Arabic orthographic variation into a single comparable form."""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ARABIC_CHAR_FOLDS)
    return text


def normalize_english(text: str) -> str:
    """Case-fold and strip accents so "Rebook" and "rebook" agree."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.casefold()


def normalize(text: str) -> str:
    """Full normalization pipeline used for both queries and lexicon entries.

    >>> normalize("  Hôtel  has NO room!! ")
    'hotel has no room'
    >>> normalize("إلغاء الحجز")
    'الغاء الحجز'
    >>> normalize("الغــاء ٢٠٢٦")
    'الغاء 2026'
    >>> normalize("NON-REFUNDABLE")
    'non-refundable'
    """
    if not text:
        return ""
    text = normalize_arabic(text)
    text = normalize_english(text)
    text = text.translate(_DIGIT_FOLDS)
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Split normalized text into tokens.

    Hyphenated terms stay whole: "non-refundable" is one concept, and splitting
    it would let "refundable" match a procedure that says the opposite.
    """
    return [token for token in _WHITESPACE.split(text) if token]


def strip_hyphens(text: str) -> str:
    """Variant with hyphens turned into spaces.

    Indexed documents and typed queries disagree about "non-refundable" versus
    "non refundable", so both forms are searched.
    """
    return _WHITESPACE.sub(" ", text.replace("-", " ")).strip()

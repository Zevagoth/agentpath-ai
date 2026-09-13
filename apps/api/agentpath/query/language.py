"""Language detection by script.

A statistical language detector would be overkill and non-deterministic here.
The distinction that matters for this product is Arabic script versus Latin
script, which the Unicode ranges answer exactly and in microseconds.

Agents mix languages constantly ("overbooking بالفندق"), so ``MIXED`` is a
first-class outcome rather than a rounding error: it tells the retrieval layer to
keep both the Arabic and the English surface forms in the search text, and tells
the UI not to force a text direction.
"""

from __future__ import annotations

import unicodedata

from agentpath.domain import Language

#: Arabic, Arabic Supplement, Arabic Extended-A, Arabic Presentation Forms.
_ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

#: Above this share of letters, the query counts as containing that script.
_PRESENCE_THRESHOLD = 0.15
#: Above this share, the query is treated as that script outright.
_DOMINANCE_THRESHOLD = 0.60


def _is_arabic(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _ARABIC_RANGES)


def arabic_ratio(text: str) -> float:
    """Share of letter characters that belong to the Arabic script.

    Digits, punctuation and whitespace are ignored, so "NW-HTL-014 الفندق"
    is not diluted by its reference number.
    """
    letters = [char for char in text if unicodedata.category(char).startswith("L")]
    if not letters:
        return 0.0
    arabic = sum(1 for char in letters if _is_arabic(char))
    return arabic / len(letters)


def detect_language(text: str) -> Language:
    """Classify the script of *text*.

    >>> detect_language("hotel has no room")
    <Language.ENGLISH: 'en'>
    >>> detect_language("الفندق ما عنده غرف")
    <Language.ARABIC: 'ar'>
    >>> detect_language("overbooking بالفندق")
    <Language.MIXED: 'mixed'>
    >>> detect_language("14")
    <Language.UNKNOWN: 'unknown'>

    A bare reference like ``NW-HTL-014`` is Latin script, so it reports
    ``ENGLISH``. That is the useful answer: the corpus pivot language is English,
    and the UI only switches to right-to-left for ``ARABIC`` and ``MIXED``.
    """
    if not text or not text.strip():
        return Language.UNKNOWN

    ratio = arabic_ratio(text)
    has_letters = any(unicodedata.category(char).startswith("L") for char in text)
    if not has_letters:
        # Reference numbers and codes carry no script signal.
        return Language.UNKNOWN

    latin_ratio = 1.0 - ratio
    if ratio >= _DOMINANCE_THRESHOLD:
        return Language.ARABIC
    if latin_ratio >= _DOMINANCE_THRESHOLD and ratio < _PRESENCE_THRESHOLD:
        return Language.ENGLISH
    if ratio >= _PRESENCE_THRESHOLD and latin_ratio >= _PRESENCE_THRESHOLD:
        return Language.MIXED
    return Language.ARABIC if ratio > latin_ratio else Language.ENGLISH

"""Query understanding: language detection, normalization, lexicon expansion.

Deliberately deterministic and model-free. Every expansion and every tag can be
traced to a line in ``data/synthetic/lexicon.yaml``, which keeps this stage fast
(sub-millisecond), testable, and explainable to the agent reading the result.
"""

from agentpath.query.language import arabic_ratio, detect_language
from agentpath.query.lexicon import Lexicon, load_lexicon
from agentpath.query.normalize import normalize, normalize_arabic, normalize_english
from agentpath.query.pipeline import QueryPreprocessor

__all__ = [
    "Lexicon",
    "QueryPreprocessor",
    "arabic_ratio",
    "detect_language",
    "load_lexicon",
    "normalize",
    "normalize_arabic",
    "normalize_english",
]

"""The query understanding pipeline.

Steps 1 through 5 of the AgentPath search pipeline:

1. Receive the original query (kept verbatim).
2. Detect its language.
3. Normalize spelling and customer-service terminology.
4. Classify the likely product line and scenario.
5. Preserve the original and add English search equivalents.

Retrieval (steps 6 onward) happens behind the ``Retriever`` port.
"""

from __future__ import annotations

from agentpath.domain import Language, PreparedQuery
from agentpath.query.language import detect_language
from agentpath.query.lexicon import Lexicon
from agentpath.query.normalize import normalize, strip_hyphens

#: Cap on appended expansion terms. Beyond this, extra terms dilute BM25 scoring
#: more than they help recall.
MAX_EXPANSION_TERMS = 8


class QueryPreprocessor:
    """Turns a raw agent query into a :class:`PreparedQuery`."""

    def __init__(self, lexicon: Lexicon) -> None:
        self._lexicon = lexicon

    @property
    def lexicon_version(self) -> str:
        return self._lexicon.version

    def prepare(self, query: str, *, product_line: str | None = None) -> PreparedQuery:
        """Run query understanding.

        The original text always leads the search string. RAGFlow tokenizes and
        weights terms, so putting the agent's own words first keeps the query's
        intent dominant and the appended equivalents supporting rather than
        competing.
        """
        original = query.strip()
        language = detect_language(original)
        normalized = normalize(original)

        corrected, corrections = self._lexicon.correct(normalized)
        concepts = self._lexicon.match_concepts(corrected)
        scenario_tags = self._lexicon.scenario_tags(concepts)
        detected_product_line = self._lexicon.classify_product_line(corrected, concepts)

        expansions = self._build_expansions(
            language=language,
            normalized=normalized,
            corrected=corrected,
            concepts=concepts,
        )

        search_text = self._build_search_text(original, corrected, expansions)

        return PreparedQuery(
            original=original,
            normalized=corrected,
            language=language,
            search_text=search_text,
            expansions=expansions,
            corrections=corrections,
            concepts=concepts,
            scenario_tags=scenario_tags,
            detected_product_line=detected_product_line,
            requested_product_line=product_line,
        )

    # -- internals --------------------------------------------------------- #

    def _build_expansions(
        self,
        *,
        language: Language,
        normalized: str,
        corrected: str,
        concepts: tuple,
    ) -> tuple[str, ...]:
        """Terms appended to the query, in priority order.

        Priority: spelling corrections the agent did not type, then English
        equivalents of matched concepts, then a hyphen-free variant. Everything
        already present in the query is skipped, since repeating a term inflates
        its term frequency without adding information.
        """
        expansions: list[str] = []

        def add(term: str) -> None:
            term = term.strip()
            if not term or term in corrected or term in expansions:
                return
            if len(expansions) >= MAX_EXPANSION_TERMS:
                return
            expansions.append(term)

        # A corrected token is a term the index has and the agent's text does not.
        if corrected != normalized:
            for token in corrected.split():
                if token not in normalized:
                    add(token)

        for term in self._lexicon.english_equivalents(concepts):
            add(term)

        hyphen_free = strip_hyphens(corrected)
        if hyphen_free != corrected:
            add(hyphen_free)

        return tuple(expansions)

    @staticmethod
    def _build_search_text(original: str, corrected: str, expansions: tuple[str, ...]) -> str:
        parts = [original]
        # The corrected form is added only when it genuinely differs, so a clean
        # query is not sent twice.
        if corrected and normalize(original) != corrected:
            parts.append(corrected)
        parts.extend(expansions)
        return " ".join(part for part in parts if part).strip()

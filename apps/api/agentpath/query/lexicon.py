"""The customer-service lexicon: typos, abbreviations, concepts, product lines.

This is where "hotel has no room" becomes "overbooking", where "الفندق ما عنده
غرف" reaches an English procedure, and where "nonref" becomes "non-refundable".

Every entry is data in ``data/synthetic/lexicon.yaml``, not code, so extending
coverage is a review of a YAML diff rather than a code change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agentpath.domain import ConceptMatch, Language
from agentpath.paths import lexicon_path
from agentpath.query.normalize import normalize, strip_hyphens, tokenize


class LexiconError(RuntimeError):
    """Raised when the lexicon file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Concept:
    """One canonical concept with its bilingual surface forms."""

    concept_id: str
    canonical: str
    product_lines: tuple[str, ...]
    scenario_tags: tuple[str, ...]
    english_forms: tuple[str, ...]
    arabic_forms: tuple[str, ...]
    #: Normalized surface form -> language, precomputed at load time.
    normalized_forms: dict[str, Language] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Lexicon:
    """Loaded lexicon with matching helpers.

    Matching is substring-based on normalized text for multi-word forms and
    token-based for single words. Substring matching is what makes "the hotel has
    no room tonight" match the form "hotel has no room"; token matching prevents
    "ref" inside "reference" from being treated as "refund".
    """

    version: str
    typos: dict[str, str]
    abbreviations: dict[str, str]
    concepts: tuple[Concept, ...]
    product_line_hints: dict[str, tuple[str, ...]]

    # -- corrections ------------------------------------------------------- #

    def correct(self, normalized_text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
        """Fix known misspellings and expand shorthand.

        Returns the corrected text and the list of ``(before, after)`` changes so
        the UI can show the agent what was interpreted.
        """
        corrections: list[tuple[str, str]] = []
        text = normalized_text

        # Multi-word abbreviations first: "non ref" must win over "ref".
        for phrase, replacement in sorted(self.abbreviations.items(), key=lambda kv: -len(kv[0])):
            if " " not in phrase:
                continue
            if phrase in text:
                text = text.replace(phrase, replacement)
                corrections.append((phrase, replacement))

        tokens = tokenize(text)
        rebuilt: list[str] = []
        for token in tokens:
            if token in self.typos:
                replacement = self.typos[token]
                corrections.append((token, replacement))
                rebuilt.append(replacement)
            elif token in self.abbreviations:
                replacement = self.abbreviations[token]
                corrections.append((token, replacement))
                rebuilt.append(replacement)
            else:
                rebuilt.append(token)

        return " ".join(rebuilt), tuple(corrections)

    # -- concept matching -------------------------------------------------- #

    def match_concepts(self, normalized_text: str) -> tuple[ConceptMatch, ...]:
        """Every concept recognised in *normalized_text*, longest form first."""
        haystacks = {normalized_text, strip_hyphens(normalized_text)}
        token_sets = {frozenset(tokenize(text)) for text in haystacks}

        matches: list[ConceptMatch] = []
        for concept in self.concepts:
            best: tuple[str, Language] | None = None
            for surface, language in sorted(
                concept.normalized_forms.items(), key=lambda kv: -len(kv[0])
            ):
                if " " in surface or "-" in surface:
                    if any(surface in haystack for haystack in haystacks):
                        best = (surface, language)
                        break
                elif any(surface in tokens for tokens in token_sets):
                    best = (surface, language)
                    break
            if best:
                matches.append(
                    ConceptMatch(
                        concept_id=concept.concept_id,
                        canonical=concept.canonical,
                        matched_surface=best[0],
                        language=best[1],
                    )
                )
        return tuple(matches)

    def concept_by_id(self, concept_id: str) -> Concept | None:
        for concept in self.concepts:
            if concept.concept_id == concept_id:
                return concept
        return None

    # -- expansion --------------------------------------------------------- #

    def english_equivalents(self, matches: tuple[ConceptMatch, ...], limit_per_concept: int = 3) -> tuple[str, ...]:
        """English search terms for the matched concepts.

        English is the pivot language of the corpus, so an Arabic query needs
        these to reach the lexical (BM25) half of hybrid retrieval. Semantic
        search would often find the procedure anyway; exact terms like
        "non-refundable" are what lexical scoring needs to see.
        """
        terms: list[str] = []
        for match in matches:
            concept = self.concept_by_id(match.concept_id)
            if concept is None:
                continue
            candidates = [concept.canonical, *concept.english_forms]
            for candidate in candidates:
                normalized_candidate = normalize(candidate)
                if normalized_candidate and normalized_candidate not in terms:
                    terms.append(normalized_candidate)
                if len([t for t in terms if t]) and candidates.index(candidate) >= limit_per_concept:
                    break
        return tuple(terms)

    # -- classification ---------------------------------------------------- #

    def scenario_tags(self, matches: tuple[ConceptMatch, ...]) -> tuple[str, ...]:
        tags: list[str] = []
        for match in matches:
            concept = self.concept_by_id(match.concept_id)
            if concept is None:
                continue
            for tag in concept.scenario_tags:
                if tag not in tags:
                    tags.append(tag)
        return tuple(tags)

    def classify_product_line(
        self, normalized_text: str, matches: tuple[ConceptMatch, ...]
    ) -> str | None:
        """Best-guess product line, or ``None`` when the signal is ambiguous.

        Concept matches count double: "duplicate booking" is a stronger signal
        than the bare word "booking" appearing in a hint list. An ambiguous
        result returns ``None`` rather than a coin flip, because a wrong metadata
        filter removes the correct procedure from the candidate set entirely.
        """
        votes: Counter[str] = Counter()

        for match in matches:
            concept = self.concept_by_id(match.concept_id)
            if concept is None:
                continue
            for product_line in concept.product_lines:
                votes[product_line] += 2

        tokens = set(tokenize(normalized_text))
        haystack = normalized_text
        for product_line, hints in self.product_line_hints.items():
            for hint in hints:
                if " " in hint:
                    if hint in haystack:
                        votes[product_line] += 1
                elif hint in tokens:
                    votes[product_line] += 1

        if not votes:
            return None
        ranked = votes.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]


def load_lexicon(path: Path | None = None) -> Lexicon:
    """Load and precompute the lexicon.

    Surface forms are normalized once here rather than on every query, which is
    what keeps query understanding in the microsecond range.
    """
    path = path or lexicon_path()
    if not path.is_file():
        raise LexiconError(f"lexicon not found at {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LexiconError(f"lexicon is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise LexiconError("lexicon must be a YAML mapping")

    typos = {normalize(key): normalize(value) for key, value in (raw.get("typos") or {}).items()}
    abbreviations = {
        normalize(key): normalize(value) for key, value in (raw.get("abbreviations") or {}).items()
    }

    concepts: list[Concept] = []
    for entry in raw.get("concepts") or []:
        if not isinstance(entry, dict) or "id" not in entry:
            raise LexiconError(f"malformed concept entry: {entry!r}")
        english_forms = tuple(entry.get("en") or ())
        arabic_forms = tuple(entry.get("ar") or ())
        if not english_forms:
            raise LexiconError(f"concept {entry['id']} has no English surface forms")

        normalized_forms: dict[str, Language] = {}
        for form in english_forms:
            normalized_form = normalize(form)
            if normalized_form:
                normalized_forms[normalized_form] = Language.ENGLISH
        for form in arabic_forms:
            normalized_form = normalize(form)
            if normalized_form:
                normalized_forms[normalized_form] = Language.ARABIC

        concepts.append(
            Concept(
                concept_id=str(entry["id"]),
                canonical=str(entry.get("canonical") or entry["id"]),
                product_lines=tuple(entry.get("product_lines") or ()),
                scenario_tags=tuple(entry.get("scenario_tags") or ()),
                english_forms=english_forms,
                arabic_forms=arabic_forms,
                normalized_forms=normalized_forms,
            )
        )

    if not concepts:
        raise LexiconError("lexicon contains no concepts")

    product_line_hints = {
        str(name): tuple(normalize(hint) for hint in hints or ())
        for name, hints in (raw.get("product_lines") or {}).items()
    }

    return Lexicon(
        version=str(raw.get("version", "unknown")),
        typos=typos,
        abbreviations=abbreviations,
        concepts=tuple(concepts),
        product_line_hints=product_line_hints,
    )

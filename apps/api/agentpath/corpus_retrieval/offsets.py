"""Find a retrieved passage's position in the document it came from.

The problem this solves is a direct consequence of a Stage 2B1 decision. RAGFlow
parses and chunks the uploaded documents itself, so a retrieved passage arrives as
text with no offsets, while a clickable citation needs a character position. The
alternative was to upload our own pre-computed chunks so the offsets would come
back with them - rejected because RAGFlow deletes manual chunks on every re-parse
(see ``agentpath_ingestion.corpora.ingest``).

So the position is recovered instead, and recovery is cheap and exact for one
reason: **the document text was never rewritten.** What was uploaded is what the
corpus published, byte for byte, so the passage really is a substring of it. When
it is not - RAGFlow trims, and its markdown handling sometimes drops a blank line -
a whitespace-insensitive search finds it anyway.

What is deliberately *not* done here is approximate matching. No fuzzy ratio, no
longest-common-subsequence, no nearest paragraph. An offset is the thing a reader
is sent to, and sending them to a paragraph that merely resembles the citation is
worse than sending them to the top of the document and saying so: the first is
wrong and looks right. A passage that cannot be located exactly or
whitespace-insensitively is reported as :attr:`OffsetSource.UNRESOLVED`, and the
evaluation counts how often that happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentpath.corpus_retrieval.ports import OffsetSource

#: Below this, a passage is too short for a unique match to mean anything: a
#: ten-character string occurs in a 168,000-character manual by coincidence, and
#: the first occurrence is then a coin toss rather than the passage's position.
MIN_LOCATABLE_CHARS = 24

#: RAGFlow prefixes some chunks with the document's heading path. Tried as a
#: fallback by trimming leading lines, bounded so the trimming cannot eat the whole
#: passage and match on a fragment.
MAX_TRIMMED_LEADING_LINES = 3


@dataclass(frozen=True, slots=True)
class Located:
    """Where a passage sits, and how confidently that was established."""

    char_start: int | None
    char_end: int | None
    source: OffsetSource
    #: Set when the passage could not be located, for the result's ``problems``.
    problem: str | None = None

    @property
    def found(self) -> bool:
        return self.char_start is not None and self.char_end is not None


def locate_passage(document_text: str, passage_text: str) -> Located:
    """Locate *passage_text* in *document_text*.

    Four attempts, in decreasing confidence, stopping at the first that succeeds:

    1. verbatim :meth:`str.find`, which is what almost always happens;
    2. verbatim after stripping the passage's own leading and trailing whitespace,
       since RAGFlow trims;
    3. verbatim after dropping up to :data:`MAX_TRIMMED_LEADING_LINES` leading
       lines, which handles a chunk prefixed with its heading path;
    4. whitespace-insensitive, mapping the match back to original offsets.

    Ambiguity is reported, not hidden: when the passage occurs more than once the
    first occurrence is returned, because a citation must point somewhere and the
    first is the only non-arbitrary choice, but the count travels in ``problem`` so
    the evaluation can see it.
    """
    if not document_text or not passage_text:
        return Located(None, None, OffsetSource.UNRESOLVED, problem="empty document or passage")

    stripped = passage_text.strip()
    if len(stripped) < MIN_LOCATABLE_CHARS:
        return Located(
            None,
            None,
            OffsetSource.UNRESOLVED,
            problem=f"passage is {len(stripped)} characters, below the {MIN_LOCATABLE_CHARS} needed to locate uniquely",
        )

    for candidate in _candidates(passage_text, stripped):
        index = document_text.find(candidate)
        if index != -1:
            occurrences = document_text.count(candidate)
            problem = None
            if occurrences > 1:
                problem = f"passage occurs {occurrences} times; cited the first"
            return Located(index, index + len(candidate), OffsetSource.EXACT_MATCH, problem=problem)

    return _locate_normalised(document_text, stripped)


def _candidates(passage_text: str, stripped: str) -> tuple[str, ...]:
    """The verbatim forms worth trying, most faithful first, without duplicates."""
    forms = [passage_text, stripped]
    lines = stripped.split("\n")
    for drop in range(1, min(MAX_TRIMMED_LEADING_LINES, len(lines) - 1) + 1):
        tail = "\n".join(lines[drop:]).strip()
        if len(tail) >= MIN_LOCATABLE_CHARS:
            forms.append(tail)

    unique: list[str] = []
    for form in forms:
        if form and form not in unique:
            unique.append(form)
    return tuple(unique)


def _locate_normalised(document_text: str, passage_text: str) -> Located:
    """Whitespace-insensitive search, mapped back to original offsets.

    The document is projected onto a form where every run of whitespace is one
    space, keeping an index from each projected character back to its original
    position. A match in the projection therefore yields real offsets, not
    projected ones - which is the whole point, since the offset is handed to a
    reader.

    The edges are the honest weakness and the reason this reports
    :attr:`OffsetSource.NORMALISED_MATCH` rather than an exact match: the span is
    right, but where collapsed whitespace sat at a boundary the endpoints can be off
    by the characters that were collapsed.
    """
    flat_document, index_map = _flatten(document_text)
    flat_passage, _ = _flatten(passage_text)
    if not flat_passage:
        return Located(None, None, OffsetSource.UNRESOLVED, problem="passage is only whitespace")

    position = flat_document.find(flat_passage)
    if position == -1:
        return Located(
            None,
            None,
            OffsetSource.UNRESOLVED,
            problem=f"passage of {len(flat_passage)} characters is not a substring of its document",
        )

    start = index_map[position]
    # The character *after* the last matched one, so the span is half-open like
    # every other offset pair in this system.
    last = index_map[position + len(flat_passage) - 1]
    occurrences = flat_document.count(flat_passage)
    problem = f"passage occurs {occurrences} times; cited the first" if occurrences > 1 else None
    return Located(start, last + 1, OffsetSource.NORMALISED_MATCH, problem=problem)


def _flatten(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space, keeping a map back to real offsets.

    ``index_map[i]`` is the original index of ``flattened[i]``. A collapsed run maps
    to the position of its first whitespace character, so a span that begins at a
    collapsed boundary starts at the beginning of the whitespace rather than inside
    it.
    """
    flattened: list[str] = []
    index_map: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_whitespace:
                flattened.append(" ")
                index_map.append(index)
                in_whitespace = True
            continue
        in_whitespace = False
        flattened.append(char)
        index_map.append(index)

    # A leading or trailing collapsed run would make every offset comparison depend
    # on whitespace the reader cannot see, so both ends are trimmed together with
    # their map entries.
    start = 1 if flattened and flattened[0] == " " else 0
    end = len(flattened) - 1 if flattened and flattened[-1] == " " else len(flattened)
    return "".join(flattened[start:end]), index_map[start:end]


__all__ = [
    "MAX_TRIMMED_LEADING_LINES",
    "MIN_LOCATABLE_CHARS",
    "Located",
    "locate_passage",
]

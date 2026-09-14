"""The lexical baseline: what plain keyword search would do.

**Benchmark reference only. Never a production retrieval path.** It exists to
answer one question honestly: how much does the AgentPath pipeline actually add
over the kind of search a knowledge base ships with by default?

Deliberately naive, and each omission is the point:

* token overlap on normalized text, no embeddings;
* no lexicon, so no synonym expansion, no typo correction, no abbreviation
  expansion - "overboking" simply does not match "overbooking";
* fields weighted by how a keyword index would treat them (title over body);
* abstains when nothing overlaps.

It reads the procedure YAML directly rather than going through RAGFlow, because a
baseline that depended on the embedding stack would not be a baseline.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from agentpath.query.normalize import normalize, tokenize
from agentpath_ingestion.dataset import load_procedures

#: Field weights. A term in the title is stronger evidence than the same term
#: buried in a step, which is how a conventional keyword index behaves.
TITLE_WEIGHT = 3.0
SYNONYM_WEIGHT = 2.0
BODY_WEIGHT = 1.0

#: Words carrying no discriminative signal in this corpus.
STOPWORDS = frozenset(
    """
    a an the and or of to for in on at is are was were be been being with without
    that this these those it its as by from into out up down over under again
    then than so such no not only own same too very can will just should now do
    does did done how what when where which who whom why my our your their his her
    """.split()
)


@dataclass(frozen=True, slots=True)
class BaselineDocument:
    """One procedure, reduced to weighted term counts."""

    procedure_id: str
    title: str
    product_line: str
    term_weights: dict[str, float]

    def score(self, query_tokens: list[str]) -> float:
        """Weighted overlap between query tokens and this document."""
        return sum(self.term_weights.get(token, 0.0) for token in query_tokens)


def _weighted_terms(text: str, weight: float, sink: Counter[str]) -> None:
    for token in tokenize(normalize(text)):
        if token in STOPWORDS or len(token) < 2:
            continue
        sink[token] += weight


def build_index() -> list[BaselineDocument]:
    """Build the baseline index from the procedure source files."""
    documents: list[BaselineDocument] = []

    for loaded in load_procedures():
        procedure = loaded.procedure
        counter: Counter[str] = Counter()

        _weighted_terms(procedure.title, TITLE_WEIGHT, counter)
        _weighted_terms(procedure.title_ar, TITLE_WEIGHT, counter)
        for synonym in procedure.all_synonyms:
            _weighted_terms(synonym, SYNONYM_WEIGHT, counter)
        _weighted_terms(procedure.procedure_id, TITLE_WEIGHT, counter)

        body_parts = [
            procedure.summary.en,
            procedure.summary.ar,
            *procedure.applies_when,
            *procedure.does_not_apply_when,
            *(step.instruction for step in procedure.steps),
            *(step.instruction_ar or "" for step in procedure.steps),
            *(step.screen or "" for step in procedure.steps),
            *(step.ui_element or "" for step in procedure.steps),
            *procedure.exceptions,
            *procedure.do,
            *procedure.do_not,
            procedure.escalation.trigger,
            procedure.escalation.queue,
        ]
        for part in body_parts:
            if part:
                _weighted_terms(part, BODY_WEIGHT, counter)

        documents.append(
            BaselineDocument(
                procedure_id=procedure.procedure_id,
                title=procedure.title,
                product_line=procedure.product_line.value,
                term_weights=dict(counter),
            )
        )
    return documents


class LexicalBaseline:
    """Keyword-overlap search over the procedure corpus."""

    def __init__(self, documents: list[BaselineDocument] | None = None) -> None:
        self._documents = documents if documents is not None else build_index()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def search(
        self, query: str, *, limit: int = 3, product_line: str | None = None
    ) -> tuple[list[tuple[str, float]], bool]:
        """Return ``(ranked (procedure_id, score) pairs, abstained)``.

        Scores are normalized by query length so they stay roughly comparable
        across queries; the ordering is unaffected.
        """
        tokens = [token for token in tokenize(normalize(query)) if token not in STOPWORDS and len(token) >= 2]
        if not tokens:
            return [], True

        candidates = self._documents
        if product_line:
            candidates = [doc for doc in candidates if doc.product_line == product_line]

        divisor = math.sqrt(len(tokens)) or 1.0
        scored = [(doc.procedure_id, doc.score(tokens) / divisor) for doc in candidates]
        scored = [(procedure_id, score) for procedure_id, score in scored if score > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))

        # Nothing overlapped: a keyword index has no answer, and saying so is the
        # honest behaviour to compare against.
        return scored[:limit], not scored

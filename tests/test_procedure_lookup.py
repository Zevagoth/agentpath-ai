"""A procedure id typed by the agent is an exact request, not a search term.

Why this exists. Adding the English control labels to the Arabic instructions
changed the indexed text, and one benchmark query regressed: searching for the
literal id ``NW-HTL-014`` returned ``NW-HTL-021`` first, at 0.7642 against
0.7624. The cause is structural rather than incidental - the corpus
cross-references procedures by id, so ``NW-HTL-021``'s "does not apply when"
section contains the string ``NW-HTL-014``, and both semantic and lexical scoring
treat that as evidence about the query. A 0.0018 margin between two procedures is
noise, and no amount of re-weighting makes it reliable.

So an id is filtered, not scored. The tests below fix the three properties that
makes worth having:

* the id is recognised wherever it appears in the query, in either case;
* it becomes a metadata restriction, so the wrong procedure cannot win;
* recognising it never widens anything - a query with no id, or with two, behaves
  exactly as before.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentpath.query.lexicon import load_lexicon
from agentpath.query.pipeline import QueryPreprocessor
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever, _metadata_condition

BASE_URL = "http://ragflow.test:9380"
DATASET = "dataset-1"


@pytest.fixture(scope="module")
def preprocessor() -> QueryPreprocessor:
    return QueryPreprocessor(load_lexicon())


# --------------------------------------------------------------------------- #
# Recognising the id
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("NW-HTL-014", "NW-HTL-014"),
        # Case: an agent pasting a reference does not reliably shift-key it.
        ("nw-htl-014", "NW-HTL-014"),
        ("Nw-Htl-014", "NW-HTL-014"),
        # Embedded in a real question, in both languages. Narrowing to the named
        # procedure is what the agent asked for either way.
        ("NW-HTL-014 what about the cost difference?", "NW-HTL-014"),
        ("what does NW-CXL-032 say about waivers", "NW-CXL-032"),
        ("شو يقول NW-HTL-014 عن نقل الضيف؟", "NW-HTL-014"),
        ("procedure nw-ref-018 step 5", "NW-REF-018"),
    ],
)
def test_an_id_in_the_query_is_recognised(preprocessor: QueryPreprocessor, query: str, expected: str) -> None:
    prepared = preprocessor.prepare(query)
    assert prepared.requested_procedure_id == expected
    assert prepared.is_procedure_lookup


@pytest.mark.parametrize(
    "query",
    [
        "hotel has no room for the guest",
        "الفندق ما عنده غرف والضيف واصل",
        "overboking at the hotle",
        # Nearly an id, but not one. A filter built from these would return
        # nothing and turn a normal search into an abstention.
        "NW-HTL-14",
        "NW-HTL-0141",
        "NWHTL014",
        "NW-HT-014",
        "HTL-014",
        "014",
    ],
)
def test_a_query_without_an_id_is_untouched(preprocessor: QueryPreprocessor, query: str) -> None:
    prepared = preprocessor.prepare(query)
    assert prepared.requested_procedure_id is None
    assert not prepared.is_procedure_lookup


def test_two_ids_are_a_comparison_not_a_lookup(preprocessor: QueryPreprocessor) -> None:
    """Narrowing to one of two named procedures answers a question nobody asked.

    "What is the difference between NW-HTL-014 and NW-HTL-021" is about both. A
    filter would silently delete half the answer, so no filter is applied and the
    query runs as an ordinary search over both.
    """
    prepared = preprocessor.prepare("what is the difference between NW-HTL-014 and NW-HTL-021?")
    assert prepared.requested_procedure_id is None
    assert not prepared.is_procedure_lookup


def test_recognising_an_id_does_not_disturb_the_rest_of_understanding(
    preprocessor: QueryPreprocessor,
) -> None:
    """The id is extra information, so nothing else about the query may change."""
    question = "hotel has no room for the guest"
    plain = preprocessor.prepare(question)
    with_id = preprocessor.prepare(f"NW-HTL-014 {question}")

    assert with_id.requested_procedure_id == "NW-HTL-014"
    # Language, scenario tags and product-line detection are unaffected.
    assert with_id.language == plain.language
    assert with_id.scenario_tags == plain.scenario_tags
    assert with_id.detected_product_line == plain.detected_product_line
    # The agent's words still lead the search text.
    assert with_id.search_text.startswith("NW-HTL-014 hotel has no room")


# --------------------------------------------------------------------------- #
# The filter it becomes
# --------------------------------------------------------------------------- #


def test_the_filter_is_an_exact_metadata_comparison() -> None:
    condition = _metadata_condition(None, "NW-HTL-014")
    assert condition == {
        "logic": "and",
        "conditions": [{"name": "procedure_id", "comparison_operator": "=", "value": "NW-HTL-014"}],
    }


def test_a_product_line_and_an_id_are_combined_with_and() -> None:
    """An agent who picks a product line and names a procedure has stated both."""
    condition = _metadata_condition("hotels", "NW-HTL-014")
    assert condition is not None
    assert condition["logic"] == "and"
    assert [entry["name"] for entry in condition["conditions"]] == ["product_line", "procedure_id"]


def test_no_restriction_means_no_filter_at_all() -> None:
    """An empty condition object is not the same as none, and RAGFlow treats it so."""
    assert _metadata_condition(None, None) is None
    assert _metadata_condition("", "") is None


@respx.mock
async def test_the_retriever_sends_the_id_as_a_metadata_condition(
    preprocessor: QueryPreprocessor,
) -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": [], "total": 0}})

    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": [{"id": DATASET, "name": "ds"}]})
    )
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)

    prepared = preprocessor.prepare("NW-HTL-014")
    async with RAGFlowClient(BASE_URL, "key") as client:
        retriever = RAGFlowRetriever(client, dataset_name="ds")
        await retriever.retrieve(prepared, limit=3, procedure_id=prepared.requested_procedure_id)

    assert captured["metadata_condition"] == {
        "logic": "and",
        "conditions": [{"name": "procedure_id", "comparison_operator": "=", "value": "NW-HTL-014"}],
    }


@respx.mock
async def test_an_ordinary_query_still_sends_no_condition(preprocessor: QueryPreprocessor) -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"code": 0, "data": {"chunks": [], "total": 0}})

    respx.get(f"{BASE_URL}/api/v1/datasets").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": [{"id": DATASET, "name": "ds"}]})
    )
    respx.post(f"{BASE_URL}/api/v1/retrieval").mock(side_effect=_handler)

    prepared = preprocessor.prepare("hotel has no room for the guest")
    async with RAGFlowClient(BASE_URL, "key") as client:
        retriever = RAGFlowRetriever(client, dataset_name="ds")
        await retriever.retrieve(prepared, limit=3, procedure_id=prepared.requested_procedure_id)

    assert "metadata_condition" not in captured


# --------------------------------------------------------------------------- #
# The search service passes it through
# --------------------------------------------------------------------------- #


async def test_the_search_service_forwards_the_id_it_detected() -> None:
    """The service must not require the caller to have noticed the id."""
    from dataclasses import dataclass, field

    from agentpath.config import Settings
    from agentpath.domain import DependencyHealth, PreparedQuery, RetrievalMode, RetrievedPassages
    from agentpath.search_service import SearchService

    @dataclass
    class Recorder:
        calls: list[dict] = field(default_factory=list)

        @property
        def mode(self) -> RetrievalMode:
            return RetrievalMode.HYBRID

        async def retrieve(
            self,
            query: PreparedQuery,
            *,
            limit: int,
            product_line: str | None = None,
            procedure_id: str | None = None,
        ) -> RetrievedPassages:
            self.calls.append({"product_line": product_line, "procedure_id": procedure_id})
            return RetrievedPassages(passages=(), mode=self.mode, engine_ms=1.0)

        async def health(self) -> DependencyHealth:
            return DependencyHealth(name="ragflow", reachable=True, detail="fake")

    recorder = Recorder()
    lexicon = load_lexicon()
    settings = Settings(RAGFLOW_API_KEY="test-key", AGENTPATH_LOG_LEVEL="WARNING")
    service = SearchService(retriever=recorder, lexicon=lexicon, settings=settings)  # type: ignore[arg-type]

    await service.search("NW-HTL-014")
    await service.search("hotel has no room", product_line="hotels")

    assert recorder.calls[0] == {"product_line": None, "procedure_id": "NW-HTL-014"}
    assert recorder.calls[1] == {"product_line": "hotels", "procedure_id": None}

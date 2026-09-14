"""Tests for the Ollama generation client.

Unit tests only: every response is a recorded payload served by ``respx``, so
nothing here loads a model, touches the GPU or needs the daemon. The payload
shapes and the three quirks they encode were verified against the live Ollama
**0.34.0** host during T7, not taken from memory:

* ``think`` is a top-level field on ``/api/chat``. Passing it inside ``options``
  is silently ignored, and with thinking left on the model spent its whole token
  budget reasoning and returned *empty* content.
* ``format`` accepts a full JSON Schema, ``$defs``/``$ref`` included, and Ollama
  compiles it into a decoding grammar. ``pattern`` and ``enum`` are enforced too.
* ``options`` is merged over the model's own parameters, and an absent key leaves
  the model's value standing - which for ``qwen3.5:4b``'s ``presence_penalty`` is
  1.5, not 0.

Anything that needs the real model is in the T7 harness
(``scripts/measure_model_fitness.py``), not here.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from agentpath.generation import (
    ChatCompletion,
    OllamaClient,
    OllamaError,
    OllamaRuntimeOptions,
    OllamaUnavailable,
    StructuredOutputError,
)
from agentpath.generation.ollama_client import OllamaModelMissing

BASE_URL = "http://127.0.0.1:11434"
CHAT_URL = f"{BASE_URL}/api/chat"
TAGS_URL = f"{BASE_URL}/api/tags"
MODEL = "qwen3.5:4b"

MESSAGES = [{"role": "user", "content": "what do I do?"}]


class Step(BaseModel):
    instruction: str
    citation: str = Field(pattern=r"^NW-[A-Z]{3}-[0-9]{3}#S[0-9]{2}$")


class Answer(BaseModel):
    """Nested on purpose: this is what makes Pydantic emit ``$defs``/``$ref``."""

    summary: str
    steps: list[Step]


def _completion(content: str, *, thinking: str = "", done_reason: str = "stop") -> dict:
    """A ``/api/chat`` response, shaped as Ollama 0.34.0 returns it."""
    return {
        "model": MODEL,
        "created_at": "2026-03-01T00:00:00Z",
        "message": {"role": "assistant", "content": content, "thinking": thinking},
        "done": True,
        "done_reason": done_reason,
        # Nanoseconds, as Ollama reports every duration.
        "total_duration": 5_200_000_000,
        "load_duration": 60_000_000,
        "prompt_eval_count": 1693,
        "prompt_eval_duration": 900_000_000,
        "eval_count": 271,
        "eval_duration": 4_100_000_000,
    }


def _tags(*names: str) -> dict:
    return {"models": [{"name": name, "model": name, "size": 3_389_983_735} for name in names]}


def client(**kwargs) -> OllamaClient:
    return OllamaClient(BASE_URL, model=MODEL, **kwargs)


# -- request shape ---------------------------------------------------------- #


@respx.mock
async def test_think_is_top_level_and_not_an_option() -> None:
    """The quirk that cost the most to find: inside ``options`` it does nothing.

    A regression here does not raise, it silently returns an empty answer, so the
    position of the field is asserted rather than trusted.
    """
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("ready"))

    respx.post(CHAT_URL).mock(side_effect=_handler)

    async with client() as ollama:
        await ollama.chat(messages=MESSAGES)

    assert captured["think"] is False
    assert "think" not in captured["options"]


@respx.mock
async def test_streaming_is_always_off() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("ready"))

    respx.post(CHAT_URL).mock(side_effect=_handler)

    async with client() as ollama:
        await ollama.chat(messages=MESSAGES)

    assert captured["stream"] is False


@respx.mock
async def test_every_option_is_sent_including_the_zeros() -> None:
    """Zeros are load-bearing: an absent key inherits the model's own value.

    ``qwen3.5:4b`` ships ``presence_penalty 1.5``. Omitting it because it is
    falsy would let the model penalise the repeated step tokens a grounded answer
    is made of.
    """
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("ready"))

    respx.post(CHAT_URL).mock(side_effect=_handler)

    async with client() as ollama:
        await ollama.chat(messages=MESSAGES)

    options = captured["options"]
    assert options["num_ctx"] == 8192
    assert options["temperature"] == 0.0
    assert options["seed"] == 20260301
    assert options["presence_penalty"] == 0.0
    assert options["frequency_penalty"] == 0.0
    assert options["repeat_penalty"] == 1.0


@respx.mock
async def test_the_full_context_window_is_never_requested_by_default() -> None:
    """8192, not the 262144 the model advertises. The KV cache is why."""
    async with client() as ollama:
        assert ollama.options.num_ctx == 8192


@respx.mock
async def test_per_call_options_override_the_client_defaults() -> None:
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("ready"))

    respx.post(CHAT_URL).mock(side_effect=_handler)

    async with client() as ollama:
        await ollama.chat(messages=MESSAGES, options=OllamaRuntimeOptions(num_ctx=2048, seed=7))

    assert captured["options"]["num_ctx"] == 2048
    assert captured["options"]["seed"] == 7


# -- response parsing ------------------------------------------------------- #


@respx.mock
async def test_durations_are_converted_from_nanoseconds() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("ready")))

    async with client() as ollama:
        completion = await ollama.chat(messages=MESSAGES)

    assert completion.total_ms == pytest.approx(5200.0)
    assert completion.prompt_ms == pytest.approx(900.0)
    assert completion.generation_ms == pytest.approx(4100.0)
    assert completion.prompt_tokens == 1693
    assert completion.generation_tokens == 271


@respx.mock
async def test_thinking_text_is_surfaced_rather_than_dropped() -> None:
    """A regression in the thinking-off setting must be visible as data."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("", thinking="Let me consider...")))

    async with client() as ollama:
        completion = await ollama.chat(messages=MESSAGES)

    assert completion.thinking == "Let me consider..."


@respx.mock
async def test_a_length_stop_is_reported_as_truncated() -> None:
    """So a caller can say "the cap was too low", not "it cannot follow a schema"."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion('{"a":', done_reason="length")))

    async with client() as ollama:
        completion = await ollama.chat(messages=MESSAGES)

    assert completion.truncated is True


@respx.mock
async def test_a_response_without_a_message_is_an_error() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"model": MODEL, "done": True}))

    async with client() as ollama:
        with pytest.raises(OllamaError, match="no message"):
            await ollama.chat(messages=MESSAGES)


@respx.mock
async def test_a_non_json_body_is_an_error_not_a_crash() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, text="<html>gateway</html>"))

    async with client() as ollama:
        with pytest.raises(OllamaError, match="non-JSON"):
            await ollama.chat(messages=MESSAGES)


# -- structured output ------------------------------------------------------ #


@respx.mock
async def test_the_pydantic_schema_is_sent_as_the_format_field() -> None:
    """Nested models emit ``$defs``/``$ref``; Ollama 0.34.0 resolves them.

    Verified against the live host before the harness relied on it, because a
    schema the grammar compiler cannot read is rejected outright.
    """
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completion('{"summary":"s","steps":[{"instruction":"i","citation":"NW-HTL-014#S01"}]}'),
        )

    respx.post(CHAT_URL).mock(side_effect=_handler)

    async with client() as ollama:
        result = await ollama.chat_structured(messages=MESSAGES, schema=Answer)

    assert captured["format"]["$defs"]["Step"]["properties"]["citation"]["pattern"]
    assert captured["format"]["properties"]["steps"]["items"]["$ref"] == "#/$defs/Step"
    assert result.value.steps[0].citation == "NW-HTL-014#S01"


@respx.mock
async def test_output_that_is_not_json_carries_the_raw_text() -> None:
    """A compliance measurement is worthless without the thing that failed."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("I cannot help with that.")))

    async with client() as ollama:
        with pytest.raises(StructuredOutputError) as caught:
            await ollama.chat_structured(messages=MESSAGES, schema=Answer)

    assert caught.value.raw == "I cannot help with that."


@respx.mock
async def test_truncated_json_says_the_cap_was_hit() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=_completion('{"summary":"s","steps":[', done_reason="length"))
    )

    async with client() as ollama:
        with pytest.raises(StructuredOutputError, match="num_predict"):
            await ollama.chat_structured(messages=MESSAGES, schema=Answer)


@respx.mock
async def test_valid_json_that_violates_the_schema_still_fails() -> None:
    """The grammar cannot express ``pattern``-level rules on its own.

    Which is exactly why validation is a second layer rather than a formality:
    this body is legal JSON of the right shape and the citation is not a step
    token.
    """
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_completion('{"summary":"s","steps":[{"instruction":"i","citation":"see the Do list"}]}'),
        )
    )

    async with client() as ollama:
        with pytest.raises(StructuredOutputError, match="validation"):
            await ollama.chat_structured(messages=MESSAGES, schema=Answer)


@respx.mock
async def test_an_empty_body_is_a_structured_failure_not_a_parse_crash() -> None:
    """The shape a thinking model returns when it spends the budget reasoning."""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=_completion("   ")))

    async with client() as ollama:
        with pytest.raises(StructuredOutputError, match="empty"):
            await ollama.chat_structured(messages=MESSAGES, schema=Answer)


# -- error and timeout handling --------------------------------------------- #


@respx.mock
async def test_a_timeout_is_unavailable_not_a_rejection() -> None:
    """The split the answer path depends on: down is not the same as refused."""
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("too slow"))

    async with client(timeout=1.0) as ollama:
        with pytest.raises(OllamaUnavailable, match=r"timed out after 1\.0s"):
            await ollama.chat(messages=MESSAGES)


@respx.mock
async def test_a_refused_connection_is_unavailable() -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

    async with client() as ollama:
        with pytest.raises(OllamaUnavailable, match="unreachable"):
            await ollama.chat(messages=MESSAGES)


@respx.mock
async def test_a_missing_model_is_its_own_error_with_the_fix_in_it() -> None:
    """404 plus "not found" is an operator problem, not a wrong URL."""
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(404, json={"error": f'model "{MODEL}" not found, try pulling it first'})
    )

    async with client() as ollama:
        with pytest.raises(OllamaModelMissing, match="not pulled"):
            await ollama.chat(messages=MESSAGES)


@respx.mock
async def test_a_server_error_is_unavailable_not_a_rejection() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, json={"error": "runner exited"}))

    async with client() as ollama:
        with pytest.raises(OllamaUnavailable, match="HTTP 500"):
            await ollama.chat(messages=MESSAGES)


@respx.mock
async def test_a_bad_request_is_a_rejection_not_an_outage() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(400, json={"error": "invalid format schema"}))

    async with client() as ollama:
        with pytest.raises(OllamaError, match="rejected the request") as caught:
            await ollama.chat(messages=MESSAGES)

    assert not isinstance(caught.value, OllamaUnavailable)
    assert caught.value.status == 400


@respx.mock
async def test_an_error_body_that_is_not_json_still_produces_a_message() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(400, text="Bad Request"))

    async with client() as ollama:
        with pytest.raises(OllamaError, match="Bad Request"):
            await ollama.chat(messages=MESSAGES)


# -- health and introspection ----------------------------------------------- #


@respx.mock
async def test_health_reports_up_but_unpulled_as_unhealthy() -> None:
    """Every connection check passes while every request fails. One flag, not two."""
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags("bge-m3:latest")))

    async with client() as ollama:
        reachable, detail = await ollama.health()

    assert reachable is False
    assert "ollama pull qwen3.5:4b" in detail


@respx.mock
async def test_health_is_happy_when_the_model_is_present() -> None:
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags(MODEL, "bge-m3:latest")))

    async with client() as ollama:
        reachable, detail = await ollama.health()

    assert reachable is True
    assert MODEL in detail


@respx.mock
async def test_health_returns_a_flag_rather_than_raising_when_down() -> None:
    """It is called to report status, so an outage is the answer, not an error."""
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    async with client() as ollama:
        reachable, detail = await ollama.health()

    assert reachable is False
    assert "unreachable" in detail


@respx.mock
async def test_a_tagless_model_name_matches_the_latest_tag() -> None:
    """So no caller has to know that Ollama appends ``:latest``."""
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json=_tags("bge-m3:latest")))

    async with OllamaClient(BASE_URL, model="bge-m3") as ollama:
        assert await ollama.has_model() is True


# -- lifecycle -------------------------------------------------------------- #


async def test_an_injected_client_is_not_closed_by_us() -> None:
    """The caller owns what the caller passed in."""
    async with httpx.AsyncClient() as shared:
        ollama = OllamaClient(BASE_URL, model=MODEL, client=shared)
        await ollama.aclose()
        assert not shared.is_closed


def test_the_base_url_loses_its_trailing_slash() -> None:
    """Otherwise every path becomes a double slash."""
    assert OllamaClient(f"{BASE_URL}/", model=MODEL)._base_url == BASE_URL


def test_a_completion_without_ollama_timings_reports_the_wall_clock() -> None:
    """An older build or a fake must not report zero as if it were fast."""
    completion = ChatCompletion(
        content="x",
        model=MODEL,
        done_reason="stop",
        total_ms=12.0,
        load_ms=0.0,
        prompt_ms=0.0,
        generation_ms=0.0,
        prompt_tokens=0,
        generation_tokens=0,
    )
    assert completion.total_ms == 12.0
    assert completion.truncated is False

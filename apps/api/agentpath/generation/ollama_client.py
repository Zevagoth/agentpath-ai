"""Async HTTP client for the Ollama API.

One place owns the Ollama wire protocol: the request shape, the runtime options,
schema-constrained decoding, timeouts and error translation. Verified against
Ollama **0.34.0** running natively on the host - not in Docker, so the
CUDA-in-Docker failure that kept the reranker on CPU does not apply to this path.

Three things about this API are worth knowing before reading the code, because
each one is a field in a place you would not guess:

* ``think`` is a **top-level** field on ``/api/chat``, not an entry in
  ``options``. Passing it inside ``options`` is silently ignored, and a thinking
  model then spends its token budget reasoning out loud.
* ``format`` is also top-level, and it takes either the string ``"json"`` (free
  JSON, shape unconstrained) or a **JSON Schema object** (decoding is constrained
  to the schema). Only the second one is worth having; see
  :meth:`OllamaClient.chat_structured`.
* everything else - ``num_ctx``, ``temperature``, ``seed`` - goes inside
  ``options``, and an unknown key there is dropped without complaint. That is why
  :class:`OllamaRuntimeOptions` is a closed dataclass rather than a free dict: a
  typo in a sampler name would otherwise cost nothing at request time and show up
  only as unexplained variance in the output.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)

#: Ollama reports every duration in nanoseconds.
NANOSECONDS_PER_MS = 1_000_000


class OllamaError(RuntimeError):
    """Ollama answered, but rejected the request or produced nothing usable."""

    def __init__(self, message: str, *, status: int | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.path = path


class OllamaUnavailable(OllamaError):
    """Ollama could not be reached at all, or did not answer in time.

    Split from :class:`OllamaError` for the same reason RAGFlow's client splits
    them: "the model server is down" and "the model refused this request" need
    different handling and must never be shown to an agent as the same thing.
    """


class OllamaModelMissing(OllamaError):
    """The requested model is not pulled on this host.

    Its own class because it is the one failure with an obvious operator fix
    (``ollama pull <model>``), and because it arrives as an HTTP 404 that would
    otherwise be indistinguishable from a wrong URL path.
    """


class StructuredOutputError(OllamaError):
    """The model returned text that does not satisfy the requested schema.

    Carries ``raw`` so a caller measuring compliance can record *what* came back
    instead of only that something failed. Truncated when logged, never when
    returned, because the harness needs the whole thing and a log line does not.
    """

    def __init__(self, message: str, *, raw: str, path: str | None = None) -> None:
        super().__init__(message, path=path)
        self.raw = raw


@dataclass(frozen=True, slots=True)
class OllamaRuntimeOptions:
    """Sampler and context settings for one request.

    Defaults are the Stage 2A grounded-answer settings, chosen for reproducibility
    over variety:

    ``num_ctx=8192``
        Deliberately far below the 262144 the model advertises. Context is
        allocated as KV cache the moment the model loads, so asking for a window
        we do not need would spend VRAM we do not have on an 8 GB card. 8192 fits
        one procedure's evidence with room to spare.
    ``temperature=0.0``
        With ``top_k=1`` this makes decoding greedy. ``seed`` is then almost
        redundant, and it is set anyway because greedy decoding is only
        deterministic while every other sampler stays out of the way, and a future
        edit to this dataclass should not silently reintroduce sampling noise.
    ``num_predict=1536``
        A cap, not a target. Without one, a schema-constrained generation that
        goes wrong can run to the context limit; with it the request fails fast
        and visibly (``done_reason == "length"``).
    ``repeat_penalty=1.0``, ``presence_penalty=0.0``, ``frequency_penalty=0.0``
        All three off, and the middle one is the reason this dataclass carries
        fields the defaults would otherwise make redundant. ``ollama show
        qwen3.5:4b`` reports that the model *ships* ``presence_penalty 1.5``,
        which is a strong discouragement of any token that has already appeared.
        A grounded answer repeats procedure ids, step tokens and control labels
        constantly, and that is exactly how ``NW-HTL-014#S04`` drifts into
        ``NW-HTL-014#S4`` - a citation that no longer matches the evidence. So the
        model's own value is overridden rather than inherited.

        Honest caveat: an A/B at 0.0 against 2.0 on a repetition-heavy prompt
        produced byte-identical output, so this probe did *not* prove the override
        reaches the sampler. Under greedy decoding (``temperature=0``,
        ``top_k=1``) a logit penalty has to be enormous to change the argmax, so
        the probe was blind by construction rather than informative. The field is
        real - ``PresencePenalty`` is in Ollama's ``api.Options`` struct - and
        setting it costs nothing, so it is set.
    """

    num_ctx: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 20260301
    num_predict: int = 1536
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def as_payload(self) -> dict[str, Any]:
        """Render the ``options`` object Ollama expects.

        Every field is emitted, including the zeros. Ollama merges this map over
        the model's own parameters, so a key that is absent leaves the model's
        value standing - which for ``presence_penalty`` is 1.5, not 0.
        """
        return {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """One non-streaming chat response, with the numbers needed to report it.

    ``prompt_ms`` is Ollama's own prompt-evaluation time and ``generation_ms`` its
    decode time; they are kept apart from ``total_ms`` because a slow answer is
    either a long prompt or a long output and the fix differs. ``load_ms`` is
    non-zero only when the model had to be brought into VRAM, which is precisely
    the measurement a warm-up run exists to exclude.

    ``thinking`` should be empty. It is surfaced rather than dropped so a
    regression in the thinking-off setting is visible as data instead of as an
    unexplained latency increase.
    """

    content: str
    model: str
    done_reason: str
    total_ms: float
    load_ms: float
    prompt_ms: float
    generation_ms: float
    prompt_tokens: int
    generation_tokens: int
    thinking: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """``True`` when generation stopped because it hit ``num_predict``.

        A truncated structured answer is usually invalid JSON, so callers get to
        report "the cap was too low" rather than "the model cannot follow a
        schema" - two different verdicts.
        """
        return self.done_reason == "length"


@dataclass(frozen=True, slots=True)
class StructuredCompletion[ModelT: BaseModel]:
    """A chat response already validated against a Pydantic model."""

    value: ModelT
    completion: ChatCompletion


class OllamaClient:
    """Minimal, typed client for the Ollama endpoints AgentPath AI needs."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float = 120.0,
        options: OllamaRuntimeOptions | None = None,
        think: bool = False,
        keep_alive: str = "5m",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client.

        ``timeout`` defaults far higher than the RAGFlow client's 30 s because the
        units are different: retrieval is a database query, generation is hundreds
        of forward passes, and the first request after a cold start also pays for
        loading ~3 GB of weights into VRAM.

        ``keep_alive`` holds the model in VRAM between requests. Left at Ollama's
        own default rather than raised: on an 8 GB card the desktop compositor is
        a co-tenant, and pinning weights indefinitely to save a few seconds of
        reload is the wrong trade on this machine.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._options = options or OllamaRuntimeOptions()
        self._think = think
        self._keep_alive = keep_alive
        self._client = client
        self._owns_client = client is None

    @property
    def model(self) -> str:
        return self._model

    @property
    def options(self) -> OllamaRuntimeOptions:
        return self._options

    # -- lifecycle --------------------------------------------------------- #

    async def __aenter__(self) -> Self:
        self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- transport --------------------------------------------------------- #

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        url = f"{self._base_url}{path}"
        try:
            response = await client.post(url, json=dict(payload))
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(f"Ollama timed out after {self._timeout}s", path=path) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama is unreachable: {exc}", path=path) from exc

        # Ollama reports errors as a JSON body with a real HTTP status, so the
        # status is checked before the body is trusted to be a completion.
        if response.status_code != httpx.codes.OK:
            detail = _error_detail(response)
            if response.status_code == httpx.codes.NOT_FOUND and "not found" in detail.lower():
                raise OllamaModelMissing(
                    f"model {self._model!r} is not pulled on this host: {detail}",
                    status=response.status_code,
                    path=path,
                )
            if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
                raise OllamaUnavailable(
                    f"Ollama returned HTTP {response.status_code}: {detail}",
                    status=response.status_code,
                    path=path,
                )
            raise OllamaError(
                f"Ollama rejected the request (HTTP {response.status_code}): {detail}",
                status=response.status_code,
                path=path,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned a non-JSON body (HTTP {response.status_code})", path=path) from exc
        if not isinstance(body, dict):
            raise OllamaError(f"unexpected Ollama response shape: {type(body).__name__}", path=path)
        return body

    # -- generation -------------------------------------------------------- #

    async def chat(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_schema: Mapping[str, Any] | None = None,
        options: OllamaRuntimeOptions | None = None,
    ) -> ChatCompletion:
        """One non-streaming chat turn.

        ``stream`` is pinned to ``False``: a single response object is what the
        answer path wants, and it is also the only shape from which the timing and
        token counts can be read without reassembling a stream. Streaming to the
        UI is a later decision (see T10) and belongs at a different layer.

        ``response_schema`` is passed through to Ollama's top-level ``format``.
        Prefer :meth:`chat_structured`, which derives the schema from a Pydantic
        model and validates the result.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            # Top-level, not an option. See the module docstring.
            "think": self._think,
            "keep_alive": self._keep_alive,
            "options": (options or self._options).as_payload(),
        }
        if response_schema is not None:
            payload["format"] = dict(response_schema)

        started = time.perf_counter()
        body = await self._post("/api/chat", payload)
        wall_ms = (time.perf_counter() - started) * 1000

        message = body.get("message")
        if not isinstance(message, dict):
            raise OllamaError(f"Ollama response carried no message: {body!r}", path="/api/chat")

        completion = ChatCompletion(
            content=str(message.get("content") or ""),
            model=str(body.get("model") or self._model),
            done_reason=str(body.get("done_reason") or ""),
            # Ollama's own total excludes our HTTP overhead. The wall clock is
            # used when it is absent so a fake or an older build still reports
            # something honest rather than zero.
            total_ms=_ns_to_ms(body.get("total_duration")) or wall_ms,
            load_ms=_ns_to_ms(body.get("load_duration")),
            prompt_ms=_ns_to_ms(body.get("prompt_eval_duration")),
            generation_ms=_ns_to_ms(body.get("eval_duration")),
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            generation_tokens=int(body.get("eval_count") or 0),
            thinking=str(message.get("thinking") or ""),
            raw=body,
        )

        logger.debug(
            "ollama.chat",
            model=completion.model,
            done_reason=completion.done_reason,
            prompt_tokens=completion.prompt_tokens,
            generation_tokens=completion.generation_tokens,
            total_ms=round(completion.total_ms, 1),
            thinking_chars=len(completion.thinking),
        )
        return completion

    async def chat_structured[ModelT: BaseModel](
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: type[ModelT],
        options: OllamaRuntimeOptions | None = None,
    ) -> StructuredCompletion[ModelT]:
        """Chat, constrained to *schema*, and validated against it.

        Two layers, and both are needed. Ollama compiles the JSON Schema into a
        decoding grammar, so the token stream can only form JSON of that shape -
        that removes the "model wrote prose instead of JSON" failure entirely.
        What it does not remove is everything the grammar cannot express:
        ``minLength``, ``enum`` membership on a free-text field, cross-field
        rules, and the plain fact that a grammar can be satisfied by an empty
        object. So the result is parsed and validated by Pydantic as well, and a
        failure at that point is :class:`StructuredOutputError` carrying the raw
        text, because a compliance measurement is worthless without the thing that
        failed.

        The schema is taken from ``schema.model_json_schema()``, so the Pydantic
        model is the single definition of the contract and the two layers cannot
        drift apart.
        """
        completion = await self.chat(
            messages=messages,
            response_schema=schema.model_json_schema(),
            options=options,
        )
        raw = completion.content.strip()
        if not raw:
            raise StructuredOutputError(
                "Ollama returned an empty body for a schema-constrained request",
                raw=completion.content,
                path="/api/chat",
            )
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            hint = " (generation hit num_predict)" if completion.truncated else ""
            raise StructuredOutputError(
                f"schema-constrained output is not valid JSON{hint}: {exc}",
                raw=completion.content,
                path="/api/chat",
            ) from exc
        try:
            value = schema.model_validate(parsed)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"output parsed but failed {schema.__name__} validation: {exc.error_count()} error(s)",
                raw=completion.content,
                path="/api/chat",
            ) from exc
        return StructuredCompletion(value=value, completion=completion)

    # -- introspection ----------------------------------------------------- #

    async def list_models(self) -> list[dict[str, Any]]:
        """Every model pulled on this host.

        A GET, unlike everything else here, so it does not go through
        :meth:`_post`.
        """
        client = self._ensure_client()
        url = f"{self._base_url}/api/tags"
        try:
            response = await client.get(url)
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable(f"Ollama timed out after {self._timeout}s", path="/api/tags") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(f"Ollama is unreachable: {exc}", path="/api/tags") from exc
        if response.status_code != httpx.codes.OK:
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code} for /api/tags",
                status=response.status_code,
                path="/api/tags",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a non-JSON body for /api/tags", path="/api/tags") from exc
        models = body.get("models") if isinstance(body, dict) else None
        return [entry for entry in (models or []) if isinstance(entry, dict)]

    async def has_model(self, name: str | None = None) -> bool:
        """Whether *name* (default: the configured model) is pulled.

        Matches ``"qwen3.5:4b"`` against a listing that may say
        ``"qwen3.5:4b"`` or, for a tagless request, ``"qwen3.5:latest"``, so a
        caller does not have to know that Ollama appends ``:latest``.
        """
        wanted = name or self._model
        candidates = {wanted, wanted if ":" in wanted else f"{wanted}:latest"}
        return any(str(entry.get("name") or "") in candidates for entry in await self.list_models())

    async def health(self) -> tuple[bool, str]:
        """Reachability plus whether the configured model is present.

        Returns a flag and a human-readable detail rather than raising: this is
        called to report status, and an unreachable dependency is the answer, not
        an error. Model-present is folded in because "Ollama is up but the model
        was never pulled" fails every request while every connection check passes.
        """
        started = time.perf_counter()
        try:
            present = await self.has_model()
        except OllamaError as exc:
            return False, str(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        if not present:
            return False, f"Ollama is up but {self._model!r} is not pulled (run: ollama pull {self._model})"
        return True, f"{self._model} available ({latency_ms:.0f} ms)"


def _ns_to_ms(value: Any) -> float:
    """Ollama's nanosecond durations, in milliseconds. Absent means zero."""
    try:
        return float(value) / NANOSECONDS_PER_MS
    except (TypeError, ValueError):
        return 0.0


def _error_detail(response: httpx.Response) -> str:
    """Pull Ollama's error string out of a failure body.

    Ollama uses ``{"error": "..."}``, but a proxy or a crash can produce plain
    text, so the text body is the fallback. Truncated: an error message ends up in
    logs, and a multi-megabyte HTML error page in a log line is its own outage.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])[:500]
    return response.text[:200]

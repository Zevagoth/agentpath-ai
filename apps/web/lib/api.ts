/**
 * Client for the AgentPath API.
 *
 * Error handling is the point of this module. "The engine is down" and "no
 * procedure matched" must never reach the UI as the same state: the first is a
 * system failure the agent should escalate, the second is a real answer about the
 * knowledge base. The API separates them by status code, and so does this.
 */

import type {
  AnswerRequest,
  AnswerResponse,
  ApiError,
  SearchRequest,
  SearchResponse,
} from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENTPATH_API_URL ?? "http://127.0.0.1:8000";

/**
 * A request that never produced a result, for either endpoint.
 *
 * Only transport and configuration failures reach this. An answer that abstained,
 * needs a clarification or could not be generated arrives as a normal HTTP 200
 * with a `status` field, because those are results rather than errors, and
 * collapsing them into an exception is exactly how "the model is down" ends up
 * looking like "there is no procedure".
 */
export class SearchError extends Error {
  readonly kind: ApiError["kind"];
  readonly detail?: string;

  constructor({ kind, message, detail }: ApiError) {
    super(message);
    this.name = "SearchError";
    this.kind = kind;
    this.detail = detail;
  }
}

export async function search(
  request: SearchRequest,
  options: { signal?: AbortSignal; baseUrl?: string } = {},
): Promise<SearchResponse> {
  const baseUrl = options.baseUrl ?? API_BASE_URL;

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: request.query,
        product_line: request.product_line ?? null,
        limit: request.limit ?? 3,
      }),
      signal: options.signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw cause;
    }
    throw new SearchError({
      kind: "unreachable",
      message: "Cannot reach the AgentPath API.",
      detail: `Tried ${baseUrl}. Start it with: make api`,
    });
  }

  if (response.ok) {
    return (await response.json()) as SearchResponse;
  }

  // 503 means the retrieval engine could not be consulted at all. Saying
  // "no procedure found" here would be a lie with operational consequences.
  if (response.status === 503) {
    const body = await safeJson(response);
    throw new SearchError({
      kind: "engine_unavailable",
      message:
        "The retrieval engine is unavailable, so no search was performed. This is not the same as finding no procedure.",
      detail: typeof body?.detail === "string" ? body.detail : undefined,
    });
  }

  if (response.status === 502) {
    const body = await safeJson(response);
    throw new SearchError({
      kind: "engine_failed",
      message: "The retrieval engine rejected the request.",
      detail: typeof body?.detail === "string" ? body.detail : undefined,
    });
  }

  if (response.status === 422) {
    throw new SearchError({
      kind: "invalid",
      message: "That query could not be accepted. Try rephrasing the case.",
    });
  }

  throw new SearchError({
    kind: "unknown",
    message: `Unexpected response from the API (HTTP ${response.status}).`,
  });
}

async function safeJson(response: Response): Promise<Record<string, unknown> | null> {
  try {
    return (await response.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/**
 * Ask for a grounded answer.
 *
 * The important difference from {@link search}: almost nothing here is an
 * exception. The endpoint answers HTTP 200 whenever the service ran, and the
 * interesting outcomes - abstained, needs a clarification, the model server
 * failed - are values in `status`. Only two things throw: the API is not
 * reachable at all, and the deployment has no answer model configured.
 */
export async function answer(
  request: AnswerRequest,
  options: { signal?: AbortSignal; baseUrl?: string } = {},
): Promise<AnswerResponse> {
  const baseUrl = options.baseUrl ?? API_BASE_URL;

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/api/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: request.query,
        product_line: request.product_line ?? null,
        limit: request.limit ?? 3,
        case_context: request.case_context ?? null,
      }),
      signal: options.signal,
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw cause;
    }
    throw new SearchError({
      kind: "unreachable",
      message: "Cannot reach the AgentPath API.",
      detail: `Tried ${baseUrl}. Start it with: make api`,
    });
  }

  if (response.ok) {
    return (await response.json()) as AnswerResponse;
  }

  // 503 on this route means one specific thing: no answer model is wired up on
  // this deployment. It is a deployment fact, not a result, and it must not read
  // as "the model failed" - which is a 200 with status=generation_failed - nor as
  // "no procedure exists".
  if (response.status === 503) {
    const body = await safeJson(response);
    throw new SearchError({
      kind: "generation_unavailable",
      message:
        typeof body?.message === "string"
          ? body.message
          : "Answer generation is not configured on this deployment. Search is unaffected.",
      detail: "Set OLLAMA_CHAT_MODEL and start Ollama, then restart the API.",
    });
  }

  // The case-context contract rejects personal and financial fields by schema
  // rather than by ignoring them, so a 422 here is usually informative.
  if (response.status === 422) {
    const body = await safeJson(response);
    throw new SearchError({
      kind: "invalid",
      message: "The API rejected that request.",
      detail: describeValidationError(body),
    });
  }

  throw new SearchError({
    kind: "unknown",
    message: `Unexpected response from the API (HTTP ${response.status}).`,
  });
}

/**
 * Turn FastAPI's validation payload into one readable line.
 *
 * Worth doing rather than showing "422": the most likely cause is a field the
 * case-context contract refuses, and naming it teaches the agent something true
 * about the system instead of looking like a bug.
 */
function describeValidationError(body: Record<string, unknown> | null): string | undefined {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return undefined;

  const parts = detail.slice(0, 4).map((entry) => {
    const item = entry as { loc?: unknown[]; msg?: string };
    const field = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
    return field ? `${field}: ${item.msg ?? "rejected"}` : (item.msg ?? "rejected");
  });
  return parts.length > 0 ? parts.join("; ") : undefined;
}

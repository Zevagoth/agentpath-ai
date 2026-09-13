/**
 * Client for the AgentPath API.
 *
 * Error handling is the point of this module. "The engine is down" and "no
 * procedure matched" must never reach the UI as the same state: the first is a
 * system failure the agent should escalate, the second is a real answer about the
 * knowledge base. The API separates them by status code, and so does this.
 */

import type { ApiError, SearchRequest, SearchResponse } from "./types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENTPATH_API_URL ?? "http://127.0.0.1:8000";

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

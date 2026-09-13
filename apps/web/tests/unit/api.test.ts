import { afterEach, describe, expect, it, vi } from "vitest";

import { SearchError, search } from "@/lib/api";

const BASE = "http://api.test";

function mockFetch(response: Response | Error | DOMException) {
  const fn = vi.fn(async () => {
    // jsdom's DOMException is not an `instanceof Error`, so it needs naming
    // explicitly here or an aborted request would look like a bad response.
    if (response instanceof Error || response instanceof DOMException) throw response;
    return response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("search client", () => {
  it("posts the query, filter and limit", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ procedures: [] }), { status: 200 }),
    );

    await search({ query: "hotel has no room", product_line: "hotels" }, { baseUrl: BASE });

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${BASE}/api/search`);
    expect(JSON.parse(init.body as string)).toEqual({
      query: "hotel has no room",
      product_line: "hotels",
      limit: 3,
    });
  });

  it("distinguishes an unreachable API from an empty result", async () => {
    mockFetch(new TypeError("Failed to fetch"));

    await expect(search({ query: "x" }, { baseUrl: BASE })).rejects.toMatchObject({
      kind: "unreachable",
    });
  });

  it("treats 503 as engine unavailable and says so explicitly", async () => {
    mockFetch(
      new Response(JSON.stringify({ detail: "connection refused" }), { status: 503 }),
    );

    try {
      await search({ query: "x" }, { baseUrl: BASE });
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(SearchError);
      const searchError = error as SearchError;
      expect(searchError.kind).toBe("engine_unavailable");
      // The wording matters: this must not read as "no procedure exists".
      expect(searchError.message).toContain("not the same as finding no procedure");
      expect(searchError.detail).toBe("connection refused");
    }
  });

  it("maps 502 to an engine failure", async () => {
    mockFetch(new Response("{}", { status: 502 }));
    await expect(search({ query: "x" }, { baseUrl: BASE })).rejects.toMatchObject({
      kind: "engine_failed",
    });
  });

  it("maps 422 to an invalid query", async () => {
    mockFetch(new Response("{}", { status: 422 }));
    await expect(search({ query: "x" }, { baseUrl: BASE })).rejects.toMatchObject({
      kind: "invalid",
    });
  });

  it("propagates an abort without turning it into an error state", async () => {
    mockFetch(new DOMException("aborted", "AbortError"));
    await expect(search({ query: "x" }, { baseUrl: BASE })).rejects.toThrowError(
      /aborted/,
    );
  });
});

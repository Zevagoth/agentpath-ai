import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AgentConsole } from "@/components/AgentConsole";
import type { SearchResponse } from "@/lib/types";

/**
 * These cover the search mode, which is now one of two modes on the console. The
 * assertions are unchanged; they are pinned to search explicitly because the
 * console opens on the answer mode, which is what an agent mid-call needs.
 */
function renderSearchMode() {
  return render(<AgentConsole initialMode="search" />);
}

function response(overrides: Partial<SearchResponse> = {}): SearchResponse {
  return {
    mode: "hybrid",
    query: {
      original: "hotel has no room",
      normalized: "hotel has no room",
      language: "en",
      search_text: "hotel has no room overbooking",
      expansions: ["overbooking"],
      corrections: [{ before: "overboking", after: "overbooking" }],
      concepts: [],
      scenario_tags: ["overbooking"],
      detected_product_line: "hotels",
      requested_product_line: null,
    },
    evidence: {
      level: "strong",
      reason: "NW-HTL-014 scored 0.61, ahead of the next procedure by 0.08.",
      abstained: false,
      message: null,
    },
    procedures: [
      {
        rank: 1,
        procedure_id: "NW-HTL-014",
        title: "Property unable to accommodate a confirmed guest",
        product_line: "hotels",
        version: "2.1",
        effective_date: "2026-03-01",
        score: 0.612,
        scenario_tags: ["overbooking"],
        matched_sections: 5,
        citations: [
          {
            chunk_id: "c1",
            document_id: "d1",
            snippet: "the property <em>cannot host</em> a confirmed guest",
            score: 0.612,
            lexical_score: 0.31,
            semantic_score: 0.81,
          },
        ],
      },
    ],
    conflicts: [],
    timings_ms: { understanding: 0.1, retrieval: 148, aggregation: 0, total: 148.4 },
    diagnostics: {},
    ...overrides,
  };
}

function stubFetch(payload: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(payload), { status })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("search console", () => {
  it("renders the case field, product filter and branding", () => {
    renderSearchMode();
    expect(screen.getByLabelText(/describe the customer's case/i)).toBeTruthy();
    expect(screen.getByTestId("product-line")).toBeTruthy();
    expect(screen.getByText("AgentPath AI")).toBeTruthy();
    expect(screen.getByText(/fictional/i)).toBeTruthy();
  });

  it("disables search until the agent types something", async () => {
    renderSearchMode();
    const button = screen.getByTestId("search-button") as HTMLButtonElement;
    expect(button.disabled).toBe(true);

    await userEvent.type(screen.getByTestId("case-input"), "hotel");
    expect(button.disabled).toBe(false);
  });

  it("flips the case field to right-to-left for Arabic input", async () => {
    renderSearchMode();
    const input = screen.getByTestId("case-input");
    await userEvent.type(input, "الفندق ما عنده غرف");
    expect(input.getAttribute("dir")).toBe("rtl");
  });

  it("shows ranked procedures with citations and timing", async () => {
    stubFetch(response());
    renderSearchMode();

    await userEvent.type(screen.getByTestId("case-input"), "hotel has no room");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("procedure-card")).toBeTruthy());
    expect(screen.getByTestId("procedure-id").textContent).toBe("NW-HTL-014");
    expect(screen.getByTestId("evidence-badge").getAttribute("data-level")).toBe("strong");
    expect(screen.getByText(/cannot host/)).toBeTruthy();
    expect(screen.getByTestId("timing").textContent).toMatch(/ms server/);
  });

  it("shows the corrections and expansions it applied", async () => {
    stubFetch(response());
    renderSearchMode();

    await userEvent.type(screen.getByTestId("case-input"), "overboking at the hotle");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("correction")).toBeTruthy());
    expect(screen.getByTestId("correction").textContent).toContain("overboking");
    expect(screen.getByTestId("expansion").textContent).toContain("overbooking");
  });

  it("renders abstention as its own state, not as an error", async () => {
    stubFetch(
      response({
        evidence: {
          level: "insufficient",
          reason: "No passage scored above the similarity threshold.",
          abstained: true,
          message: "No confirmed procedure was found in the available knowledge base.",
        },
        procedures: [],
      }),
    );
    renderSearchMode();

    await userEvent.type(screen.getByTestId("case-input"), "how do I reset my password");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("abstention")).toBeTruthy());
    expect(screen.getByTestId("abstention-message").textContent).toBe(
      "No confirmed procedure was found in the available knowledge base.",
    );
    expect(screen.queryByTestId("error")).toBeNull();
    expect(screen.queryByTestId("procedure-card")).toBeNull();
  });

  it("renders an engine outage as an error, distinct from abstention", async () => {
    stubFetch({ detail: "connection refused" }, 503);
    renderSearchMode();

    await userEvent.type(screen.getByTestId("case-input"), "hotel has no room");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("error")).toBeTruthy());
    expect(screen.getByTestId("error").textContent).toContain(
      "not the same as finding no procedure",
    );
    expect(screen.queryByTestId("abstention")).toBeNull();
  });

  it("surfaces overlapping procedures when the API reports them", async () => {
    stubFetch(
      response({
        conflicts: ["NW-HTL-014 (v2.1) and NW-HTL-021 (v1.4) both cover: overbooking."],
      }),
    );
    renderSearchMode();

    await userEvent.type(screen.getByTestId("case-input"), "hotel problem");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("conflicts")).toBeTruthy());
  });
});

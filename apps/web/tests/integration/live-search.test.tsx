import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it } from "vitest";

import { AgentConsole } from "@/components/AgentConsole";

/**
 * The demo, asserted end to end against the running stack.
 *
 * This renders the real console component, which calls the real client, which
 * talks to the real FastAPI service, which queries real RAGFlow. Only the browser
 * engine is substituted (jsdom instead of Chromium), so everything the product
 * actually does is exercised: query understanding, hybrid retrieval, evidence
 * grading, and the three states an agent sees.
 *
 * The equivalent Chromium specs live in tests/e2e/ and assert the same three
 * outcomes; they need a Playwright browser download, which is why this exists as
 * the runnable proof.
 *
 * Requires: make ragflow-up && make api
 */

const API = process.env.NEXT_PUBLIC_AGENTPATH_API_URL ?? "http://127.0.0.1:8000";

let apiReachable = false;

beforeAll(async () => {
  try {
    const response = await fetch(`${API}/health`);
    apiReachable = response.ok;
  } catch {
    apiReachable = false;
  }
  if (!apiReachable) {
    console.warn(`AgentPath API not reachable at ${API}; skipping live tests.`);
  }
});

async function submit(query: string) {
  render(<AgentConsole initialMode="search" />);
  const input = screen.getByTestId("case-input");
  await userEvent.click(input);
  await userEvent.paste(query);
  await userEvent.click(screen.getByTestId("search-button"));
}

describe("live search against the running stack", () => {
  it("Arabic case returns NW-HTL-014 first", async () => {
    if (!apiReachable) return;
    await submit("الفندق ما عنده غرف والضيف واصل");

    await waitFor(
      () => expect(screen.getAllByTestId("procedure-card").length).toBeGreaterThan(0),
      { timeout: 20_000 },
    );

    const cards = screen.getAllByTestId("procedure-card");
    expect(cards[0].getAttribute("data-procedure-id")).toBe("NW-HTL-014");
    expect(screen.getByTestId("detected-language").textContent).toContain("Arabic");
    // Right-to-left content must render right-to-left.
    expect(screen.getByTestId("case-input").getAttribute("dir")).toBe("rtl");
    // A recommendation without its source is not shippable: every returned card
    // carries one, so this matches once per card.
    expect(screen.getAllByText(/Source passages/i).length).toBe(cards.length);
  });

  it("misspelled case is corrected and still returns NW-HTL-014", async () => {
    if (!apiReachable) return;
    await submit("overboking at the hotle");

    await waitFor(
      () => expect(screen.getAllByTestId("procedure-card").length).toBeGreaterThan(0),
      { timeout: 20_000 },
    );

    const cards = screen.getAllByTestId("procedure-card");
    expect(cards[0].getAttribute("data-procedure-id")).toBe("NW-HTL-014");

    const corrections = screen.getAllByTestId("correction").map((node) => node.textContent);
    expect(corrections.join(" ")).toContain("overboking");
    expect(corrections.join(" ")).toContain("hotle");
  });

  it("email password question abstains instead of inventing a procedure", async () => {
    if (!apiReachable) return;
    await submit("how do I reset my email password");

    await waitFor(() => expect(screen.getByTestId("abstention")).toBeTruthy(), {
      timeout: 20_000,
    });

    expect(screen.getByTestId("abstention-message").textContent).toBe(
      "No confirmed procedure was found in the available knowledge base.",
    );
    expect(screen.queryAllByTestId("procedure-card")).toHaveLength(0);
    expect(screen.getByTestId("evidence-badge").getAttribute("data-level")).toBe(
      "insufficient",
    );
  });

  it("product line filter narrows to the refunds procedure", async () => {
    if (!apiReachable) return;
    render(<AgentConsole initialMode="search" />);
    await userEvent.click(screen.getByTestId("case-input"));
    await userEvent.paste("refund has not arrived");
    await userEvent.selectOptions(screen.getByTestId("product-line"), "refunds");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(
      () => expect(screen.getAllByTestId("procedure-card").length).toBeGreaterThan(0),
      { timeout: 20_000 },
    );
    expect(screen.getAllByTestId("procedure-card")[0].getAttribute("data-procedure-id")).toBe(
      "NW-REF-018",
    );
  });

  it("shows the response time to the agent", async () => {
    if (!apiReachable) return;
    await submit("hotel has no room for the guest");

    await waitFor(() => expect(screen.getByTestId("timing")).toBeTruthy(), {
      timeout: 20_000,
    });
    expect(screen.getByTestId("timing").textContent).toMatch(/ms server/);
  });
});

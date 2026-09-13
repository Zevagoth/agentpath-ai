import { expect, test } from "@playwright/test";

/**
 * Browser checks against the real stack.
 *
 * These three cases are the demo: an Arabic query finding an English procedure, a
 * misspelled query being corrected, and an unanswerable question producing an
 * explicit refusal rather than an invented answer.
 *
 * Requires the API and RAGFlow to be running (`make ragflow-up && make api`).
 */

const API = process.env.NEXT_PUBLIC_AGENTPATH_API_URL ?? "http://127.0.0.1:8000";

test.beforeAll(async ({ request }) => {
  const response = await request.get(`${API}/health`);
  test.skip(
    !response.ok(),
    `AgentPath API is not healthy at ${API}; start the stack with: make ragflow-up && make api`,
  );
});

test("Arabic case finds the English procedure first", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("case-input").fill("الفندق ما عنده غرف والضيف واصل");
  await page.getByTestId("search-button").click();

  const cards = page.getByTestId("procedure-card");
  await expect(cards.first()).toBeVisible();
  await expect(cards.first()).toHaveAttribute("data-procedure-id", "NW-HTL-014");
  await expect(page.getByTestId("detected-language")).toContainText("Arabic");

  // Right-to-left input must render right-to-left.
  await expect(page.getByTestId("case-input")).toHaveAttribute("dir", "rtl");

  // Every recommendation must carry its source.
  await expect(cards.first().getByText(/Source passages/i)).toBeVisible();
});

test("misspelled case is corrected and still finds the procedure", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("case-input").fill("overboking at the hotle");
  await page.getByTestId("search-button").click();

  const cards = page.getByTestId("procedure-card");
  await expect(cards.first()).toBeVisible();
  await expect(cards.first()).toHaveAttribute("data-procedure-id", "NW-HTL-014");

  const corrections = page.getByTestId("correction");
  await expect(corrections.first()).toBeVisible();
  await expect(page.getByTestId("correction").first()).toContainText("overboking");
});

test("unanswerable question abstains instead of inventing a procedure", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("case-input").fill("how do I reset my email password");
  await page.getByTestId("search-button").click();

  await expect(page.getByTestId("abstention")).toBeVisible();
  await expect(page.getByTestId("abstention-message")).toHaveText(
    "No confirmed procedure was found in the available knowledge base.",
  );
  await expect(page.getByTestId("procedure-card")).toHaveCount(0);
  await expect(page.getByTestId("evidence-badge")).toHaveAttribute(
    "data-level",
    "insufficient",
  );
});

test("product line filter is available and search still returns results", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("case-input").fill("refund has not arrived");
  await page.getByTestId("product-line").selectOption("refunds");
  await page.getByTestId("search-button").click();

  const cards = page.getByTestId("procedure-card");
  await expect(cards.first()).toBeVisible();
  await expect(cards.first()).toHaveAttribute("data-procedure-id", "NW-REF-018");
});

test("response time is displayed to the agent", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("case-input").fill("hotel has no room for the guest");
  await page.getByTestId("search-button").click();

  await expect(page.getByTestId("timing")).toContainText("ms server");
});

import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end configuration.
 *
 * These specs run against the real stack: the Next.js build talking to the real
 * FastAPI service talking to real RAGFlow. That is the point - they are the only
 * tests that prove the demo works from a browser.
 *
 * Start the API separately (`make api`); Playwright starts the web server itself.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
    locale: "en-US",
  },
  // `channel: "chromium"` uses the full Chromium build that `playwright install
  // chromium` downloads, instead of the separate headless-shell package. One
  // browser download instead of two, and the same rendering path a real agent sees.
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chromium" },
    },
  ],
  webServer: {
    // Invoked directly rather than through `npm run start` so the port is not
    // passed twice (the npm script already pins one).
    command: "npx next start --hostname 127.0.0.1 --port 3100",
    url: "http://127.0.0.1:3100",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});

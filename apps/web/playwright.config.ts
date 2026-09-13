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

/**
 * An already-installed chromium-family browser can be used instead of the
 * Playwright download, which is 173.7 MiB and has to come over the network.
 * Export the path and it is passed straight to `launchOptions.executablePath`:
 *
 *     export PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
 *
 * Unset - the default - changes nothing: the downloaded build is used exactly as
 * before. `make browser-check` reports which of the two applies and why.
 */
const localChromium = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH?.trim();

const browserSelection = localChromium
  ? // `channel` and `executablePath` are mutually exclusive, so the channel is
    // dropped rather than added to.
    { launchOptions: { executablePath: localChromium } }
  : { channel: "chromium" as const };

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
      use: { ...devices["Desktop Chrome"], ...browserSelection },
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

#!/usr/bin/env node
/**
 * Which browser will Playwright actually use, and can it be launched?
 *
 * Read-only inspection plus one real browser launch. Nothing is downloaded and
 * nothing is installed. The 173.7 MiB `playwright install chromium` archive is
 * deliberately avoided: any locally installed chromium-family build can be used
 * instead by exporting PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH.
 *
 * Resolution order - the same order apps/web/playwright.config.ts applies, and
 * the reason it lives in scripts/resolve-browser.mjs rather than here:
 *   1. PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH, if set. Wired to
 *      `launchOptions.executablePath`.
 *   2. Otherwise the Chromium build downloaded into the Playwright browser
 *      cache, which is what `channel: "chromium"` resolves to.
 *
 * This check stops there, exactly like playwright.config.ts. Browsers found on
 * PATH are reported either way, so the fix for case 2 failing is visible without
 * guessing. (scripts/generate-screenshots.mjs opts into a third tier - using a
 * PATH browser directly - because rendering a local HTML file has no reason to
 * require a download. It prints which browser it picked.)
 *
 * Usage:
 *     node scripts/browser-check.mjs            # report, then launch
 *     node scripts/browser-check.mjs --no-launch
 *
 * Exit codes:
 *     0  a usable browser was resolved (and launched, unless --no-launch)
 *     1  no usable browser, or the resolved one failed to launch
 */

import {
  ENV_VAR,
  browserCacheDir,
  detectPathBrowsers,
  downloadedChromium,
  flatpakBrowsers,
  resolveBrowser,
  versionOf,
} from "./resolve-browser.mjs";

const argv = process.argv.slice(2);
const shouldLaunch = !argv.includes("--no-launch");

function fail(message, hints = []) {
  console.error(`\nbrowser-check: FAIL - ${message}`);
  for (const hint of hints) console.error(`  ${hint}`);
  console.error("");
  process.exit(1);
}

async function main() {
  const configured = (process.env[ENV_VAR] ?? "").trim();
  const detected = detectPathBrowsers();
  const flatpak = flatpakBrowsers();
  const downloaded = downloadedChromium();

  console.log("Browsers on PATH");
  if (detected.length === 0) {
    console.log("  none of the known chromium-family names resolve");
  }
  for (const browser of detected) {
    console.log(`  ${browser.executable}`);
    console.log(`      version : ${browser.version}`);
    if (browser.real !== browser.executable) {
      console.log(`      real    : ${browser.real}`);
    }
    for (const alias of browser.aliases) {
      console.log(`      alias   : ${alias}`);
    }
  }

  console.log("\nFlatpak");
  if (!flatpak.available) {
    console.log("  flatpak is not installed");
  } else if (flatpak.apps.length === 0) {
    console.log("  installed, no chromium-family application");
  } else {
    for (const app of flatpak.apps) {
      console.log(`  ${app} (needs a 'flatpak run' wrapper; not usable as an executable path)`);
    }
  }

  console.log(`\nPlaywright browser cache (${browserCacheDir()})`);
  console.log(`  downloaded Chromium build: ${downloaded ?? "none"}`);

  console.log(`\n${ENV_VAR}`);
  console.log(`  ${configured === "" ? "not set" : configured}`);

  // ----------------------------------------------------------------------- #
  // The actual question.
  // ----------------------------------------------------------------------- #

  const resolution = resolveBrowser();
  if (!resolution.executable) {
    fail(resolution.problem, resolution.hints);
  }

  const executable = resolution.executable;

  console.log("\nPlaywright will use");
  console.log(`  executable : ${executable}`);
  console.log(`  version    : ${versionOf(executable)}`);
  console.log(`  because    : ${resolution.reason}`);

  if (!shouldLaunch) {
    console.log("\nbrowser-check: OK (launch skipped, --no-launch)\n");
    return;
  }

  // Launching is the only evidence that counts. A path that exists but cannot
  // be driven over the DevTools protocol is not a usable browser.
  let chromium;
  try {
    ({ chromium } = await import("@playwright/test"));
  } catch (error) {
    fail(`@playwright/test could not be imported: ${error.message}`, [
      "Run 'make install-web' first.",
    ]);
  }

  let browser;
  try {
    browser = await chromium.launch({ executablePath: executable });
  } catch (error) {
    fail(`Playwright could not launch ${executable}`, [
      String(error.message).split("\n").slice(0, 6).join("\n  "),
    ]);
  }

  try {
    const page = await browser.newPage();
    await page.setContent("<title>browser-check</title><h1>ok</h1>");
    const heading = await page.textContent("h1");
    if (heading !== "ok") {
      fail(`launched, but the page did not render as expected (got ${JSON.stringify(heading)})`);
    }
    console.log("\nLaunched successfully");
    console.log(`  reported by browser : ${browser.version()}`);
    console.log(`  rendered a page and read back its DOM text: "${heading}"`);
  } finally {
    await browser.close();
  }

  console.log("\nbrowser-check: OK\n");
}

main().catch((error) => {
  fail(error instanceof Error ? error.message : String(error));
});

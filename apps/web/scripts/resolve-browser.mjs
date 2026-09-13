/**
 * Which chromium-family browser should Playwright drive, and where is it?
 *
 * Extracted from scripts/browser-check.mjs so the diagnostic and the screenshot
 * generator answer that question the same way instead of each inventing its own
 * search. Nothing in here downloads, installs or modifies anything: it inspects
 * the filesystem and runs `--version` on candidates it already found.
 *
 * Resolution order:
 *   1. PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH, if set. This is what
 *      apps/web/playwright.config.ts wires to `launchOptions.executablePath`.
 *   2. The Chromium build in the Playwright browser cache, which is what
 *      `channel: "chromium"` resolves to.
 *   3. Only with `allowPathDiscovery: true` - the first chromium-family browser
 *      on PATH. playwright.config.ts deliberately does not do this, so the
 *      default is off and callers that opt in must say which browser they used.
 *
 * The 173.7 MiB `playwright install chromium` archive is never fetched from
 * here; tier 3 exists precisely so screenshot generation works on a machine that
 * already has a browser.
 */

import { accessSync, constants, readdirSync, realpathSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir } from "node:os";
import path from "node:path";

export const ENV_VAR = "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH";

/**
 * Chromium-family executables worth trying, most standard name first. Every one
 * of these speaks the DevTools protocol Playwright drives.
 */
export const PATH_CANDIDATES = [
  "chromium",
  "chromium-browser",
  "google-chrome",
  "google-chrome-stable",
  "google-chrome-beta",
  "google-chrome-unstable",
  "brave",
  "brave-browser",
  "microsoft-edge",
  "microsoft-edge-stable",
  "microsoft-edge-beta",
  "vivaldi",
  "vivaldi-stable",
  "thorium",
  "thorium-browser",
];

/** Flatpak application ids for the same browsers. Reported, never used: Flatpak
 * needs a `flatpak run` wrapper, and Playwright requires a real executable. */
export const FLATPAK_CANDIDATES = [
  "org.chromium.Chromium",
  "com.google.Chrome",
  "com.brave.Browser",
  "com.microsoft.Edge",
  "com.vivaldi.Vivaldi",
  "io.github.ungoogled_software.ungoogled_chromium",
];

// --------------------------------------------------------------------------- #
// Inspection helpers. No shell is involved anywhere: every child process is
// spawned with an argument vector, so a path with spaces or shell characters in
// it cannot turn into a command.
// --------------------------------------------------------------------------- #

export function isExecutableFile(candidate) {
  try {
    if (!statSync(candidate).isFile()) return false;
    accessSync(candidate, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Resolve a bare name against PATH without invoking a shell. */
export function onPath(name) {
  const dirs = (process.env.PATH ?? "").split(path.delimiter).filter(Boolean);
  for (const dir of dirs) {
    const candidate = path.join(dir, name);
    if (isExecutableFile(candidate)) return candidate;
  }
  return null;
}

export function versionOf(executable) {
  try {
    const out = execFileSync(executable, ["--version"], {
      encoding: "utf8",
      timeout: 15_000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    return out.trim().split("\n")[0] || "unknown";
  } catch {
    return "version unavailable";
  }
}

export function resolvedTarget(executable) {
  try {
    const real = realpathSync(executable);
    return real === executable ? null : real;
  } catch {
    return null;
  }
}

/** The directory `playwright install` writes into. */
export function browserCacheDir() {
  if (process.env.PLAYWRIGHT_BROWSERS_PATH) return process.env.PLAYWRIGHT_BROWSERS_PATH;
  return path.join(homedir(), ".cache", "ms-playwright");
}

/**
 * The downloaded Chromium build `channel: "chromium"` needs. Looked up on disk
 * rather than through the Playwright registry so the answer stays the same
 * whether or not the package can be imported.
 */
export function downloadedChromium() {
  const cache = browserCacheDir();
  let entries;
  try {
    entries = readdirSync(cache);
  } catch {
    return null;
  }
  const builds = entries.filter((name) => /^chromium-\d+$/.test(name)).sort();
  for (const build of builds.reverse()) {
    const executable = path.join(cache, build, "chrome-linux", "chrome");
    if (isExecutableFile(executable)) return executable;
  }
  return null;
}

export function flatpakBrowsers() {
  const flatpak = onPath("flatpak");
  if (!flatpak) return { available: false, apps: [] };
  try {
    const out = execFileSync(flatpak, ["list", "--app", "--columns=application"], {
      encoding: "utf8",
      timeout: 20_000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    const installed = new Set(
      out
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean),
    );
    return { available: true, apps: FLATPAK_CANDIDATES.filter((app) => installed.has(app)) };
  } catch {
    return { available: true, apps: [] };
  }
}

/** Every distinct chromium-family browser reachable through PATH. */
export function detectPathBrowsers() {
  const seen = new Map();
  for (const name of PATH_CANDIDATES) {
    const executable = onPath(name);
    if (!executable) continue;
    const real = resolvedTarget(executable) ?? executable;
    // Several names routinely point at one binary (google-chrome -> chromium).
    // Group them so the report shows one browser, not four.
    const entry = seen.get(real);
    if (entry) {
      entry.aliases.push(`${name} -> ${executable}`);
      continue;
    }
    seen.set(real, {
      name,
      executable,
      real,
      version: versionOf(executable),
      aliases: [],
    });
  }
  return [...seen.values()];
}

/**
 * Resolve the browser to launch.
 *
 * Returns `{ executable, reason, tier }` on success and
 * `{ executable: null, problem, hints, detected }` when nothing is usable, so
 * the caller decides how to fail. `detected` is always the PATH scan, because
 * that list is the fix when tiers 1 and 2 are both empty.
 */
export function resolveBrowser({ allowPathDiscovery = false } = {}) {
  const configured = (process.env[ENV_VAR] ?? "").trim();
  const detected = detectPathBrowsers();

  if (configured !== "") {
    if (!isExecutableFile(configured)) {
      return {
        executable: null,
        detected,
        problem: `${ENV_VAR} is set but is not an executable file: ${configured}`,
        hints: ["Point it at a browser binary, or unset it to fall back to the downloaded build."],
      };
    }
    return {
      executable: configured,
      tier: "env",
      reason: `${ENV_VAR} is set, so playwright.config.ts passes launchOptions.executablePath`,
      detected,
    };
  }

  const downloaded = downloadedChromium();
  if (downloaded) {
    return {
      executable: downloaded,
      tier: "download",
      reason: 'no override is set, so playwright.config.ts uses channel: "chromium"',
      detected,
    };
  }

  if (allowPathDiscovery && detected.length > 0) {
    const first = detected[0];
    return {
      executable: first.executable,
      tier: "path",
      reason: `no override is set and no Chromium build is downloaded, so the first chromium-family browser on PATH is used (${first.name})`,
      detected,
    };
  }

  const hints = [
    "Either download the build (~173 MiB):  make install-web-browsers",
    "or point Playwright at a browser already on this machine, no download:",
  ];
  if (detected.length > 0) {
    hints.push(`    export ${ENV_VAR}=${detected[0].executable}`);
    hints.push("  then re-run: make browser-check");
  } else {
    hints.push(`    export ${ENV_VAR}=/path/to/chromium`);
    hints.push("  No chromium-family browser was found on PATH, so one has to be installed");
    hints.push("  first. That is a system-level change and is out of scope for this check.");
  }

  return {
    executable: null,
    detected,
    problem: `no usable browser: ${ENV_VAR} is not set and no Chromium build is downloaded, so channel: "chromium" has nothing to resolve`,
    hints,
  };
}

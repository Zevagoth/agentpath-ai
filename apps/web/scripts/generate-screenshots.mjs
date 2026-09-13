#!/usr/bin/env node
/**
 * Render the synthetic AgentDesk screens to PNG plus a sidecar JSON.
 *
 * Source of truth for the screens is data/synthetic/screens/*.html - committed,
 * reviewable, self-contained. This script only renders them; it never invents
 * content. Source of truth for the procedure metadata is
 * data/synthetic/procedures/NW-HTL-014.yaml, so the version, step order, screen
 * name and control name in the sidecar cannot drift from the procedure.
 *
 * It lives in apps/web/scripts because that is where @playwright/test is
 * installed, and it reuses scripts/resolve-browser.mjs so it finds the same
 * browser `make browser-check` reports. Nothing is downloaded, ever.
 *
 * Usage:
 *     node scripts/generate-screenshots.mjs              # write PNG + JSON
 *     node scripts/generate-screenshots.mjs --check      # verify, write nothing
 *     node scripts/generate-screenshots.mjs --only agentdesk-supplier-log
 *
 * Exit codes:
 *     0  every screen rendered (or verified) successfully
 *     1  a render, an assertion or a comparison failed
 *
 * --------------------------------------------------------------------------- *
 * Determinism contract
 * --------------------------------------------------------------------------- *
 * Byte-identical PNGs are NOT the requirement: two chromium builds, or two
 * machines with different font packages, legitimately rasterise the same DOM to
 * different bytes. What --check asserts instead is:
 *
 *   1. identical dimensions (1440x900, deviceScaleFactor 1);
 *   2. identical dom_text, compared exactly - this is the authoritative text of
 *      the screenshot, so a content change can never slip through;
 *   3. a 64-bit difference hash within HAMMING_TOLERANCE of the committed one.
 *
 * Everything that could make a render vary is removed rather than tolerated:
 * no CSS animation or transition, no prefers-color-scheme rule, no Date-derived
 * text (every timestamp in the HTML is a literal), no randomness, no network.
 *
 * There is deliberately no OCR anywhere in this project. `ocr_text` is emitted
 * as null purely so the field exists for schema compatibility; dom_text, read
 * out of the rendered DOM, is what the system depends on.
 */

import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

import { ENV_VAR, resolveBrowser, versionOf } from "./resolve-browser.mjs";

// --------------------------------------------------------------------------- #
// Contract constants
// --------------------------------------------------------------------------- #

const VIEWPORT = { width: 1440, height: 900 };
const DEVICE_SCALE_FACTOR = 1;
const WATERMARK = "FICTIONAL — SYNTHETIC DEMO DATA";
const PROCEDURE_FILE = "NW-HTL-014.yaml";

/**
 * Maximum Hamming distance accepted between two 64-bit difference hashes of the
 * same screen.
 *
 * Measured on this repository's three screens with Chromium 153 (see below), not
 * guessed:
 *
 *   re-render of an unchanged screen, 3 consecutive runs ... 0 bits
 *     (the PNG bytes even came out identical, but that is not relied on)
 *   callout and arrow shifted 140px sideways, text unchanged ... 3 bits
 *   two different screens from this set ....................... 14-22 bits
 *
 * So the noise floor is 0 and the smallest change measured is 3. The tolerance
 * sits between them at 2: high enough that pure rasteriser noise - antialiasing,
 * font hinting, a different chromium build - does not fail the check, low enough
 * that the smallest visual change measured still does.
 *
 * A 64-bit dhash is coarse by design, and the two experiments above show the
 * limit honestly: the chip label change from "Confirmed" to "On hold" moved the
 * hash by 0 bits. That is why dom_text is compared exactly rather than fuzzily -
 * text is the axis that carries the meaning, and the hash only guards layout.
 * Highlight placement is guarded separately, and more strictly, by the geometry
 * assertions in measureHighlight(), which run in --check too.
 *
 * Raise this only with new measurements attached.
 */
const HAMMING_TOLERANCE = 2;

/**
 * The three screens, and only these three: they are the steps of NW-HTL-014
 * that genuinely need visual navigation. `screen` and `control` are asserted
 * against the YAML rather than trusted, so a rename in the procedure fails the
 * generator instead of silently producing a mislabelled screenshot.
 */
const SCREENS = [
  {
    id: "agentdesk-booking-detail-status",
    step: 1,
    screen: "Booking Detail",
    control: "Booking status chip",
    description: {
      en:
        "AgentDesk Booking Detail screen for the fictional booking NW-FAKE-100482 at the invented " +
        "Verdant Quay Hotel, Aldermere Bay. The Booking summary panel at the top of the main column " +
        "is highlighted with an orange ring around the green Booking status chip reading Confirmed; " +
        "the callout to its right is labelled step 1. Below the chip the screen shows the booking " +
        "reference NW-FAKE-100482, and the Stay and room panel shows check-in Thu 16 Apr 2026, " +
        "check-out Sun 19 Apr 2026, three nights, and the room type Twin Deluxe Room. Click the " +
        "status chip area to confirm the status before contacting the property. The whole screen " +
        "carries the watermark FICTIONAL — SYNTHETIC DEMO DATA.",
      ar:
        "شاشة تفاصيل الحجز في AgentDesk للحجز الوهمي NW-FAKE-100482 في فندق Verdant Quay المُتخيَّل " +
        "في Aldermere Bay. في أعلى العمود الرئيسي، تحيط دائرة برتقالية بشريحة حالة الحجز الخضراء " +
        "التي تقرأ Confirmed، والشرح المجاور لها على اليمين يحمل رقم الخطوة 1. تحت الشريحة يظهر رقم " +
        "الحجز NW-FAKE-100482، وتُظهر لوحة تفاصيل الإقامة تسجيل الوصول Thu 16 Apr 2026 وتسجيل " +
        "المغادرة Sun 19 Apr 2026 وثلاث ليالٍ ونوع الغرفة Twin Deluxe Room. انظر إلى شريحة الحالة " +
        "وتأكد منها قبل الاتصال بمكان الإقامة. تظهر على الشاشة بالكامل العلامة " +
        "FICTIONAL — SYNTHETIC DEMO DATA.",
    },
  },
  {
    id: "agentdesk-supplier-log",
    step: 4,
    screen: "Supplier Log",
    control: "Log supplier contact",
    description: {
      en:
        "AgentDesk Supplier Log screen for the fictional booking NW-FAKE-100482. The blue Log " +
        "supplier contact button sits at the top right of the Contact history panel and is " +
        "highlighted with an orange ring, with a step 4 callout and arrow to its left. The table " +
        "below already holds the timestamped entry 14 Apr 2026, 18:42 (UTC+03) with the red outcome " +
        "badge Cannot accommodate, recording that the property has no room for any of the three " +
        "booked nights. Click Log supplier contact to add such an entry. The whole screen carries " +
        "the watermark FICTIONAL — SYNTHETIC DEMO DATA.",
      ar:
        "شاشة سجل المورد في AgentDesk للحجز الوهمي NW-FAKE-100482. يقع زر Log supplier contact " +
        "الأزرق في أعلى يمين لوحة سجل الاتصالات وتحيط به دائرة برتقالية، مع شرح للخطوة 4 وسهم على " +
        "يساره. يحتوي الجدول أدناه على قيد بطابع زمني 14 Apr 2026, 18:42 (UTC+03) مع شارة النتيجة " +
        "الحمراء Cannot accommodate، وهي تسجّل أن مكان الإقامة لا يملك غرفة لأي من الليالي الثلاث " +
        "المحجوزة. اضغط Log supplier contact لإضافة قيد مماثل. تظهر على الشاشة بالكامل العلامة " +
        "FICTIONAL — SYNTHETIC DEMO DATA.",
    },
  },
  {
    id: "agentdesk-relocation-finder",
    step: 5,
    screen: "Relocation Finder",
    control: "Find equivalent stays",
    description: {
      en:
        "AgentDesk Relocation Finder screen for the fictional booking NW-FAKE-100482. The search " +
        "panel keeps the affected nights 16, 17 and 18 Apr 2026 and the default distance filter " +
        "Within 5 km. The blue Find equivalent stays button below the filters is highlighted with " +
        "an orange ring, with a step 5 callout and arrow to its left. The results panel underneath " +
        "lists three invented properties: Lantern Court Suites badged Better, Harbourline Grand " +
        "Stay badged Equivalent, and Old Mill Guesthouse badged Lower category. Click Find " +
        "equivalent stays without changing the distance filter. The whole screen carries the " +
        "watermark FICTIONAL — SYNTHETIC DEMO DATA.",
      ar:
        "شاشة أداة إيجاد البديل في AgentDesk للحجز الوهمي NW-FAKE-100482. تُبقي لوحة البحث على " +
        "الليالي المتأثرة 16 و17 و18 Apr 2026 وعلى فلتر المسافة الافتراضي Within 5 km. زر " +
        "Find equivalent stays الأزرق أسفل الفلاتر تحيط به دائرة برتقالية، مع شرح للخطوة 5 وسهم " +
        "على يساره. تعرض لوحة النتائج أدناه ثلاثة أماكن إقامة متخيَّلة: Lantern Court Suites بشارة " +
        "Better، وHarbourline Grand Stay بشارة Equivalent، وOld Mill Guesthouse بشارة " +
        "Lower category. اضغط Find equivalent stays دون تغيير فلتر المسافة. تظهر على الشاشة " +
        "بالكامل العلامة FICTIONAL — SYNTHETIC DEMO DATA.",
    },
  },
];

/**
 * The console's own UI language. AgentDesk labels are English on every screen,
 * including for an Arabic-speaking agent, so an Arabic answer still has to name
 * the English control - "Find equivalent stays", not a translation of it.
 */
const UI_LANGUAGE = "en";
const UI_LANGUAGE_NOTE =
  "AgentDesk renders its labels in English for every agent, so control names such as " +
  '"Find equivalent stays" stay in English even when the answer is Arabic.';

// --------------------------------------------------------------------------- #
// Paths. Everything written into a committed file is repository-relative: an
// absolute filesystem path, machine-specific by definition, must never reach the
// sidecar JSON. assertNoAbsolutePath() below enforces that on the way out.
// --------------------------------------------------------------------------- #

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..", "..");
const screensDir = path.join(repoRoot, "data", "synthetic", "screens");
const proceduresDir = path.join(repoRoot, "data", "synthetic", "procedures");
const outputDir = path.join(repoRoot, "data", "synthetic", "screenshots");

const REL_SCREENS = "data/synthetic/screens";
const REL_OUTPUT = "data/synthetic/screenshots";

// --------------------------------------------------------------------------- #
// A narrow reader for the procedure YAML
//
// The repository's YAML parsing lives in Python (services/ingestion). Rather
// than add a Node YAML dependency for four fields, this reads exactly those
// four and fails loudly if the file does not look the way it expects. It is not
// a YAML parser and must not grow into one.
// --------------------------------------------------------------------------- #

function readProcedureFacts(file) {
  const text = readFileSync(file, "utf8");
  const lines = text.split("\n");

  const idMatch = text.match(/^procedure_id:[ \t]*(\S+)[ \t]*$/m);
  if (!idMatch) throw new Error(`${path.basename(file)}: no top-level procedure_id`);

  // Top-level `version:`, anchored at column 0 so nested keys cannot match.
  const versionMatch = text.match(/^version:[ \t]*"?([^"\n]+?)"?[ \t]*$/m);
  if (!versionMatch) throw new Error(`${path.basename(file)}: no top-level version`);

  const stepsStart = lines.findIndex((line) => line === "steps:");
  if (stepsStart === -1) throw new Error(`${path.basename(file)}: no top-level steps: block`);

  const steps = [];
  let current = null;
  for (let i = stepsStart + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^\S/.test(line)) break; // next top-level key ends the block
    const itemMatch = line.match(/^ {2}- number:[ \t]*(\d+)[ \t]*$/);
    if (itemMatch) {
      current = { number: Number(itemMatch[1]) };
      steps.push(current);
      continue;
    }
    if (!current) continue;
    const field = line.match(/^ {4}(screen|ui_element|screenshot|expected_result):[ \t]*(.+?)[ \t]*$/);
    if (field) current[field[1]] = field[2];
  }
  if (steps.length === 0) throw new Error(`${path.basename(file)}: steps: block held no items`);

  return {
    procedureId: idMatch[1],
    version: versionMatch[1],
    steps,
  };
}

/** Cross-check the manifest against the procedure and return the merged facts. */
function stepFactsFor(procedure, screen) {
  const step = procedure.steps.find((item) => item.screenshot === screen.id);
  if (!step) {
    throw new Error(
      `${PROCEDURE_FILE} has no step whose screenshot is ${screen.id}; the generator and the procedure disagree`,
    );
  }
  const mismatches = [];
  if (step.number !== screen.step) mismatches.push(`step ${step.number} != ${screen.step}`);
  if (step.screen !== screen.screen) mismatches.push(`screen "${step.screen}" != "${screen.screen}"`);
  if (step.ui_element !== screen.control) {
    mismatches.push(`ui_element "${step.ui_element}" != "${screen.control}"`);
  }
  if (mismatches.length > 0) {
    throw new Error(`${screen.id}: procedure and generator disagree: ${mismatches.join("; ")}`);
  }
  return {
    procedure_id: procedure.procedureId,
    procedure_version: procedure.version,
    step_order: step.number,
    expected_result: step.expected_result ?? null,
  };
}

// --------------------------------------------------------------------------- #
// PNG decoding with node:zlib only
//
// A perceptual hash needs raw pixels, and adding an image library for that would
// mean a new dependency for ~80 lines of well-specified format. Chromium writes
// 8-bit non-interlaced PNGs, which is the subset handled here; anything else
// throws rather than guessing.
// --------------------------------------------------------------------------- #

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const CHANNELS_BY_COLOR_TYPE = { 0: 1, 2: 3, 4: 2, 6: 4 };

function decodePng(buffer) {
  if (!buffer.subarray(0, 8).equals(PNG_SIGNATURE)) throw new Error("not a PNG");

  let header = null;
  const idat = [];
  let offset = 8;
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("latin1", offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      header = {
        width: data.readUInt32BE(0),
        height: data.readUInt32BE(4),
        bitDepth: data[8],
        colorType: data[9],
        interlace: data[12],
      };
    } else if (type === "IDAT") {
      idat.push(Buffer.from(data));
    } else if (type === "IEND") {
      break;
    }
    offset += 12 + length; // length + type + data + CRC
  }

  if (!header) throw new Error("PNG has no IHDR");
  if (header.bitDepth !== 8) throw new Error(`unsupported PNG bit depth ${header.bitDepth}`);
  if (header.interlace !== 0) throw new Error("interlaced PNG is not supported");
  const channels = CHANNELS_BY_COLOR_TYPE[header.colorType];
  if (!channels) throw new Error(`unsupported PNG colour type ${header.colorType}`);
  if (idat.length === 0) throw new Error("PNG has no IDAT");

  const raw = zlib.inflateSync(Buffer.concat(idat));
  const { width, height } = header;
  const stride = width * channels;
  const pixels = Buffer.alloc(height * stride);

  let read = 0;
  let previous = Buffer.alloc(stride);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[read];
    read += 1;
    const line = raw.subarray(read, read + stride);
    read += stride;
    const row = pixels.subarray(y * stride, (y + 1) * stride);
    line.copy(row);
    unfilterRow(filter, row, previous, channels);
    previous = row;
  }

  return { width, height, channels, pixels };
}

/** PNG filter reversal, RFC 2083 section 6. */
function unfilterRow(filter, row, previous, bpp) {
  const stride = row.length;
  switch (filter) {
    case 0:
      return;
    case 1:
      for (let i = bpp; i < stride; i += 1) row[i] = (row[i] + row[i - bpp]) & 0xff;
      return;
    case 2:
      for (let i = 0; i < stride; i += 1) row[i] = (row[i] + previous[i]) & 0xff;
      return;
    case 3:
      for (let i = 0; i < stride; i += 1) {
        const left = i >= bpp ? row[i - bpp] : 0;
        row[i] = (row[i] + ((left + previous[i]) >> 1)) & 0xff;
      }
      return;
    case 4:
      for (let i = 0; i < stride; i += 1) {
        const a = i >= bpp ? row[i - bpp] : 0;
        const b = previous[i];
        const c = i >= bpp ? previous[i - bpp] : 0;
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        const predictor = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        row[i] = (row[i] + predictor) & 0xff;
      }
      return;
    default:
      throw new Error(`unknown PNG filter type ${filter}`);
  }
}

// --------------------------------------------------------------------------- #
// 64-bit difference hash
//
// The image is reduced to a 9x8 grid of grayscale cell averages, then each cell
// is compared with its right-hand neighbour: 8 rows x 8 comparisons = 64 bits.
// Box-averaging over every pixel (rather than sampling) makes the hash depend on
// layout and content, not on which pixels happened to be sampled.
// --------------------------------------------------------------------------- #

const HASH_COLUMNS = 9;
const HASH_ROWS = 8;

function differenceHash({ width, height, channels, pixels }) {
  const sums = new Float64Array(HASH_COLUMNS * HASH_ROWS);
  const counts = new Uint32Array(HASH_COLUMNS * HASH_ROWS);

  for (let y = 0; y < height; y += 1) {
    const cellY = Math.min(HASH_ROWS - 1, Math.floor((y * HASH_ROWS) / height));
    const rowStart = y * width * channels;
    for (let x = 0; x < width; x += 1) {
      const cellX = Math.min(HASH_COLUMNS - 1, Math.floor((x * HASH_COLUMNS) / width));
      const i = rowStart + x * channels;
      let gray;
      if (channels >= 3) {
        gray = 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
      } else {
        gray = pixels[i];
      }
      const cell = cellY * HASH_COLUMNS + cellX;
      sums[cell] += gray;
      counts[cell] += 1;
    }
  }

  const averages = new Float64Array(HASH_COLUMNS * HASH_ROWS);
  for (let cell = 0; cell < averages.length; cell += 1) {
    averages[cell] = counts[cell] === 0 ? 0 : sums[cell] / counts[cell];
  }

  const bytes = Buffer.alloc(8);
  let bit = 0;
  for (let row = 0; row < HASH_ROWS; row += 1) {
    for (let column = 0; column < HASH_COLUMNS - 1; column += 1) {
      const left = averages[row * HASH_COLUMNS + column];
      const right = averages[row * HASH_COLUMNS + column + 1];
      if (left > right) bytes[bit >> 3] |= 0x80 >> bit % 8;
      bit += 1;
    }
  }
  return bytes.toString("hex");
}

function hammingDistance(hexA, hexB) {
  const a = Buffer.from(hexA, "hex");
  const b = Buffer.from(hexB, "hex");
  if (a.length !== b.length) throw new Error("hash length mismatch");
  let distance = 0;
  for (let i = 0; i < a.length; i += 1) {
    let diff = a[i] ^ b[i];
    while (diff) {
      distance += diff & 1;
      diff >>= 1;
    }
  }
  return distance;
}

// --------------------------------------------------------------------------- #
// Geometry assertions
//
// "The highlight points at the control and covers nothing important" is checked,
// not assumed: the ring has to enclose the control with clearance, and the
// callout and arrow must not intersect the control, the ring, or the client rect
// of any other text on the page.
// --------------------------------------------------------------------------- #

function intersection(a, b) {
  const w = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y);
  return w > 0.5 && h > 0.5 ? { width: w, height: h } : null;
}

async function measureHighlight(page, screenId) {
  const counts = await page.evaluate(() => ({
    targets: document.querySelectorAll("[data-agentdesk-target]").length,
    rings: document.querySelectorAll(".hl-ring").length,
    callouts: document.querySelectorAll(".hl-callout").length,
    arrows: document.querySelectorAll(".hl-arrow").length,
    watermarks: document.querySelectorAll("[data-agentdesk-watermark]").length,
  }));
  for (const [key, expected] of [
    ["targets", 1],
    ["rings", 1],
    ["callouts", 1],
    ["watermarks", 1],
  ]) {
    if (counts[key] !== expected) {
      throw new Error(`${screenId}: expected ${expected} ${key}, found ${counts[key]}`);
    }
  }
  if (counts.arrows < 1) throw new Error(`${screenId}: no highlight arrow`);

  const target = await (await page.locator("[data-agentdesk-target]").elementHandle()).boundingBox();
  const ring = await (await page.locator(".hl-ring").elementHandle()).boundingBox();
  const callout = await (await page.locator(".hl-callout").elementHandle()).boundingBox();
  const arrow = await (await page.locator(".hl-arrow").first().elementHandle()).boundingBox();
  if (!target || !ring || !callout || !arrow) {
    throw new Error(`${screenId}: a highlight element has no bounding box (is it hidden?)`);
  }

  // The ring must sit outside the control on all four sides, so its 3px border
  // is never drawn over the control's own label.
  const clearance = {
    left: target.x - ring.x,
    top: target.y - ring.y,
    right: ring.x + ring.width - (target.x + target.width),
    bottom: ring.y + ring.height - (target.y + target.height),
  };
  for (const [side, value] of Object.entries(clearance)) {
    if (value < 4) {
      throw new Error(
        `${screenId}: highlight ring only clears the control by ${value.toFixed(1)}px on the ${side}; it would sit on the control`,
      );
    }
  }

  // Callout and arrow keep off the control and off the ring.
  for (const [name, rect] of [
    ["callout", callout],
    ["arrow", arrow],
  ]) {
    if (intersection(rect, target)) throw new Error(`${screenId}: the ${name} covers the control`);
    if (intersection(rect, ring)) throw new Error(`${screenId}: the ${name} overlaps the ring`);
    if (
      rect.x < 4 ||
      rect.y < 4 ||
      rect.x + rect.width > VIEWPORT.width - 4 ||
      rect.y + rect.height > VIEWPORT.height - 4
    ) {
      throw new Error(`${screenId}: the ${name} is clipped by the viewport edge`);
    }
  }

  // Nothing else on the page may be hidden behind the callout or the arrow, and
  // the ring band may not sit over any text except the control's own label.
  const collisions = await page.evaluate(() => {
    const target = document.querySelector("[data-agentdesk-target]");
    const overlayRects = [...document.querySelectorAll(".hl-callout, .hl-arrow, .hl-ring")].map(
      (element) => ({
        what: element.classList.contains("hl-ring")
          ? "the highlight ring"
          : element.classList.contains("hl-arrow")
            ? "the arrow"
            : "the callout",
        // The ring encloses the control by design, so its own control's text is
        // the one thing it is allowed to sit around.
        skipTargetText: element.classList.contains("hl-ring"),
        rect: element.getBoundingClientRect(),
      }),
    );
    const insideOverlay = (node) => {
      for (let element = node.parentElement; element; element = element.parentElement) {
        if (
          element.classList.contains("hl-callout") ||
          element.classList.contains("hl-arrow") ||
          element.classList.contains("hl-ring")
        ) {
          return true;
        }
      }
      return false;
    };

    const hits = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent.trim();
      if (text === "" || insideOverlay(node)) continue;
      const isTargetText = target !== null && target.contains(node);
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const rect of range.getClientRects()) {
        if (rect.width === 0 || rect.height === 0) continue;
        for (const overlay of overlayRects) {
          if (overlay.skipTargetText && isTargetText) continue;
          const w =
            Math.min(rect.right, overlay.rect.right) - Math.max(rect.left, overlay.rect.left);
          const h =
            Math.min(rect.bottom, overlay.rect.bottom) - Math.max(rect.top, overlay.rect.top);
          if (w > 0.5 && h > 0.5) {
            hits.push(`${overlay.what} sits over "${text.slice(0, 60)}"`);
          }
        }
      }
    }
    return [...new Set(hits)];
  });
  if (collisions.length > 0) {
    throw new Error(
      `${screenId}: the highlight is not clear of the surrounding text:\n    ${collisions.join("\n    ")}`,
    );
  }

  return {
    control: target,
    ring,
    callout,
    clearance,
  };
}

// --------------------------------------------------------------------------- #
// Rendering
// --------------------------------------------------------------------------- #

function round(value) {
  return Math.round(value * 100) / 100;
}

async function renderScreen(context, screen, facts) {
  const htmlPath = path.join(screensDir, `${screen.id}.html`);
  const page = await context.newPage();

  // Proof of no network dependency. Anything that is not a local file: or data:
  // URL is aborted and recorded, and generation fails if the list is not empty.
  // about: is allowed through without being recorded: about:blank is the initial
  // empty document Playwright starts every page on, not a fetch.
  const offNetwork = [];
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = request.url();
    if (url.startsWith("file:") || url.startsWith("data:") || url.startsWith("about:")) {
      await route.continue();
      return;
    }
    offNetwork.push(`${request.method()} ${url}`);
    await route.abort();
  });

  // Pinned rather than inherited: a dark-mode desktop or a "reduce motion off"
  // setting must not change the capture.
  await page.emulateMedia({
    media: "screen",
    colorScheme: "light",
    reducedMotion: "reduce",
    forcedColors: "none",
  });

  const response = await page.goto(`file://${htmlPath}`, { waitUntil: "load" });
  if (response && !response.ok() && response.status() !== 0) {
    throw new Error(`${screen.id}: loading the HTML failed with status ${response.status()}`);
  }
  // Resolve, then discard: FontFaceSet itself is not serialisable across the
  // protocol, so the promise is mapped to a boolean.
  await page.evaluate(() => document.fonts.ready.then(() => true));

  if (offNetwork.length > 0) {
    throw new Error(
      `${screen.id}: the screen tried to reach off-file resources:\n    ${offNetwork.join("\n    ")}`,
    );
  }

  // A screen that overflows 1440x900 would be silently cropped by the capture.
  const layout = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
  }));
  if (layout.scrollWidth > VIEWPORT.width || layout.scrollHeight > VIEWPORT.height) {
    throw new Error(
      `${screen.id}: content is ${layout.scrollWidth}x${layout.scrollHeight}, larger than the ${VIEWPORT.width}x${VIEWPORT.height} viewport`,
    );
  }

  const highlight = await measureHighlight(page, screen.id);

  // innerText, not textContent: it is the text a sighted agent actually reads,
  // in layout order, with hidden nodes excluded. This is the authoritative text
  // for the screenshot - there is no OCR in this project.
  const domText = await page.evaluate(() => document.body.innerText.trim());
  if (!domText.includes(WATERMARK)) {
    throw new Error(`${screen.id}: the watermark is missing from the rendered DOM text`);
  }

  const png = await page.screenshot({
    type: "png",
    fullPage: false,
    animations: "disabled",
    caret: "hide",
    scale: "css",
  });

  await page.close();

  const decoded = decodePng(png);
  if (decoded.width !== VIEWPORT.width || decoded.height !== VIEWPORT.height) {
    throw new Error(
      `${screen.id}: rendered ${decoded.width}x${decoded.height}, expected ${VIEWPORT.width}x${VIEWPORT.height}`,
    );
  }

  const sidecar = {
    screenshot_id: screen.id,
    screen_name: screen.screen,
    control_name: screen.control,
    procedure_id: facts.procedure_id,
    procedure_version: facts.procedure_version,
    step_order: facts.step_order,
    expected_result: facts.expected_result,
    image_path: `${REL_OUTPUT}/${screen.id}.png`,
    source_html_path: `${REL_SCREENS}/${screen.id}.html`,
    width: decoded.width,
    height: decoded.height,
    device_scale_factor: DEVICE_SCALE_FACTOR,
    bytes: png.length,
    sha256: createHash("sha256").update(png).digest("hex"),
    language: UI_LANGUAGE,
    language_note: UI_LANGUAGE_NOTE,
    watermark: WATERMARK,
    dom_text: domText,
    // No OCR exists in this project and nothing may depend on one. The field is
    // carried so the record shape matches the schema that will hold it, and is
    // explicitly null so a consumer cannot mistake an empty string for "the OCR
    // ran and found nothing". dom_text above is the authoritative text.
    ocr_text: null,
    image_description: screen.description,
    highlight: {
      control_name: screen.control,
      selector: "[data-agentdesk-target]",
      source: "Playwright boundingBox() of the control element",
      units: "CSS pixels, top-left origin, deviceScaleFactor 1",
      x: round(highlight.control.x),
      y: round(highlight.control.y),
      width: round(highlight.control.width),
      height: round(highlight.control.height),
      ring: {
        x: round(highlight.ring.x),
        y: round(highlight.ring.y),
        width: round(highlight.ring.width),
        height: round(highlight.ring.height),
      },
    },
    perceptual_hash: {
      algorithm: "dhash-64",
      grid: `${HASH_COLUMNS}x${HASH_ROWS} grayscale cell averages, horizontal comparison`,
      hex: differenceHash(decoded),
      hamming_tolerance: HAMMING_TOLERANCE,
    },
  };

  return { png, sidecar };
}

// --------------------------------------------------------------------------- #
// Commands
// --------------------------------------------------------------------------- #

function sidecarJson(sidecar) {
  return `${JSON.stringify(sidecar, null, 2)}\n`;
}

function assertNoAbsolutePath(screenId, json) {
  const suspicious = json.match(/"(?:\/[^"]*|[A-Za-z]:\\[^"]*)"/g);
  if (suspicious) {
    throw new Error(`${screenId}: sidecar JSON contains an absolute path: ${suspicious[0]}`);
  }
}

async function withBrowser(run) {
  const resolution = resolveBrowser({ allowPathDiscovery: true });
  if (!resolution.executable) {
    console.error(`\ngenerate-screenshots: FAIL - ${resolution.problem}`);
    for (const hint of resolution.hints ?? []) console.error(`  ${hint}`);
    console.error("");
    process.exit(1);
  }

  console.log("Browser");
  console.log(`  executable : ${resolution.executable}`);
  console.log(`  version    : ${versionOf(resolution.executable)}`);
  console.log(`  because    : ${resolution.reason}`);
  if (resolution.tier === "path") {
    console.log(`  note       : set ${ENV_VAR} to pin this explicitly`);
  }

  const { chromium } = await import("@playwright/test");
  const browser = await chromium.launch({
    executablePath: resolution.executable,
    args: [
      // Rasteriser determinism: sRGB rather than the display profile, grayscale
      // antialiasing rather than display-dependent subpixel LCD filtering, and
      // no font hinting, which otherwise varies with the local fontconfig.
      "--force-color-profile=srgb",
      "--disable-lcd-text",
      "--font-render-hinting=none",
      "--disable-gpu",
    ],
  });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: DEVICE_SCALE_FACTOR,
    colorScheme: "light",
    reducedMotion: "reduce",
    forcedColors: "none",
    locale: "en-US",
    // Nothing in the DOM is derived from the clock, but a pinned zone means a
    // future mistake shows up as a diff rather than as a machine difference.
    timezoneId: "UTC",
    javaScriptEnabled: true,
    offline: true,
  });
  try {
    return await run(context);
  } finally {
    await context.close();
    await browser.close();
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const check = argv.includes("--check");
  const onlyIndex = argv.indexOf("--only");
  const only = onlyIndex === -1 ? null : argv[onlyIndex + 1];

  const screens = only ? SCREENS.filter((screen) => screen.id === only) : SCREENS;
  if (screens.length === 0) {
    throw new Error(`--only ${only}: no such screen. Known: ${SCREENS.map((s) => s.id).join(", ")}`);
  }

  const procedure = readProcedureFacts(path.join(proceduresDir, PROCEDURE_FILE));
  console.log(
    `Procedure ${procedure.procedureId} v${procedure.version} (read from ${PROCEDURE_FILE})`,
  );
  console.log(`Mode: ${check ? "--check (nothing is written to the repository)" : "generate"}\n`);

  const rendered = await withBrowser(async (context) => {
    const results = [];
    for (const screen of screens) {
      const facts = stepFactsFor(procedure, screen);
      const result = await renderScreen(context, screen, facts);
      results.push({ screen, ...result });
    }
    return results;
  });

  if (!check) {
    mkdirSync(outputDir, { recursive: true });
    for (const { screen, png, sidecar } of rendered) {
      const json = sidecarJson(sidecar);
      assertNoAbsolutePath(screen.id, json);
      writeFileSync(path.join(outputDir, `${screen.id}.png`), png);
      writeFileSync(path.join(outputDir, `${screen.id}.json`), json, "utf8");
      console.log(`wrote ${REL_OUTPUT}/${screen.id}.png  ${sidecar.width}x${sidecar.height}, ${sidecar.bytes} bytes`);
      console.log(`      ${REL_OUTPUT}/${screen.id}.json dhash ${sidecar.perceptual_hash.hex}`);
    }
    console.log("\ngenerate-screenshots: OK\n");
    return;
  }

  // --check renders into a temporary directory so a failed comparison cannot
  // leave a half-updated repository behind, then compares against the committed
  // output on the three axes the determinism contract names.
  const temporary = mkdtempSync(path.join(tmpdir(), "agentpath-screenshots-"));
  const failures = [];
  try {
    for (const { screen, png, sidecar } of rendered) {
      const json = sidecarJson(sidecar);
      assertNoAbsolutePath(screen.id, json);
      writeFileSync(path.join(temporary, `${screen.id}.png`), png);
      writeFileSync(path.join(temporary, `${screen.id}.json`), json, "utf8");

      let committed;
      try {
        committed = JSON.parse(readFileSync(path.join(outputDir, `${screen.id}.json`), "utf8"));
      } catch (error) {
        failures.push(`${screen.id}: no committed sidecar to compare against (${error.message})`);
        continue;
      }
      let committedPng;
      try {
        committedPng = readFileSync(path.join(outputDir, `${screen.id}.png`));
      } catch (error) {
        failures.push(`${screen.id}: no committed PNG to compare against (${error.message})`);
        continue;
      }
      const committedDecoded = decodePng(committedPng);

      const problems = [];
      if (
        committedDecoded.width !== sidecar.width ||
        committedDecoded.height !== sidecar.height ||
        committed.width !== sidecar.width ||
        committed.height !== sidecar.height
      ) {
        problems.push(
          `dimensions: committed PNG ${committedDecoded.width}x${committedDecoded.height}, sidecar ${committed.width}x${committed.height}, fresh ${sidecar.width}x${sidecar.height}`,
        );
      }
      if (committed.dom_text !== sidecar.dom_text) {
        problems.push(`dom_text differs (${describeTextDiff(committed.dom_text, sidecar.dom_text)})`);
      }
      if (!sidecar.dom_text.includes(WATERMARK)) {
        problems.push("watermark missing from dom_text");
      }
      const distance = hammingDistance(
        differenceHash(committedDecoded),
        sidecar.perceptual_hash.hex,
      );
      if (distance > HAMMING_TOLERANCE) {
        problems.push(`perceptual hash distance ${distance} > tolerance ${HAMMING_TOLERANCE}`);
      }

      const identicalBytes = committedPng.equals(png);
      console.log(
        `${screen.id}\n` +
          `  dimensions      : ${sidecar.width}x${sidecar.height} (match: ${
            committedDecoded.width === sidecar.width && committedDecoded.height === sidecar.height
          })\n` +
          `  dom_text        : ${committed.dom_text === sidecar.dom_text ? "identical" : "DIFFERENT"} (${sidecar.dom_text.length} chars)\n` +
          `  dhash distance  : ${distance} (tolerance ${HAMMING_TOLERANCE})\n` +
          `  png bytes       : ${identicalBytes ? "identical" : "differ"} (not required)`,
      );
      if (problems.length > 0) failures.push(`${screen.id}: ${problems.join("; ")}`);
    }
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }

  if (failures.length > 0) {
    console.error("\ngenerate-screenshots: FAIL");
    for (const failure of failures) console.error(`  ${failure}`);
    console.error("");
    process.exit(1);
  }
  console.log("\ngenerate-screenshots: OK (--check)\n");
}

function describeTextDiff(a, b) {
  if (a === undefined || a === null) return "committed sidecar has no dom_text";
  const limit = Math.min(a.length, b.length);
  let i = 0;
  while (i < limit && a[i] === b[i]) i += 1;
  return `first difference at character ${i}: committed ${JSON.stringify(a.slice(i, i + 40))} vs fresh ${JSON.stringify(b.slice(i, i + 40))}`;
}

main().catch((error) => {
  console.error(`\ngenerate-screenshots: FAIL - ${error instanceof Error ? error.message : error}`);
  console.error("");
  process.exit(1);
});

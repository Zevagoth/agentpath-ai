import { beforeEach, describe, expect, it } from "vitest";

import {
  clearFeedback,
  readFeedback,
  recordFeedback,
  summariseForFeedback,
} from "@/lib/feedback";
import type { AnswerResponse } from "@/lib/types";

/**
 * The feedback record, and mainly what it refuses to keep.
 *
 * The case text is the field most likely to contain a guest's situation in an
 * agent's own words, which is the category the case-context contract exists to
 * keep out of this system. A test that asserts its absence is the only thing
 * stopping a later "just log the query too" from being invisible.
 */

const RESPONSE = {
  status: "clarification_needed",
  query: "The hotel walked our guest and her name is Dana Haddad, ref XYZ-77421",
  detected_language: "en",
  summary: "Confirm the booking, log the refusal, then relocate.",
  steps: [
    {
      instruction: "Open the booking in AgentDesk.",
      procedure_id: "NW-HTL-014",
      procedure_version: "2.1",
      step_id: "NW-HTL-014#S01",
      citation: "NW-HTL-014 v2.1 · NW-HTL-014#S01",
      source_section: "NW-HTL-014.md > ## Steps > ### Step 1",
      control_name: "Booking status chip",
      conditions: [],
      screenshot: null,
      source: "model",
      requires: [],
    },
    {
      instruction: "Open Relocation Finder and search for equivalent stays.",
      procedure_id: "NW-HTL-014",
      procedure_version: "2.1",
      step_id: "NW-HTL-014#S05",
      citation: "NW-HTL-014 v2.1 · NW-HTL-014#S05",
      source_section: "NW-HTL-014.md > ## Steps > ### Step 5",
      control_name: "Find equivalent stays",
      conditions: [],
      screenshot: null,
      source: "prerequisite",
      requires: [],
    },
  ],
  conditions: [],
  escalation: null,
  customer_wording: null,
  clarifying_questions: ["Is the guest at the property right now?"],
  conflicts: [],
  sources: ["NW-HTL-014 v2.1 (effective 2026-03-01) — Property unable to accommodate"],
  screenshots: [],
  evidence: { level: "partial", reason: "scored 0.55", abstained: false, message: null },
  model_called: true,
  failure_detail: null,
  dropped_citations: [],
  inserted_prerequisites: ["NW-HTL-014#S05"],
  context_used: {
    provided: [],
    applied: {},
    unanswered: [],
    not_collected: {},
    completed_steps: [],
    product_line: null,
  },
  timings_ms: { retrieval: 182.4, model_generation: 6805.2, total: 6991.1 },
  retrieval_metadata: { model: "qwen3.5:4b" },
} as unknown as AnswerResponse;

beforeEach(() => {
  clearFeedback();
});

describe("feedback record", () => {
  it("keeps what makes a verdict traceable", () => {
    const entry = summariseForFeedback(RESPONSE, "not_helpful", new Date("2026-09-15T02:30:45Z"));

    expect(entry.verdict).toBe("not_helpful");
    expect(entry.status).toBe("clarification_needed");
    expect(entry.procedureIds).toEqual(["NW-HTL-014"]);
    expect(entry.stepIds).toEqual(["NW-HTL-014#S01", "NW-HTL-014#S05"]);
    expect(entry.insertedPrerequisites).toEqual(["NW-HTL-014#S05"]);
    expect(entry.modelMs).toBe(6805);
    expect(entry.totalMs).toBe(6991);
    expect(entry.model).toBe("qwen3.5:4b");
    expect(entry.at).toBe("2026-09-15T02:30:45Z");
  });

  it("stores the shape of the query and none of its words", () => {
    const entry = summariseForFeedback(RESPONSE, "helpful");
    const serialised = JSON.stringify(entry);

    expect(entry.queryLength).toBe(RESPONSE.query.length);
    expect(entry.language).toBe("en");

    // Nothing from the case text, including the name and reference it contained.
    expect(serialised).not.toContain("Dana");
    expect(serialised).not.toContain("XYZ-77421");
    expect(serialised).not.toContain("walked our guest");
    expect(Object.keys(entry)).not.toContain("query");
  });

  it("persists newest first", () => {
    recordFeedback(summariseForFeedback(RESPONSE, "helpful"));
    const stored = recordFeedback(summariseForFeedback(RESPONSE, "not_helpful"));

    expect(stored).toHaveLength(2);
    expect(stored[0].verdict).toBe("not_helpful");
    expect(readFeedback()).toHaveLength(2);
  });

  it("nothing written to storage contains the case text", () => {
    recordFeedback(summariseForFeedback(RESPONSE, "helpful"));
    const raw = window.localStorage.getItem("agentpath.feedback.v1") ?? "";

    expect(raw.length).toBeGreaterThan(0);
    expect(raw).not.toContain("Dana");
    expect(raw).not.toContain("XYZ-77421");
  });

  it("caps the history rather than growing without bound", () => {
    for (let index = 0; index < 60; index += 1) {
      recordFeedback(summariseForFeedback(RESPONSE, "helpful"));
    }
    expect(readFeedback()).toHaveLength(50);
  });

  it("survives unreadable storage", () => {
    window.localStorage.setItem("agentpath.feedback.v1", "{not json");
    expect(readFeedback()).toEqual([]);
  });

  it("degrades quietly when the browser has no storage at all", () => {
    // Private-mode Safari and some managed profiles either hide `localStorage` or
    // throw on access. Feedback is not important enough to break the answer view,
    // so the verdict is reflected in the UI and simply not persisted.
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new Error("access denied");
      },
      configurable: true,
    });

    try {
      expect(readFeedback()).toEqual([]);
      expect(() => recordFeedback(summariseForFeedback(RESPONSE, "helpful"))).not.toThrow();
      expect(recordFeedback(summariseForFeedback(RESPONSE, "helpful"))).toHaveLength(1);
    } finally {
      if (original) Object.defineProperty(window, "localStorage", original);
    }
  });
});

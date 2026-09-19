/**
 * Helpful / Not helpful, stored in this browser only.
 *
 * Two constraints shape this.
 *
 * **It does not leave the machine.** There is no endpoint to send it to, and
 * inventing one would be a new data flow nobody asked for. `localStorage` keeps
 * it where the agent who wrote it can see it, and `make ragflow-down` or a cleared
 * profile removes it.
 *
 * **It does not store the case.** The case text is the one field most likely to
 * contain a guest's situation in a colleague's own words, which is exactly the
 * category the case-context contract refuses to hold. So the record keeps what
 * makes the feedback actionable - which procedure, which steps, which status, how
 * long it took - plus the *shape* of the query (length, language) and none of its
 * words. A verdict tied to `NW-HTL-014#S05..S10 / clarification_needed / 11.2 s`
 * is a usable bug report; the sentence the agent typed adds nothing to it that is
 * worth retaining.
 */

import type { AnswerResponse, AnswerStatus, Language } from "./types";

const STORAGE_KEY = "agentpath.feedback.v1";

/** Old entries are dropped rather than grown without bound. */
const MAX_ENTRIES = 50;

export type Verdict = "helpful" | "not_helpful";

export interface FeedbackEntry {
  /** ISO-8601, second precision. */
  at: string;
  verdict: Verdict;
  status: AnswerStatus;
  /** Characters typed. Deliberately not the text itself. */
  queryLength: number;
  language: Language;
  procedureIds: string[];
  stepIds: string[];
  /** Steps the server had to insert, which is a useful signal about an answer. */
  insertedPrerequisites: string[];
  totalMs: number;
  modelMs: number;
  model: string | null;
}

export function summariseForFeedback(
  response: AnswerResponse,
  verdict: Verdict,
  now: Date = new Date(),
): FeedbackEntry {
  const model = response.retrieval_metadata?.model;
  return {
    at: `${now.toISOString().slice(0, 19)}Z`,
    verdict,
    status: response.status,
    queryLength: response.query.length,
    language: response.detected_language,
    procedureIds: [...new Set(response.steps.map((step) => step.procedure_id))],
    stepIds: response.steps.map((step) => step.step_id),
    insertedPrerequisites: [...response.inserted_prerequisites],
    totalMs: Math.round(response.timings_ms.total ?? 0),
    modelMs: Math.round(response.timings_ms.model_generation ?? 0),
    model: typeof model === "string" ? model : null,
  };
}

/** Guarded so the module is safe to import during server rendering. */
function storage(): Storage | null {
  try {
    if (typeof window === "undefined" || !window.localStorage) return null;
    return window.localStorage;
  } catch {
    // Private-mode Safari and some enterprise policies throw on access rather
    // than returning null. Feedback is not important enough to break the page.
    return null;
  }
}

export function readFeedback(): FeedbackEntry[] {
  const store = storage();
  if (!store) return [];
  try {
    const raw = store.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as FeedbackEntry[]) : [];
  } catch {
    return [];
  }
}

/** Append one verdict. Returns the stored list, newest first. */
export function recordFeedback(entry: FeedbackEntry): FeedbackEntry[] {
  const next = [entry, ...readFeedback()].slice(0, MAX_ENTRIES);
  const store = storage();
  if (store) {
    try {
      store.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Quota exceeded. The verdict is still reflected in the UI for this
      // session; losing the history is not worth an error state.
    }
  }
  return next;
}

export function clearFeedback(): void {
  storage()?.removeItem(STORAGE_KEY);
}

"use client";

import { useState } from "react";

import { recordFeedback, summariseForFeedback, type Verdict } from "@/lib/feedback";
import type { AnswerResponse } from "@/lib/types";

/**
 * Helpful / Not helpful.
 *
 * Kept honest about where it goes: the note under the buttons says the verdict
 * stays in this browser and that the case text is not part of it. A feedback
 * control that silently ships the agent's typing somewhere would undo the reason
 * the case-context contract exists.
 *
 * One verdict per answer. Clicking the other button changes it rather than
 * recording a second opinion about the same answer.
 */

export function FeedbackBar({ response }: { response: AnswerResponse }) {
  const [verdict, setVerdict] = useState<Verdict | null>(null);

  function choose(next: Verdict) {
    setVerdict(next);
    recordFeedback(summariseForFeedback(response, next));
  }

  return (
    <div className="feedback" data-testid="feedback">
      <span className="feedback-label">Was this answer usable?</span>

      <div className="feedback-buttons">
        <button
          type="button"
          className={`feedback-button${verdict === "helpful" ? " feedback-chosen" : ""}`}
          data-testid="feedback-helpful"
          aria-pressed={verdict === "helpful"}
          onClick={() => choose("helpful")}
        >
          Helpful
        </button>
        <button
          type="button"
          className={`feedback-button${verdict === "not_helpful" ? " feedback-chosen feedback-negative" : ""}`}
          data-testid="feedback-not-helpful"
          aria-pressed={verdict === "not_helpful"}
          onClick={() => choose("not_helpful")}
        >
          Not helpful
        </button>
      </div>

      <p className="feedback-note" data-testid="feedback-note" aria-live="polite">
        {verdict === null
          ? "Stored in this browser only. The case text is not recorded — just the procedure, the steps and the timing."
          : verdict === "helpful"
            ? "Recorded as helpful, in this browser only."
            : "Recorded as not helpful, in this browser only. The cited steps and timings were kept so the answer can be traced."}
      </p>
    </div>
  );
}

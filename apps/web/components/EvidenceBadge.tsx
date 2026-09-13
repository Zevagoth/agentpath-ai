import type { EvidenceLevel } from "@/lib/types";

/**
 * Confidence indicator.
 *
 * Shows a level and a word, never a percentage. The retrieval scores are not
 * calibrated probabilities, so "87% confident" would communicate a precision the
 * system does not have. The wording is chosen so that the best case still reads as
 * "check the source", which is the behaviour we want from an agent.
 */

const LABELS: Record<EvidenceLevel, string> = {
  strong: "Strong match",
  partial: "Verify before acting",
  insufficient: "No confirmed procedure",
};

export function EvidenceBadge({ level }: { level: EvidenceLevel }) {
  return (
    <span
      className={`evidence-badge evidence-${level}`}
      data-testid="evidence-badge"
      data-level={level}
    >
      <span className="evidence-dot" aria-hidden="true" />
      {LABELS[level]}
    </span>
  );
}

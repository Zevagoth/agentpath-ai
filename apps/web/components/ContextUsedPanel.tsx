import type { ContextUsed } from "@/lib/types";

/**
 * What the case facts actually changed.
 *
 * Supplying context and receiving an answer proves nothing about whether the
 * context was used, so the API reports its own effect and this renders it: which
 * field narrowed the search, which one stopped a question being asked, what the
 * procedure still needs, and what it needs that we decline to ask for.
 *
 * `not_collected` is the row worth reading. Those are requirements the procedure
 * genuinely has and that this system refuses to hold, so the agent is never
 * prompted for them. Showing the reason turns a silently missing requirement into
 * a visible, deliberate decision.
 */

export function ContextUsedPanel({ context }: { context: ContextUsed }) {
  const appliedEntries = Object.entries(context.applied);
  const declinedEntries = Object.entries(context.not_collected);

  const hasAnything =
    appliedEntries.length > 0 ||
    context.unanswered.length > 0 ||
    declinedEntries.length > 0 ||
    context.completed_steps.length > 0;

  if (!hasAnything) return null;

  return (
    <section className="context-used" data-testid="context-used" aria-label="How the case facts were used">
      <h3 className="section-heading">Case facts used</h3>

      {appliedEntries.length > 0 ? (
        <dl className="applied-list">
          {appliedEntries.map(([field, effect]) => (
            <div key={field} className="applied-row" data-testid="context-applied">
              <dt className="applied-field">{field.replace(/_/g, " ")}</dt>
              <dd className="applied-effect">{effect}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="panel-body" data-testid="context-none">
          No case facts were supplied, so the answer used the case description alone.
        </p>
      )}

      {context.completed_steps.length > 0 && (
        <p className="context-line" data-testid="context-completed">
          <span className="context-line-label">Already done</span>
          {context.completed_steps.join(", ")}
        </p>
      )}

      {context.unanswered.length > 0 && (
        <div className="context-block" data-testid="context-unanswered">
          <p className="context-line-label">Still needed for this procedure</p>
          <ul className="context-items">
            {context.unanswered.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {declinedEntries.length > 0 && (
        <div className="context-block context-block-declined" data-testid="context-not-collected">
          <p className="context-line-label">
            Required by the procedure, and never requested by this system
          </p>
          <ul className="context-items">
            {declinedEntries.map(([requirement, reason]) => (
              <li key={requirement}>
                <span className="declined-item">{requirement}</span>
                <span className="declined-reason">{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

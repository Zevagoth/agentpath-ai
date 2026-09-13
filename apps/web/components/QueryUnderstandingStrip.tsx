import type { QueryUnderstanding } from "@/lib/types";

/**
 * What the system understood, shown back to the agent.
 *
 * This exists because retrieval that silently rewrites a query is impossible to
 * trust or debug. If "overboking" became "overbooking", the agent should see that
 * it did - and see it was not something else.
 */

const LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic",
  en: "English",
  mixed: "Arabic + English",
  unknown: "undetermined",
};

export function QueryUnderstandingStrip({ query }: { query: QueryUnderstanding }) {
  const hasAnything =
    query.corrections.length > 0 ||
    query.expansions.length > 0 ||
    query.scenario_tags.length > 0 ||
    query.detected_product_line !== null;

  if (!hasAnything) return null;

  return (
    <section className="understanding" aria-label="How the query was interpreted">
      <div className="understanding-group">
        <span className="understanding-label">Language</span>
        <span className="tag" data-testid="detected-language">
          {LANGUAGE_LABELS[query.language] ?? query.language}
        </span>
      </div>

      {query.detected_product_line && (
        <div className="understanding-group">
          <span className="understanding-label">Product line</span>
          <span className="tag" data-testid="detected-product-line">
            {query.detected_product_line}
            {query.requested_product_line ? " (filter applied)" : " (detected)"}
          </span>
        </div>
      )}

      {query.corrections.length > 0 && (
        <div className="understanding-group">
          <span className="understanding-label">Spelling</span>
          {query.corrections.map((correction) => (
            <span
              key={`${correction.before}-${correction.after}`}
              className="tag tag-correction tag-mono"
              data-testid="correction"
            >
              {correction.before} → {correction.after}
            </span>
          ))}
        </div>
      )}

      {query.expansions.length > 0 && (
        <div className="understanding-group">
          <span className="understanding-label">Also searched</span>
          {query.expansions.map((term) => (
            <span key={term} className="tag tag-expansion" data-testid="expansion">
              {term}
            </span>
          ))}
        </div>
      )}

      {query.scenario_tags.length > 0 && (
        <div className="understanding-group">
          <span className="understanding-label">Scenario</span>
          {query.scenario_tags.map((tag) => (
            <span key={tag} className="tag" data-testid="scenario-tag">
              {tag.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

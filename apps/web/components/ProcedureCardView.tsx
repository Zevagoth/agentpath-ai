import type { ProcedureCard } from "@/lib/types";

import { HighlightedSnippet } from "./HighlightedSnippet";

/**
 * One retrieved procedure with its citations.
 *
 * The citations are not an optional detail: an agent has to be able to read the
 * source text that caused this procedure to surface, in the source's own
 * direction, before acting on it. The engine's `<em>` highlights are parsed into
 * React nodes rather than injected as HTML - see `lib/highlight.ts`.
 */

export function ProcedureCardView({ procedure }: { procedure: ProcedureCard }) {
  const isPrimary = procedure.rank === 1;

  return (
    <article
      className={`procedure${isPrimary ? " procedure-primary" : ""}`}
      data-testid="procedure-card"
      data-procedure-id={procedure.procedure_id}
      data-rank={procedure.rank}
    >
      <div className="procedure-head">
        <div style={{ display: "flex", gap: 11, alignItems: "flex-start" }}>
          <span className="procedure-rank" aria-label={`Rank ${procedure.rank}`}>
            {procedure.rank}
          </span>
          <div>
            <h3 className="procedure-title">{procedure.title}</h3>
            <div className="procedure-meta">
              <span className="procedure-id" data-testid="procedure-id">
                {procedure.procedure_id}
              </span>
              {procedure.product_line && <span>{procedure.product_line}</span>}
              {procedure.version && <span>version {procedure.version}</span>}
              {procedure.effective_date && <span>effective {procedure.effective_date}</span>}
              <span>
                {procedure.matched_sections} matching{" "}
                {procedure.matched_sections === 1 ? "section" : "sections"}
              </span>
            </div>
          </div>
        </div>
        <span className="score" title="Blended term and vector similarity of the best matching section">
          score {procedure.score.toFixed(3)}
        </span>
      </div>

      {procedure.citations.length > 0 && (
        <div className="citations">
          <p className="citations-title">Source passages</p>
          {procedure.citations.slice(0, 2).map((citation) => (
            <div key={citation.chunk_id} className="citation">
              <HighlightedSnippet className="citation-text" snippet={citation.snippet} />
              <p className="citation-scores">
                score {citation.score.toFixed(3)}
                {citation.lexical_score !== null && ` · term ${citation.lexical_score.toFixed(3)}`}
                {citation.semantic_score !== null && ` · vector ${citation.semantic_score.toFixed(3)}`}
              </p>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}


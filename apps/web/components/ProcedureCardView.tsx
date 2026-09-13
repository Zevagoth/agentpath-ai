import { directionFor } from "@/lib/direction";
import type { ProcedureCard } from "@/lib/types";

/**
 * One retrieved procedure with its citations.
 *
 * The citations are not an optional detail: an agent has to be able to read the
 * source text that caused this procedure to surface, in the source's own
 * direction, before acting on it. Snippets carry the engine's `<em>` highlights,
 * inserted as HTML - safe here because the content is our own synthetic corpus and
 * the markup comes from RAGFlow's highlighter, not from user input.
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
              <p
                className="citation-text"
                dir={directionFor(citation.snippet)}
                // eslint-disable-next-line react/no-danger
                dangerouslySetInnerHTML={{ __html: clampSnippet(citation.snippet) }}
              />
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

/** Keep a snippet readable without cutting an `<em>` tag in half. */
function clampSnippet(snippet: string, maxLength = 420): string {
  if (snippet.length <= maxLength) return snippet;
  const clipped = snippet.slice(0, maxLength);
  const lastOpen = clipped.lastIndexOf("<em>");
  const lastClose = clipped.lastIndexOf("</em>");
  const safe = lastOpen > lastClose ? clipped.slice(0, lastOpen) : clipped;
  return `${safe.trimEnd()}…`;
}

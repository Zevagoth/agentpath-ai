import type { SearchResponse } from "@/lib/types";

import { EvidenceBadge } from "./EvidenceBadge";
import { ProcedureCardView } from "./ProcedureCardView";

/**
 * Ranked procedures, for the search mode.
 *
 * Search answers a narrower question than the assistant does - "which procedure
 * covers this?" - and it is kept because it answers that question without loading
 * a model, so it stays fast and it works when generation is down.
 */

export function SearchResultsView({
  response,
  elapsedMs,
}: {
  response: SearchResponse;
  elapsedMs: number;
}) {
  const { evidence, procedures, conflicts, timings_ms } = response;

  return (
    <>
      <div className="results-head">
        <h2>
          {evidence.abstained
            ? "No procedure recommended"
            : `Top ${procedures.length} ${procedures.length === 1 ? "procedure" : "procedures"}`}
        </h2>
        <div className="head-aside">
          <EvidenceBadge level={evidence.level} />
          <span className="timing" data-testid="timing">
            {Math.round(elapsedMs)} ms round trip · {Math.round(timings_ms.total ?? 0)} ms server
          </span>
        </div>
      </div>

      {evidence.abstained ? (
        <section className="panel panel-abstain" data-testid="abstention">
          <p className="panel-title" data-testid="abstention-message">
            {evidence.message ??
              "No confirmed procedure was found in the available knowledge base."}
          </p>
          <p className="panel-body">{evidence.reason}</p>
          <p className="panel-body" style={{ marginTop: 9 }}>
            Rephrase the case with more detail, or escalate: nothing in this knowledge
            base covers it.
          </p>
        </section>
      ) : (
        <>
          <p className="evidence-reason" data-testid="evidence-reason">
            {evidence.reason}
          </p>
          {procedures.map((procedure) => (
            <ProcedureCardView key={procedure.procedure_id} procedure={procedure} />
          ))}
          {conflicts.length > 0 && (
            <section className="conflicts" data-testid="conflicts">
              <h3>Procedures that may overlap</h3>
              <ul>
                {conflicts.map((conflict) => (
                  <li key={conflict}>{conflict}</li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </>
  );
}

"use client";

import { useRef, useState } from "react";

import { SearchError, search } from "@/lib/api";
import { directionFor } from "@/lib/direction";
import type { ProductLine, SearchResponse } from "@/lib/types";

import { EvidenceBadge } from "./EvidenceBadge";
import { ProcedureCardView } from "./ProcedureCardView";
import { QueryUnderstandingStrip } from "./QueryUnderstandingStrip";

/**
 * The agent console.
 *
 * Four states, each visually distinct because confusing them has real cost:
 * idle, loading, error (the system failed), and abstention (the system worked and
 * the knowledge base has no answer). The last two are the pair that must never
 * look alike.
 */

const PRODUCT_LINES: ProductLine[] = [
  "hotels",
  "flights",
  "cancellations",
  "refunds",
  "escalations",
];

const EXAMPLES = [
  "hotel has no room for the guest",
  "الفندق ما عنده غرف والضيف واصل",
  "overboking at the hotle",
  "nonref cxl request",
];

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: SearchError }
  | { status: "done"; response: SearchResponse; elapsedMs: number };

export function SearchConsole() {
  const [caseText, setCaseText] = useState("");
  const [productLine, setProductLine] = useState<ProductLine | "">("");
  const [state, setState] = useState<State>({ status: "idle" });
  const inFlight = useRef<AbortController | null>(null);

  async function runSearch(text: string, line: ProductLine | "") {
    const trimmed = text.trim();
    if (!trimmed) return;

    // A fast typist can outrun the API; the newest query always wins.
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setState({ status: "loading" });
    const startedAt = performance.now();

    try {
      const response = await search(
        { query: trimmed, product_line: line || null, limit: 3 },
        { signal: controller.signal },
      );
      setState({
        status: "done",
        response,
        elapsedMs: performance.now() - startedAt,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setState({
        status: "error",
        error:
          error instanceof SearchError
            ? error
            : new SearchError({ kind: "unknown", message: "Something went wrong." }),
      });
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              AP
            </span>
            <h1>AgentPath AI</h1>
          </div>
          <p className="tagline">Describe the case. Find the right path.</p>
        </div>
        <span className="synthetic-note">Fictional — synthetic demo data</span>
      </header>

      <form
        className="search-card"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch(caseText, productLine);
        }}
      >
        <label className="field-label" htmlFor="case">
          Describe the customer&apos;s case.
        </label>
        <textarea
          id="case"
          className="case-input"
          data-testid="case-input"
          value={caseText}
          dir={directionFor(caseText)}
          placeholder="e.g. the hotel says it has no room and the guest is already at reception"
          onChange={(event) => setCaseText(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              void runSearch(caseText, productLine);
            }
          }}
        />

        <div className="form-row">
          <div>
            <label className="field-label" htmlFor="product-line">
              Product line (optional)
            </label>
            <select
              id="product-line"
              className="select"
              data-testid="product-line"
              value={productLine}
              onChange={(event) => setProductLine(event.target.value as ProductLine | "")}
            >
              <option value="">All product lines</option>
              {PRODUCT_LINES.map((line) => (
                <option key={line} value={line}>
                  {line}
                </option>
              ))}
            </select>
          </div>
          <button
            className="button"
            type="submit"
            data-testid="search-button"
            disabled={state.status === "loading" || caseText.trim().length === 0}
          >
            {state.status === "loading" ? "Searching…" : "Find procedures"}
          </button>
        </div>

        <p className="hint">
          Arabic and English both work, and misspellings are corrected before searching.
          Ctrl/⌘ + Enter to search.
        </p>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-chip"
              dir={directionFor(example)}
              onClick={() => {
                setCaseText(example);
                void runSearch(example, productLine);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </form>

      {state.status === "loading" && <LoadingPanel />}
      {state.status === "error" && <ErrorPanel error={state.error} />}
      {state.status === "done" && <Results response={state.response} elapsedMs={state.elapsedMs} />}

      <footer className="footer">
        Every procedure here is invented for demonstration. AgentPath AI has no
        affiliation with any travel company, and the knowledge base contains no real
        policy. Retrieval runs locally through RAGFlow; nothing is sent to a
        third-party service.
      </footer>
    </div>
  );
}

function LoadingPanel() {
  return (
    <section className="panel" data-testid="loading" aria-live="polite">
      <p className="panel-title">Searching the knowledge base…</p>
      <div className="skeleton-row" style={{ width: "72%" }} />
      <div className="skeleton-row" style={{ width: "94%" }} />
      <div className="skeleton-row" style={{ width: "58%" }} />
    </section>
  );
}

function ErrorPanel({ error }: { error: SearchError }) {
  return (
    <section className="panel panel-error" data-testid="error" role="alert">
      <p className="panel-title">Search could not run</p>
      <p className="panel-body">{error.message}</p>
      {error.detail && <p className="panel-detail">{error.detail}</p>}
      {error.kind === "engine_unavailable" && (
        <p className="panel-body" style={{ marginTop: 9 }}>
          Nothing was searched, so this says nothing about whether a procedure exists.
        </p>
      )}
    </section>
  );
}

function Results({ response, elapsedMs }: { response: SearchResponse; elapsedMs: number }) {
  const { evidence, procedures, query, conflicts, timings_ms } = response;

  return (
    <>
      <QueryUnderstandingStrip query={query} />

      <div className="results-head">
        <h2>
          {evidence.abstained
            ? "No procedure recommended"
            : `Top ${procedures.length} ${procedures.length === 1 ? "procedure" : "procedures"}`}
        </h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <EvidenceBadge level={evidence.level} />
          <span className="timing" data-testid="timing">
            {Math.round(elapsedMs)} ms round trip · {Math.round(timings_ms.total ?? 0)} ms server
          </span>
        </div>
      </div>

      {evidence.abstained ? (
        <section className="panel panel-abstain" data-testid="abstention">
          <p className="panel-title" data-testid="abstention-message">
            {evidence.message ?? "No confirmed procedure was found in the available knowledge base."}
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

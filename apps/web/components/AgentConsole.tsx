"use client";

import { useRef, useState } from "react";

import { SearchError, answer, search } from "@/lib/api";
import { directionFor } from "@/lib/direction";
import type {
  AnswerResponse,
  CaseContextInput,
  ProductLine,
  SearchResponse,
} from "@/lib/types";

import { AnswerView } from "./AnswerView";
import { CaseContextForm, isEmptyContext } from "./CaseContextForm";
import { QueryUnderstandingStrip } from "./QueryUnderstandingStrip";
import { SearchResultsView } from "./SearchResultsView";

/**
 * The agent console.
 *
 * Two modes over one form, because they answer two different questions and the
 * cheaper one should not disappear behind the expensive one:
 *
 * - **Answer** calls `POST /api/answer`. Grounded steps with citations, which
 *   means a model call and several seconds.
 * - **Find procedures** calls `POST /api/search`. Which procedure covers this,
 *   in a couple of hundred milliseconds and with no model involved - so it still
 *   works when generation is down.
 *
 * Answer is the default: it is what an agent mid-call actually needs.
 *
 * Every outcome the agent can see is a distinct visual state, and the ones that
 * must never be confused are the refusals. `error` here means the request never
 * produced a result - the API is unreachable, or this deployment has no answer
 * model. An abstention and a failed generation are *results*, arrive as HTTP 200,
 * and are rendered by `AnswerView` from the `status` field.
 */

const PRODUCT_LINES: ProductLine[] = [
  "hotels",
  "flights",
  "cancellations",
  "refunds",
  "escalations",
];

const EXAMPLES = [
  "The hotel says they have no room and my guest is standing at reception. What do I do?",
  "كيف أنقل الضيف لمكان إقامة بديل وأسجل الحجز الجديد؟",
  "overboking at the hotle",
  "how do I reset my email password",
];

type Mode = "answer" | "search";

type State =
  | { status: "idle" }
  | { status: "loading"; mode: Mode }
  | { status: "error"; error: SearchError }
  | { status: "answered"; response: AnswerResponse; elapsedMs: number }
  | { status: "searched"; response: SearchResponse; elapsedMs: number };

export function AgentConsole({ initialMode = "answer" }: { initialMode?: Mode } = {}) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [caseText, setCaseText] = useState("");
  const [productLine, setProductLine] = useState<ProductLine | "">("");
  const [caseContext, setCaseContext] = useState<CaseContextInput>({});
  const [state, setState] = useState<State>({ status: "idle" });
  const inFlight = useRef<AbortController | null>(null);

  const busy = state.status === "loading";

  async function run(text: string, line: ProductLine | "", nextMode: Mode) {
    const trimmed = text.trim();
    if (!trimmed) return;

    // A fast typist can outrun the API; the newest request always wins.
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setState({ status: "loading", mode: nextMode });
    const startedAt = performance.now();

    try {
      if (nextMode === "answer") {
        const response = await answer(
          {
            query: trimmed,
            product_line: line || null,
            limit: 3,
            case_context: isEmptyContext(caseContext) ? null : caseContext,
          },
          { signal: controller.signal },
        );
        setState({ status: "answered", response, elapsedMs: performance.now() - startedAt });
      } else {
        const response = await search(
          { query: trimmed, product_line: line || null, limit: 3 },
          { signal: controller.signal },
        );
        setState({ status: "searched", response, elapsedMs: performance.now() - startedAt });
      }
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
          <p className="tagline">Describe the case. Get the procedure, step by step.</p>
        </div>
        <span className="synthetic-note">Fictional — synthetic demo data</span>
      </header>

      <form
        className="search-card"
        onSubmit={(event) => {
          event.preventDefault();
          void run(caseText, productLine, mode);
        }}
      >
        <div className="mode-switch" role="group" aria-label="What to do with the case">
          <button
            type="button"
            className={`mode-button${mode === "answer" ? " mode-active" : ""}`}
            data-testid="mode-answer"
            aria-pressed={mode === "answer"}
            onClick={() => setMode("answer")}
          >
            Answer the case
          </button>
          <button
            type="button"
            className={`mode-button${mode === "search" ? " mode-active" : ""}`}
            data-testid="mode-search"
            aria-pressed={mode === "search"}
            onClick={() => setMode("search")}
          >
            Find procedures
          </button>
          <span className="mode-hint">
            {mode === "answer"
              ? "Cited steps from the procedure. Takes a few seconds — it runs a local model."
              : "Ranked procedures only. Fast, and works without the answer model."}
          </span>
        </div>

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
              void run(caseText, productLine, mode);
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
            disabled={busy || caseText.trim().length === 0}
          >
            {busy
              ? mode === "answer"
                ? "Answering…"
                : "Searching…"
              : mode === "answer"
                ? "Answer this case"
                : "Find procedures"}
          </button>
        </div>

        {mode === "answer" && (
          <CaseContextForm value={caseContext} onChange={setCaseContext} disabled={busy} />
        )}

        <p className="hint">
          Arabic and English both work, and misspellings are corrected before searching.
          Ctrl/⌘ + Enter to submit.
        </p>
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="example-chip"
              dir={directionFor(example)}
              disabled={busy}
              onClick={() => {
                setCaseText(example);
                void run(example, productLine, mode);
              }}
            >
              {example.length > 52 ? `${example.slice(0, 52)}…` : example}
            </button>
          ))}
        </div>
      </form>

      {state.status === "loading" && <LoadingPanel mode={state.mode} />}
      {state.status === "error" && <ErrorPanel error={state.error} />}
      {state.status === "answered" && (
        <AnswerView response={state.response} roundTripMs={state.elapsedMs} />
      )}
      {state.status === "searched" && (
        <>
          <QueryUnderstandingStrip query={state.response.query} />
          <SearchResultsView response={state.response} elapsedMs={state.elapsedMs} />
        </>
      )}

      <footer className="footer">
        Every procedure here is invented for demonstration. AgentPath AI has no
        affiliation with any travel company, and the knowledge base contains no real
        policy. Retrieval and answer generation both run locally — nothing is sent to
        a third-party service.
      </footer>
    </div>
  );
}

/**
 * Loading, with the stages named.
 *
 * Answer generation takes seconds rather than milliseconds, so a spinner with no
 * explanation reads as a hang. Naming the two stages also sets the expectation
 * that the wait is the model writing, which is where the time actually goes.
 */
function LoadingPanel({ mode }: { mode: Mode }) {
  return (
    <section className="panel" data-testid="loading" aria-live="polite">
      <p className="panel-title">
        {mode === "answer" ? "Building a grounded answer…" : "Searching the knowledge base…"}
      </p>
      {mode === "answer" && (
        <ol className="progress" data-testid="generation-progress">
          <li>Searching the procedures</li>
          <li>Rebuilding each matched step from the source</li>
          <li>Writing the answer with a local model — this is the slow part</li>
          <li>Checking every citation against the procedure</li>
        </ol>
      )}
      <div className="skeleton-row" style={{ width: "72%" }} />
      <div className="skeleton-row" style={{ width: "94%" }} />
      <div className="skeleton-row" style={{ width: "58%" }} />
    </section>
  );
}

/**
 * The request never ran.
 *
 * Distinct from every state in `AnswerView`: this is the API being unreachable or
 * unconfigured, which says nothing at all about whether a procedure exists.
 */
function ErrorPanel({ error }: { error: SearchError }) {
  return (
    <section className="panel panel-error" data-testid="error" role="alert">
      <p className="panel-title">
        {error.kind === "generation_unavailable"
          ? "Answer generation is not available"
          : "The request could not run"}
      </p>
      <p className="panel-body">{error.message}</p>
      {error.detail && <p className="panel-detail">{error.detail}</p>}

      {error.kind === "engine_unavailable" && (
        <p className="panel-body" style={{ marginTop: 9 }}>
          Nothing was searched, so this says nothing about whether a procedure exists.
        </p>
      )}
      {error.kind === "generation_unavailable" && (
        <p className="panel-body" style={{ marginTop: 9 }}>
          Switch to <strong>Find procedures</strong> to locate the right procedure and
          read it directly — search does not need the answer model.
        </p>
      )}
    </section>
  );
}

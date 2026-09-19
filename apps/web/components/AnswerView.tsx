"use client";

import { useState } from "react";

import { directionFor } from "@/lib/direction";
import type { AnswerResponse } from "@/lib/types";

import { AnswerStepRow } from "./AnswerStepRow";
import { ContextUsedPanel } from "./ContextUsedPanel";
import { EvidenceBadge } from "./EvidenceBadge";
import { FeedbackBar } from "./FeedbackBar";

/**
 * One answer, in whichever of its five states it arrived.
 *
 * The states are rendered as genuinely different things rather than as one layout
 * with fields missing, because three of them are refusals with different remedies
 * and an agent who cannot tell them apart learns to distrust all three:
 *
 * - `answered` / `clarification_needed` - there is an answer. The second one also
 *   carries a question, and the steps are still shown: the steps are valid, it is
 *   the case that is underdetermined.
 * - `abstained` - we searched and the knowledge base has no answer. The model was
 *   never called, and the panel says so, because "the AI didn't answer" and "there
 *   is no procedure for this" are different facts.
 * - `generation_failed` - the evidence was there and the model server was not.
 *   This is an outage to report, not a statement about the corpus.
 * - `empty_query` - nothing was asked, so nothing was searched.
 */

const STATUS_HEADINGS: Record<string, string> = {
  answered: "What to do",
  clarification_needed: "What to do, once one thing is confirmed",
};

export function AnswerView({
  response,
  roundTripMs,
}: {
  response: AnswerResponse;
  roundTripMs: number;
}) {
  if (response.status === "abstained") {
    return <AbstentionPanel response={response} />;
  }
  if (response.status === "generation_failed") {
    return <GenerationFailedPanel response={response} />;
  }
  if (response.status === "empty_query") {
    return (
      <section className="panel" data-testid="empty-query">
        <p className="panel-title">{response.summary}</p>
      </section>
    );
  }

  const question = response.clarifying_questions[0];
  const language = response.detected_language;

  return (
    <section className="answer" data-testid="answer" data-status={response.status}>
      <div className="results-head">
        <h2>{STATUS_HEADINGS[response.status] ?? "Answer"}</h2>
        <div className="head-aside">
          <EvidenceBadge level={response.evidence.level} />
          <Timings response={response} roundTripMs={roundTripMs} />
        </div>
      </div>

      <p className="answer-summary" dir={directionFor(response.summary)} data-testid="answer-summary">
        {response.summary}
      </p>

      {question && (
        <div className="clarify" data-testid="clarifying-question" role="status">
          <p className="clarify-label">Confirm one thing first</p>
          <p className="clarify-question" dir={directionFor(question)}>
            {question}
          </p>
          <p className="clarify-note">
            The steps below are correct for the procedure. This answer is the part
            that depends on the case, so it is asked rather than assumed.
          </p>
        </div>
      )}

      {response.conflicts.length > 0 && (
        <section className="conflicts" data-testid="answer-conflicts">
          <h3>Procedures that may overlap</h3>
          <ul>
            {response.conflicts.map((conflict) => (
              <li key={conflict}>{conflict}</li>
            ))}
          </ul>
          <p className="panel-body">
            Both are shown rather than one being picked for you. The knowledge base
            contains an unresolved overlap, and resolving it here would hide that.
          </p>
        </section>
      )}

      <ol className="steps" data-testid="answer-steps">
        {response.steps.map((step, index) => (
          <AnswerStepRow
            key={`${step.step_id}-${index}`}
            step={step}
            position={index + 1}
            language={language}
          />
        ))}
      </ol>

      {response.inserted_prerequisites.length > 0 && (
        <p className="inserted-note" data-testid="inserted-prerequisites">
          {response.inserted_prerequisites.length === 1 ? "One step was" : `${response.inserted_prerequisites.length} steps were`}{" "}
          added by the server because a cited step records that it depends on{" "}
          {response.inserted_prerequisites.length === 1 ? "it" : "them"}:{" "}
          {response.inserted_prerequisites.join(", ")}.
        </p>
      )}

      {response.escalation && (
        <section className="escalation" data-testid="escalation">
          <h3 className="section-heading">If this needs escalating</h3>
          <p dir={directionFor(response.escalation)}>{response.escalation}</p>
        </section>
      )}

      {response.customer_wording && <CustomerWording text={response.customer_wording} />}

      <ContextUsedPanel context={response.context_used} />

      {response.sources.length > 0 && (
        <section className="sources" data-testid="answer-sources">
          <h3 className="section-heading">Sources</h3>
          <ul>
            {response.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </section>
      )}

      {response.conditions.length > 0 && (
        <details className="conditions" data-testid="answer-conditions">
          <summary>When this procedure applies ({response.conditions.length} recorded conditions)</summary>
          <ul>
            {response.conditions.map((condition) => (
              <li key={condition}>{condition}</li>
            ))}
          </ul>
          <p className="panel-body">
            Quoted from the procedure, in its own wording, so they can be matched
            against the source document.
          </p>
        </details>
      )}

      <FeedbackBar response={response} />
    </section>
  );
}

/** The wording to read to the customer, with a one-click copy. */
function CustomerWording({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard access can be denied; the text is on screen and selectable.
      setCopied(false);
    }
  }

  return (
    <section className="customer" data-testid="customer-wording">
      <div className="customer-head">
        <h3 className="section-heading">Wording for the customer</h3>
        <button type="button" className="copy-button" onClick={() => void copy()} data-testid="copy-wording">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p className="customer-text" dir={directionFor(text)}>
        {text}
      </p>
      <p className="customer-note">
        A suggestion for phrasing only. It carries no policy weight — the steps above do.
      </p>
    </section>
  );
}

function AbstentionPanel({ response }: { response: AnswerResponse }) {
  return (
    <section className="panel panel-abstain" data-testid="answer-abstention">
      <div className="results-head" style={{ margin: "0 0 10px" }}>
        <p className="panel-title" data-testid="answer-abstention-message">
          {response.summary}
        </p>
        <EvidenceBadge level={response.evidence.level} />
      </div>
      <p className="panel-body" data-testid="abstention-reason">
        {response.evidence.reason}
      </p>
      <p className="panel-body" style={{ marginTop: 9 }}>
        This is a statement about the knowledge base, not a failure. Nothing here
        covers the case, so rephrase it with more detail or escalate.
      </p>
      <p className="panel-detail" data-testid="model-not-called">
        {response.model_called
          ? "The model answered, but nothing it cited could be traced to a step, so the answer was discarded."
          : "The answer model was never called — there was nothing grounded to answer from."}
      </p>
    </section>
  );
}

function GenerationFailedPanel({ response }: { response: AnswerResponse }) {
  return (
    <section className="panel panel-error" data-testid="answer-generation-failed" role="alert">
      <p className="panel-title">The answer model did not respond</p>
      <p className="panel-body">{response.summary}</p>
      {response.failure_detail && (
        <p className="panel-detail" data-testid="failure-detail">
          {response.failure_detail}
        </p>
      )}
      {response.sources.length > 0 && (
        <div className="context-block">
          <p className="context-line-label">
            The procedure was found. Read it directly while generation is down:
          </p>
          <ul className="context-items">
            {response.sources.map((source) => (
              <li key={source}>{source}</li>
            ))}
          </ul>
        </div>
      )}
      <p className="panel-body" style={{ marginTop: 9 }}>
        Check that Ollama is running, then try again. Search is unaffected.
      </p>
    </section>
  );
}

/**
 * Retrieval and generation, reported separately.
 *
 * "The answer took nine seconds" is not actionable. "Retrieval took 180 ms and
 * generation took eight seconds" tells you which half to look at, and these two
 * halves have completely different cost profiles.
 */
function Timings({ response, roundTripMs }: { response: AnswerResponse; roundTripMs: number }) {
  const retrieval = response.timings_ms.retrieval ?? 0;
  const generation = response.timings_ms.model_generation ?? 0;
  const total = response.timings_ms.total ?? 0;

  return (
    <span className="timing" data-testid="answer-timing">
      search {Math.round(retrieval)} ms · generation {formatSeconds(generation)} · server{" "}
      {formatSeconds(total)} · round trip {formatSeconds(roundTripMs)}
    </span>
  );
}

function formatSeconds(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

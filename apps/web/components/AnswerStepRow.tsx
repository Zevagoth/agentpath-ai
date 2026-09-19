import { directionFor } from "@/lib/direction";
import type { AnswerStep, Language } from "@/lib/types";

import { ScreenshotFigure } from "./ScreenshotFigure";

/**
 * One step: what to do, where it came from, and what it looks like on screen.
 *
 * The citation is not a footnote. Everything except `instruction` was attached by
 * the server from the procedure catalogue after the model chose a step id, so the
 * procedure, the version, the control name and the source section are facts about
 * the corpus rather than model output - and the row shows them together so an
 * agent can verify the instruction against the document before acting.
 *
 * A `prerequisite` step is labelled. The model did not choose it: the server
 * inserted it because a cited step declares it depends on this one, and the
 * wording is the procedure's own sentence. Presenting that as though the model had
 * written it for this case would be a small lie in the one place the product sells
 * itself on not telling any.
 */

export function AnswerStepRow({
  step,
  position,
  language,
}: {
  step: AnswerStep;
  position: number;
  language: Language;
}) {
  const inserted = step.source === "prerequisite";

  return (
    <li
      className={`step${inserted ? " step-inserted" : ""}`}
      data-testid="answer-step"
      data-step-id={step.step_id}
      data-source={step.source}
    >
      <div className="step-index" aria-hidden="true">
        {position}
      </div>

      <div className="step-main">
        <p className="step-instruction" dir={directionFor(step.instruction)} data-testid="step-instruction">
          {step.instruction}
        </p>

        {inserted && (
          <p className="step-note" data-testid="prerequisite-note">
            Required before the step above it.{" "}
            {step.requires.length > 0
              ? `The procedure records that a later step depends on this one, so it is shown in the procedure's own words.`
              : `Added by the server in the procedure's own words.`}
          </p>
        )}

        <div className="step-meta">
          <span className="step-citation" data-testid="step-citation">
            {step.citation}
          </span>
          {step.control_name && (
            <span className="step-control" data-testid="step-control">
              {step.control_name}
            </span>
          )}
          {step.requires.length > 0 && (
            <span className="step-requires" data-testid="step-requires">
              needs {step.requires.join(", ")}
            </span>
          )}
        </div>

        <p className="step-source" data-testid="step-source-section">
          {step.source_section}
        </p>

        {step.conditions.length > 0 && (
          <ul className="step-conditions" data-testid="step-conditions">
            {step.conditions.map((condition) => (
              <li key={condition} dir={directionFor(condition)}>
                {condition}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="step-shot">
        {step.screenshot ? (
          <ScreenshotFigure screenshot={step.screenshot} language={language} stepId={step.step_id} />
        ) : (
          <p className="shot-absent" data-testid="no-screenshot">
            No capture for this step
          </p>
        )}
      </div>
    </li>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";

import { API_BASE_URL } from "@/lib/api";
import type { Language, ScreenshotRef } from "@/lib/types";

/**
 * The capture for one step, beside the step, enlargeable.
 *
 * Shown small by default and opened full size on click, because both sizes do a
 * different job: the thumbnail tells the agent *which* screen they should be
 * looking at without pushing the next instruction off the page, and the full size
 * is what they actually read a control label from.
 *
 * `<img>` rather than `next/image`: the bytes come from the API origin on a port
 * that changes between environments, and teaching the image optimiser about a
 * loopback origin buys nothing here. Intrinsic `width`/`height` are passed so the
 * row does not reflow when the image lands.
 */

export function ScreenshotFigure({
  screenshot,
  language,
  stepId,
}: {
  screenshot: ScreenshotRef;
  language: Language;
  stepId: string;
}) {
  const [enlarged, setEnlarged] = useState(false);
  const description = descriptionFor(screenshot, language);

  return (
    <>
      <figure className="shot" data-testid="step-screenshot" data-screenshot-id={screenshot.screenshot_id}>
        <button
          type="button"
          className="shot-button"
          onClick={() => setEnlarged(true)}
          data-testid="enlarge-screenshot"
          aria-label={`Enlarge the screenshot for ${stepId}: ${description}`}
        >
          <img
            src={absolute(screenshot.url)}
            alt={description}
            width={screenshot.width}
            height={screenshot.height}
            className="shot-image"
            loading="lazy"
          />
          <span className="shot-zoom" aria-hidden="true">
            Enlarge
          </span>
        </button>
        <figcaption className="shot-caption">
          <span className="shot-screen">{screenshot.screen_name}</span>
          {screenshot.control_name && <span className="shot-control">{screenshot.control_name}</span>}
        </figcaption>
      </figure>

      {enlarged && (
        <ScreenshotLightbox
          screenshot={screenshot}
          description={description}
          onClose={() => setEnlarged(false)}
        />
      )}
    </>
  );
}

/**
 * The enlarged view.
 *
 * Hand-rolled rather than `<dialog showModal>`: jsdom does not implement
 * `showModal`, so the behaviour asserted in the tests would not be the behaviour
 * that ships. Escape closes, the backdrop closes, focus moves to the dialog on
 * open and returns to the trigger on close, and the image itself does not swallow
 * the backdrop click.
 */
function ScreenshotLightbox({
  screenshot,
  description,
  onClose,
}: {
  screenshot: ScreenshotRef;
  description: string;
  onClose: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const returnFocusTo = useRef<Element | null>(null);

  useEffect(() => {
    returnFocusTo.current = document.activeElement;
    panel.current?.focus();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", onKeyDown);

    // The page behind must not scroll while the overlay is up.
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      (returnFocusTo.current as HTMLElement | null)?.focus?.();
    };
  }, [onClose]);

  return (
    <div
      className="lightbox"
      data-testid="screenshot-lightbox"
      onClick={onClose}
      role="presentation"
    >
      <div
        ref={panel}
        className="lightbox-panel"
        role="dialog"
        aria-modal="true"
        aria-label={description}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="lightbox-bar">
          <div>
            <p className="lightbox-title">{screenshot.screen_name}</p>
            <p className="lightbox-sub">
              {screenshot.procedure_id} v{screenshot.procedure_version} · step {screenshot.step_order}
              {screenshot.control_name ? ` · ${screenshot.control_name}` : ""}
            </p>
          </div>
          <button
            type="button"
            className="lightbox-close"
            onClick={onClose}
            data-testid="close-lightbox"
            aria-label="Close the enlarged screenshot"
          >
            ✕
          </button>
        </div>

        <img
          src={absolute(screenshot.url)}
          alt={description}
          width={screenshot.width}
          height={screenshot.height}
          className="lightbox-image"
        />

        <p className="lightbox-meta">
          {screenshot.width}×{screenshot.height} · {screenshot.watermark}
        </p>
      </div>
    </div>
  );
}

/** Pick the caption in the language the answer is written in. */
function descriptionFor(screenshot: ScreenshotRef, language: Language): string {
  if (language === "ar" && screenshot.description.ar) return screenshot.description.ar;
  return screenshot.description.en || screenshot.screen_name;
}

/** The API returns a root-relative URL; the console may be on another origin. */
function absolute(url: string): string {
  return url.startsWith("http") ? url : `${API_BASE_URL}${url}`;
}

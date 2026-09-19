import { clampSegments, segmentSnippet, snippetText } from "@/lib/highlight";
import { directionFor } from "@/lib/direction";

/**
 * A retrieved passage with the engine's term matches marked.
 *
 * Every run is a React text node, so the rendered output contains no element this
 * component did not create. See `lib/highlight.ts` for why that matters more than
 * the corpus being synthetic.
 */
export function HighlightedSnippet({
  snippet,
  className,
  maxLength = 420,
  testId,
}: {
  snippet: string;
  className?: string;
  maxLength?: number;
  testId?: string;
}) {
  const segments = clampSegments(segmentSnippet(snippet), maxLength);

  return (
    <p className={className} dir={directionFor(snippetText(snippet))} data-testid={testId}>
      {segments.map((segment, index) =>
        segment.emphasis ? (
          // eslint-disable-next-line react/no-array-index-key -- position *is* the identity of a text run
          <em key={index} className="snippet-match">
            {segment.text}
          </em>
        ) : (
          // eslint-disable-next-line react/no-array-index-key
          <span key={index}>{segment.text}</span>
        ),
      )}
    </p>
  );
}

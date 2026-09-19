/**
 * Turning RAGFlow's highlighted snippets into data, not markup.
 *
 * The engine returns matched terms wrapped in `<em>`. That used to be handed to
 * `dangerouslySetInnerHTML`, justified on the grounds that the corpus is our own
 * synthetic text and the tags come from RAGFlow's highlighter rather than from a
 * user. Both halves of that argument are weaker than they look.
 *
 * The snippet is assembled by the engine from *indexed document content*, and
 * document content is exactly the thing this system treats as untrusted
 * everywhere else: T8 rebuilds every generated step from the committed YAML
 * precisely so that text arriving from the index cannot reach the model. Handing
 * the same bytes to the DOM as HTML contradicts that. And the trust argument is
 * about today's corpus, whereas the code outlives the corpus - the moment anything
 * is ingested that was not written here, the sink is already in place.
 *
 * So the snippet is parsed into segments and rendered as React text nodes. The
 * output cannot contain an element React did not create, which makes the question
 * of who authored the bytes irrelevant.
 *
 * Only `<em>` is understood. Any other tag is dropped rather than shown as text:
 * the engine does not emit them, so their presence means something is wrong, and
 * printing `<script>` into the page as literal text would be noise at best.
 */

export interface Segment {
  text: string;
  /** Whether the engine marked this run as a query-term match. */
  emphasis: boolean;
}

/** The named entities RAGFlow's escaping can produce. */
const ENTITIES: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&nbsp;": " ",
};

function decodeEntities(text: string): string {
  return text.replace(/&(?:amp|lt|gt|quot|apos|nbsp|#39);/g, (match) => ENTITIES[match] ?? match);
}

/**
 * Split a snippet into emphasised and plain runs.
 *
 * Unbalanced tags are tolerated on purpose - a snippet is a *clip* of a larger
 * document, so the engine can hand back an `</em>` whose opener was cut off. An
 * orphan closer ends emphasis that was never started, which is a no-op, and an
 * unclosed `<em>` emphasises to the end of the clip. Neither should lose text.
 */
export function segmentSnippet(snippet: string): Segment[] {
  const segments: Segment[] = [];
  let emphasis = false;
  let buffer = "";

  const flush = () => {
    if (buffer.length === 0) return;
    const text = decodeEntities(buffer);
    const previous = segments[segments.length - 1];
    // Merge runs that ended up with the same styling, so dropping an unknown tag
    // in the middle of a sentence does not split it into two nodes.
    if (previous && previous.emphasis === emphasis) {
      previous.text += text;
    } else {
      segments.push({ text, emphasis });
    }
    buffer = "";
  };

  const tag = /<\/?([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*>/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = tag.exec(snippet)) !== null) {
    buffer += snippet.slice(cursor, match.index);
    cursor = match.index + match[0].length;

    const name = match[1].toLowerCase();
    if (name !== "em") continue; // unknown tag: dropped, surrounding text kept

    flush();
    emphasis = !match[0].startsWith("</");
  }

  buffer += snippet.slice(cursor);
  flush();

  return segments;
}

/**
 * Trim to a readable length, counting visible characters.
 *
 * The previous implementation clamped the raw string, so a snippet heavy with
 * `<em>` markup was cut far shorter than one without - the tags counted against
 * the budget. Segmenting first means the limit applies to what the agent actually
 * reads.
 */
export function clampSegments(segments: Segment[], maxLength = 420): Segment[] {
  let remaining = maxLength;
  const kept: Segment[] = [];

  for (const segment of segments) {
    if (remaining <= 0) break;
    if (segment.text.length <= remaining) {
      kept.push(segment);
      remaining -= segment.text.length;
      continue;
    }
    // Break on a word boundary when there is one nearby, so the clip does not end
    // mid-word.
    const slice = segment.text.slice(0, remaining);
    const lastSpace = slice.lastIndexOf(" ");
    const cut = lastSpace > remaining - 40 ? slice.slice(0, lastSpace) : slice;
    kept.push({ text: `${cut.trimEnd()}…`, emphasis: segment.emphasis });
    return kept;
  }

  const total = segments.reduce((sum, segment) => sum + segment.text.length, 0);
  if (total > maxLength && kept.length > 0) {
    kept[kept.length - 1] = {
      ...kept[kept.length - 1],
      text: `${kept[kept.length - 1].text.trimEnd()}…`,
    };
  }
  return kept;
}

/** Plain text of a snippet, for direction detection and for `title` attributes. */
export function snippetText(snippet: string): string {
  return segmentSnippet(snippet)
    .map((segment) => segment.text)
    .join("");
}

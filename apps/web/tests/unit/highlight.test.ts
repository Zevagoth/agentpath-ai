import { describe, expect, it } from "vitest";

import { clampSegments, segmentSnippet, snippetText } from "@/lib/highlight";

/**
 * The parser that replaced `dangerouslySetInnerHTML`.
 *
 * The behaviour worth pinning is not "it finds `<em>`" but "nothing else in the
 * snippet can become markup", because that is the property the old code did not
 * have.
 */

describe("snippet segmentation", () => {
  it("marks the engine's matched terms and leaves the rest plain", () => {
    const segments = segmentSnippet("the property <em>cannot host</em> a confirmed guest");

    expect(segments).toEqual([
      { text: "the property ", emphasis: false },
      { text: "cannot host", emphasis: true },
      { text: " a confirmed guest", emphasis: false },
    ]);
  });

  it("keeps Arabic text intact", () => {
    const segments = segmentSnippet("الفندق <em>ما عنده غرف</em> للضيف");
    expect(segments[1]).toEqual({ text: "ما عنده غرف", emphasis: true });
    expect(snippetText("الفندق <em>ما عنده غرف</em> للضيف")).toBe("الفندق ما عنده غرف للضيف");
  });

  it("drops any tag that is not <em>, and keeps the words around it", () => {
    const segments = segmentSnippet("before <script>alert(1)</script> after");

    // The script *text* survives as text - it is corpus content - but there is no
    // element left for a browser to execute.
    expect(segments).toEqual([{ text: "before alert(1) after", emphasis: false }]);
    expect(segments.every((segment) => !segment.text.includes("<"))).toBe(true);
  });

  it("neutralises an injected image with an error handler", () => {
    const segments = segmentSnippet(`<img src=x onerror="steal()">caption`);
    expect(snippetText(`<img src=x onerror="steal()">caption`)).toBe("caption");
    expect(segments).toEqual([{ text: "caption", emphasis: false }]);
  });

  it("decodes the entities the engine escapes", () => {
    expect(snippetText("cost &amp; recovery &lt;tab&gt;")).toBe("cost & recovery <tab>");
  });

  it("does not let a decoded entity become a tag again", () => {
    // If decoding happened before tag stripping, this would reintroduce an element.
    const segments = segmentSnippet("&lt;em&gt;not emphasis&lt;/em&gt;");
    expect(segments).toEqual([{ text: "<em>not emphasis</em>", emphasis: false }]);
    expect(segments.some((segment) => segment.emphasis)).toBe(false);
  });

  it("tolerates a clip that begins inside an emphasis", () => {
    // A snippet is a window onto a document, so the opener can be outside it. An
    // orphan closer ends emphasis that never started, which is a no-op, and the
    // two plain runs either side of it collapse into one text node.
    expect(segmentSnippet("host</em> a guest")).toEqual([
      { text: "host a guest", emphasis: false },
    ]);
  });

  it("tolerates a clip that ends inside an emphasis", () => {
    expect(segmentSnippet("the <em>relocation finder")).toEqual([
      { text: "the ", emphasis: false },
      { text: "relocation finder", emphasis: true },
    ]);
  });

  it("returns nothing for an empty snippet", () => {
    expect(segmentSnippet("")).toEqual([]);
  });
});

describe("clamping", () => {
  it("counts visible characters, not markup", () => {
    // 40 visible characters wrapped in tags. The old implementation clamped the
    // raw string, so markup stole from the budget and heavily highlighted
    // snippets were cut far shorter than plain ones.
    const heavy = Array.from({ length: 10 }, () => "<em>abcd</em>").join("");
    const clamped = clampSegments(segmentSnippet(heavy), 40);

    expect(snippetText(heavy)).toHaveLength(40);
    expect(clamped.map((segment) => segment.text).join("")).toBe("abcd".repeat(10));
  });

  it("trims to the limit and marks the cut", () => {
    const segments = clampSegments(segmentSnippet("one two three four five six seven"), 12);
    const text = segments.map((segment) => segment.text).join("");

    expect(text.endsWith("…")).toBe(true);
    expect(text.length).toBeLessThanOrEqual(13);
  });

  it("leaves a short snippet alone", () => {
    const segments = clampSegments(segmentSnippet("short <em>enough</em>"), 420);
    expect(segments.map((segment) => segment.text).join("")).toBe("short enough");
  });
});

/**
 * Text direction, decided by script rather than by a language setting.
 *
 * Agents switch languages mid-case, so the direction of an individual field
 * follows that field's own content: an Arabic case description reads
 * right-to-left even while the surrounding console stays left-to-right.
 */

const ARABIC_RANGES: [number, number][] = [
  [0x0600, 0x06ff],
  [0x0750, 0x077f],
  [0x08a0, 0x08ff],
  [0xfb50, 0xfdff],
  [0xfe70, 0xfeff],
];

export type Direction = "rtl" | "ltr";

export function arabicRatio(text: string): number {
  const letters = [...text].filter((char) => /\p{L}/u.test(char));
  if (letters.length === 0) return 0;
  const arabic = letters.filter((char) => {
    const code = char.codePointAt(0) ?? 0;
    return ARABIC_RANGES.some(([start, end]) => code >= start && code <= end);
  });
  return arabic.length / letters.length;
}

/**
 * Direction for a piece of text.
 *
 * The threshold sits at 40% rather than 50% because a mixed query such as
 * "overbooking بالفندق" reads more naturally right-to-left once a meaningful
 * share of it is Arabic.
 */
export function directionFor(text: string): Direction {
  return arabicRatio(text) >= 0.4 ? "rtl" : "ltr";
}

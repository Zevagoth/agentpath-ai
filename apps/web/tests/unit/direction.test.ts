import { describe, expect, it } from "vitest";

import { arabicRatio, directionFor } from "@/lib/direction";

describe("text direction", () => {
  it("reads Arabic right-to-left", () => {
    expect(directionFor("الفندق ما عنده غرف")).toBe("rtl");
  });

  it("reads English left-to-right", () => {
    expect(directionFor("hotel has no room")).toBe("ltr");
  });

  it("flips a mostly-Arabic mixed query to right-to-left", () => {
    expect(directionFor("overbooking بالفندق والضيف موجود واقف")).toBe("rtl");
  });

  it("keeps a mostly-English mixed query left-to-right", () => {
    expect(directionFor("the property overbooked and walked our guest الفندق")).toBe("ltr");
  });

  it("treats an empty field as left-to-right so the caret does not jump", () => {
    expect(directionFor("")).toBe("ltr");
  });

  it("ignores digits and punctuation when measuring script", () => {
    expect(arabicRatio("NW-HTL-014")).toBe(0);
    expect(arabicRatio("الفندق 2026")).toBe(1);
  });
});

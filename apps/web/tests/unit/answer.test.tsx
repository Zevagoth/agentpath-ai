import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AgentConsole } from "@/components/AgentConsole";
import { clearFeedback, readFeedback } from "@/lib/feedback";
import type { AnswerResponse, AnswerStep, ScreenshotRef } from "@/lib/types";

/**
 * The answer console, T11.
 *
 * The fixture mirrors a real `POST /api/answer` body taken from the gate run, so
 * the assertions are about the shape the API actually returns rather than a shape
 * convenient for testing.
 *
 * Weighted towards the states an agent must be able to tell apart. "No procedure
 * exists", "the model server is down" and "the API is unreachable" are three
 * different facts with three different remedies, and a console that renders them
 * alike teaches an agent to distrust all three.
 */

function shot(overrides: Partial<ScreenshotRef> = {}): ScreenshotRef {
  return {
    screenshot_id: "agentdesk-relocation-finder",
    url: "/api/screenshots/agentdesk-relocation-finder",
    media_type: "image/png",
    width: 1280,
    height: 720,
    bytes: 214_003,
    sha256: "a".repeat(64),
    procedure_id: "NW-HTL-014",
    procedure_version: "2.1",
    step_order: 5,
    screen_name: "Relocation Finder",
    control_name: "Find equivalent stays",
    description: {
      en: "The Relocation Finder with the distance filter at its default.",
      ar: "أداة إيجاد البديل مع فلتر المسافة الافتراضي.",
    },
    watermark: "SYNTHETIC — NW-HTL-014 v2.1 step 5",
    language: "en",
    highlight: null,
    ...overrides,
  };
}

function step(overrides: Partial<AnswerStep> = {}): AnswerStep {
  return {
    instruction: "Open the booking in AgentDesk and confirm the status chip reads Confirmed.",
    procedure_id: "NW-HTL-014",
    procedure_version: "2.1",
    step_id: "NW-HTL-014#S01",
    citation: "NW-HTL-014 v2.1 · NW-HTL-014#S01",
    source_section: "NW-HTL-014.md > ## Steps > ### Step 1",
    control_name: "Booking status chip",
    conditions: [],
    screenshot: null,
    source: "model",
    requires: [],
    ...overrides,
  };
}

function answerBody(overrides: Partial<AnswerResponse> = {}): AnswerResponse {
  return {
    status: "answered",
    query: "The hotel walked our guest. How do I handle the price difference?",
    detected_language: "en",
    summary:
      "The property cannot host the guest, so an equivalent stay has to be secured and the verified cost difference recorded.",
    steps: [
      step(),
      step({
        instruction: "Open Relocation Finder and search for equivalent-or-better stays.",
        step_id: "NW-HTL-014#S05",
        citation: "NW-HTL-014 v2.1 · NW-HTL-014#S05",
        source_section: "NW-HTL-014.md > ## Steps > ### Step 5",
        control_name: "Find equivalent stays",
        screenshot: shot(),
      }),
      step({
        instruction: "Present the shortlisted option to the guest and get their agreement.",
        step_id: "NW-HTL-014#S06",
        citation: "NW-HTL-014 v2.1 · NW-HTL-014#S06",
        source_section: "NW-HTL-014.md > ## Steps > ### Step 6",
        control_name: null,
        requires: ["S05"],
      }),
    ],
    conditions: ["applies_when: the booking status is Confirmed"],
    escalation: "Escalate to Accommodation Support if no equivalent option is confirmed.",
    customer_wording: "I am arranging an equivalent stay for you now.",
    clarifying_questions: [],
    conflicts: [],
    sources: [
      "NW-HTL-014 v2.1 (effective 2026-03-01) — Property unable to accommodate a confirmed guest",
    ],
    screenshots: [shot()],
    evidence: {
      level: "strong",
      reason: "NW-HTL-014 scored 0.61, ahead of the next procedure by 0.08.",
      abstained: false,
      message: null,
    },
    model_called: true,
    failure_detail: null,
    dropped_citations: [],
    inserted_prerequisites: [],
    context_used: {
      provided: [],
      applied: {},
      unanswered: [],
      not_collected: {},
      completed_steps: [],
      product_line: null,
    },
    timings_ms: {
      query_processing: 0.2,
      retrieval: 182.4,
      evidence_validation: 0.9,
      prompt_construction: 0.9,
      model_generation: 6805.2,
      total: 6991.1,
    },
    retrieval_metadata: { model: "qwen3.5:4b" },
    ...overrides,
  };
}

/** Route by URL so a test cannot accidentally assert search behaviour. */
function stubAnswer(payload: unknown, status = 200) {
  const fn = vi.fn(async (url: string) => {
    if (!String(url).includes("/api/answer")) {
      throw new Error(`unexpected request to ${url}`);
    }
    return new Response(JSON.stringify(payload), { status });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function ask(text = "the hotel walked our guest") {
  render(<AgentConsole />);
  await userEvent.click(screen.getByTestId("case-input"));
  await userEvent.paste(text);
  await userEvent.click(screen.getByTestId("search-button"));
}

beforeEach(() => {
  clearFeedback();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("answer mode", () => {
  it("opens on the answer mode and calls /api/answer", async () => {
    const fetchMock = stubAnswer(answerBody());
    render(<AgentConsole />);

    expect(screen.getByTestId("mode-answer").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("search-button").textContent).toContain("Answer this case");

    await userEvent.click(screen.getByTestId("case-input"));
    await userEvent.paste("hotel walked our guest");
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("answer")).toBeTruthy());
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toContain("/api/answer");
    expect(JSON.parse(init.body as string)).toMatchObject({
      query: "hotel walked our guest",
      limit: 3,
      case_context: null,
    });
  });

  it("shows a clear summary for the case", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("answer-summary")).toBeTruthy());
    expect(screen.getByTestId("answer-summary").textContent).toContain(
      "equivalent stay has to be secured",
    );
  });

  it("lists the steps in order, each with its procedure, version and step id", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getAllByTestId("answer-step").length).toBe(3));
    const steps = screen.getAllByTestId("answer-step");

    expect(steps.map((node) => node.getAttribute("data-step-id"))).toEqual([
      "NW-HTL-014#S01",
      "NW-HTL-014#S05",
      "NW-HTL-014#S06",
    ]);

    const citations = screen.getAllByTestId("step-citation").map((node) => node.textContent);
    expect(citations[0]).toBe("NW-HTL-014 v2.1 · NW-HTL-014#S01");
    // The version is on every step, not just in a header.
    expect(citations.every((text) => text?.includes("v2.1"))).toBe(true);
  });

  it("names the source section so the step can be checked against the document", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getAllByTestId("step-source-section").length).toBe(3));
    expect(screen.getAllByTestId("step-source-section")[0].textContent).toBe(
      "NW-HTL-014.md > ## Steps > ### Step 1",
    );
  });

  it("puts the screenshot beside the step it belongs to", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getAllByTestId("answer-step").length).toBe(3));

    const relocationStep = screen
      .getAllByTestId("answer-step")
      .find((node) => node.getAttribute("data-step-id") === "NW-HTL-014#S05")!;

    // The capture is inside its own step's row, not in a gallery at the end.
    const figure = within(relocationStep).getByTestId("step-screenshot");
    expect(figure.getAttribute("data-screenshot-id")).toBe("agentdesk-relocation-finder");

    const image = within(figure).getByRole("img") as HTMLImageElement;
    expect(image.getAttribute("src")).toContain("/api/screenshots/agentdesk-relocation-finder");
    // Alt text from the sidecar, and intrinsic size so the row does not reflow.
    expect(image.getAttribute("alt")).toContain("Relocation Finder");
    expect(image.getAttribute("width")).toBe("1280");

    // Steps without a capture say so rather than leaving a gap.
    const firstStep = screen.getAllByTestId("answer-step")[0];
    expect(within(firstStep).getByTestId("no-screenshot")).toBeTruthy();
  });

  it("enlarges a screenshot and closes again", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("enlarge-screenshot")).toBeTruthy());
    expect(screen.queryByTestId("screenshot-lightbox")).toBeNull();

    await userEvent.click(screen.getByTestId("enlarge-screenshot"));

    const lightbox = screen.getByTestId("screenshot-lightbox");
    expect(within(lightbox).getByRole("dialog").getAttribute("aria-modal")).toBe("true");
    // The enlarged view carries the provenance of the capture it is showing.
    expect(lightbox.textContent).toContain("NW-HTL-014");
    expect(lightbox.textContent).toContain("step 5");

    await userEvent.click(screen.getByTestId("close-lightbox"));
    expect(screen.queryByTestId("screenshot-lightbox")).toBeNull();
  });

  it("closes the enlarged screenshot with Escape", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("enlarge-screenshot")).toBeTruthy());
    await userEvent.click(screen.getByTestId("enlarge-screenshot"));
    expect(screen.getByTestId("screenshot-lightbox")).toBeTruthy();

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByTestId("screenshot-lightbox")).toBeNull();
  });

  it("labels a server-inserted prerequisite instead of passing it off as generated", async () => {
    stubAnswer(
      answerBody({
        steps: [
          step({
            instruction:
              "Open Relocation Finder and search for equivalent-or-better stays, keeping the default distance filter.",
            step_id: "NW-HTL-014#S05",
            citation: "NW-HTL-014 v2.1 · NW-HTL-014#S05",
            source: "prerequisite",
          }),
          step({
            instruction: "Present the shortlisted option to the guest.",
            step_id: "NW-HTL-014#S06",
            citation: "NW-HTL-014 v2.1 · NW-HTL-014#S06",
            requires: ["S05"],
          }),
        ],
        inserted_prerequisites: ["NW-HTL-014#S05"],
      }),
    );
    await ask();

    await waitFor(() => expect(screen.getAllByTestId("answer-step").length).toBe(2));
    const steps = screen.getAllByTestId("answer-step");

    expect(steps[0].getAttribute("data-source")).toBe("prerequisite");
    expect(steps[1].getAttribute("data-source")).toBe("model");
    expect(within(steps[0]).getByTestId("prerequisite-note")).toBeTruthy();
    // The dependency is shown on the step that declares it.
    expect(within(steps[1]).getByTestId("step-requires").textContent).toContain("S05");
    expect(screen.getByTestId("inserted-prerequisites").textContent).toContain("NW-HTL-014#S05");
  });

  it("asks exactly one clarifying question when the case is underdetermined", async () => {
    stubAnswer(
      answerBody({
        status: "clarification_needed",
        clarifying_questions: ["Is the guest at the property right now?"],
        evidence: {
          level: "partial",
          reason: "NW-HTL-014 scored 0.55.",
          abstained: false,
          message: null,
        },
      }),
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("clarifying-question")).toBeTruthy());
    expect(screen.getByTestId("clarifying-question").textContent).toContain(
      "Is the guest at the property right now?",
    );
    // The steps are still shown: they are valid, it is the case that is not settled.
    expect(screen.getAllByTestId("answer-step").length).toBe(3);
    expect(screen.getByTestId("answer").getAttribute("data-status")).toBe("clarification_needed");
  });

  it("reports search time and generation time separately", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("answer-timing")).toBeTruthy());
    const timing = screen.getByTestId("answer-timing").textContent ?? "";

    expect(timing).toContain("search 182 ms");
    expect(timing).toContain("generation 6.8 s");
    expect(timing).toContain("server 7.0 s");
  });

  it("shows the customer wording separately from the steps, with a copy button", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("customer-wording")).toBeTruthy());
    const panel = screen.getByTestId("customer-wording");
    expect(panel.textContent).toContain("I am arranging an equivalent stay");
    // Wording is not policy, and the panel says so.
    expect(panel.textContent).toContain("no policy weight");
    expect(within(panel).getByTestId("copy-wording")).toBeTruthy();
  });

  it("shows the escalation route and the sources", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("escalation")).toBeTruthy());
    expect(screen.getByTestId("escalation").textContent).toContain("Accommodation Support");
    expect(screen.getByTestId("answer-sources").textContent).toContain("NW-HTL-014 v2.1");
  });
});

describe("the case facts", () => {
  it("sends the synthetic case context and shows what it did", async () => {
    const fetchMock = stubAnswer(
      answerBody({
        context_used: {
          provided: ["booking_status", "supplier_status", "customer_arrived"],
          applied: {
            customer_arrived: "already known, so it is not asked again: whether the guest is at the property",
            supplier_status: "given to the model as case data, so it can follow the matching branch",
          },
          unanswered: ["Which nights of the stay are affected"],
          not_collected: {
            "Booking reference":
              "an identifier the agent is already looking at, and that this contract does not accept",
          },
          completed_steps: ["NW-HTL-014#S01"],
          product_line: "hotels",
        },
      }),
    );
    render(<AgentConsole />);

    await userEvent.click(screen.getByTestId("case-input"));
    await userEvent.paste("hotel walked our guest");

    await userEvent.click(screen.getByTestId("ctx-load-example"));
    await userEvent.click(screen.getByTestId("search-button"));

    await waitFor(() => expect(screen.getByTestId("context-used")).toBeTruthy());

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    const sent = JSON.parse(init.body as string);
    expect(sent.case_context).toMatchObject({
      booking_status: "confirmed",
      supplier_status: "cannot_accommodate",
      customer_arrived: true,
      party_size: 2,
      rooms_booked: 1,
    });

    const panel = screen.getByTestId("context-used");
    expect(panel.textContent).toContain("already known, so it is not asked again");
    expect(within(panel).getByTestId("context-completed").textContent).toContain("NW-HTL-014#S01");
    expect(within(panel).getByTestId("context-unanswered").textContent).toContain(
      "Which nights of the stay are affected",
    );
  });

  it("shows a refused requirement as declined, with the reason", async () => {
    stubAnswer(
      answerBody({
        context_used: {
          provided: [],
          applied: {},
          unanswered: [],
          not_collected: {
            "Booking reference":
              "an identifier the agent is already looking at, and that this contract does not accept",
          },
          completed_steps: [],
          product_line: null,
        },
      }),
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("context-not-collected")).toBeTruthy());
    const declined = screen.getByTestId("context-not-collected");

    expect(declined.textContent).toContain("Booking reference");
    expect(declined.textContent).toContain("does not accept");
    // It must not appear as something the agent should go and fetch.
    expect(screen.queryByTestId("context-unanswered")).toBeNull();
  });

  it("offers no field for an identifier, and says the API refuses them", async () => {
    render(<AgentConsole />);

    const form = screen.getByTestId("case-context-form");
    const labels = within(form)
      .getAllByText(/./, { selector: "label" })
      .map((node) => node.textContent?.toLowerCase() ?? "");

    expect(labels.join(" ")).not.toContain("booking reference");
    expect(labels.join(" ")).not.toContain("guest name");
    expect(labels.join(" ")).not.toContain("card");
    expect(screen.getByTestId("context-privacy-note").textContent).toContain(
      "the API rejects those keys outright",
    );
  });

  it("surfaces a rejected field instead of showing a bare 422", async () => {
    stubAnswer(
      {
        detail: [
          { loc: ["body", "case_context", "guest_name"], msg: "Extra inputs are not permitted" },
        ],
      },
      422,
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("error")).toBeTruthy());
    expect(screen.getByTestId("error").textContent).toContain("guest_name");
    expect(screen.getByTestId("error").textContent).toContain("not permitted");
  });
});

describe("the three ways an answer can be absent", () => {
  it("renders an abstention as a statement about the knowledge base", async () => {
    stubAnswer(
      answerBody({
        status: "abstained",
        summary: "No confirmed procedure was found in the available knowledge base.",
        steps: [],
        model_called: false,
        evidence: {
          level: "insufficient",
          reason: "No passage scored above the similarity threshold.",
          abstained: true,
          message: "No confirmed procedure was found in the available knowledge base.",
        },
      }),
    );
    await ask("how do I reset my email password");

    await waitFor(() => expect(screen.getByTestId("answer-abstention")).toBeTruthy());
    expect(screen.getByTestId("answer-abstention-message").textContent).toBe(
      "No confirmed procedure was found in the available knowledge base.",
    );
    // The absence of a model call is stated, because "the AI didn't answer" and
    // "there is no procedure" are different facts.
    expect(screen.getByTestId("model-not-called").textContent).toContain(
      "never called",
    );
    expect(screen.queryByTestId("answer-generation-failed")).toBeNull();
    expect(screen.queryByTestId("error")).toBeNull();
    expect(screen.queryByTestId("answer-step")).toBeNull();
  });

  it("renders a model outage as an outage, not as an empty knowledge base", async () => {
    stubAnswer(
      answerBody({
        status: "generation_failed",
        summary:
          "The procedure was found, but the local answer model did not respond, so no answer was generated. This is not the same as finding no procedure.",
        steps: [],
        failure_detail: "OllamaUnavailable: connection refused",
      }),
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("answer-generation-failed")).toBeTruthy());
    const panel = screen.getByTestId("answer-generation-failed");

    expect(panel.textContent).toContain("not the same as finding no procedure");
    expect(screen.getByTestId("failure-detail").textContent).toContain("connection refused");
    // The procedure was found, so the agent is pointed at it.
    expect(panel.textContent).toContain("NW-HTL-014 v2.1");
    expect(screen.queryByTestId("answer-abstention")).toBeNull();
  });

  it("renders an unreachable API as a request that never ran", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("error")).toBeTruthy());
    expect(screen.getByTestId("error").textContent).toContain("Cannot reach the AgentPath API");
    expect(screen.getByTestId("error").textContent).toContain("make api");
    expect(screen.queryByTestId("answer-abstention")).toBeNull();
    expect(screen.queryByTestId("answer-generation-failed")).toBeNull();
  });

  it("tells the agent to fall back to search when generation is not configured", async () => {
    stubAnswer(
      {
        error: "generation_unavailable",
        message: "Answer generation is not configured on this deployment.",
      },
      503,
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("error")).toBeTruthy());
    const panel = screen.getByTestId("error");
    expect(panel.textContent).toContain("not configured on this deployment");
    expect(panel.textContent).toContain("Find procedures");
  });
});

describe("language direction", () => {
  it("renders an Arabic answer right-to-left and keeps control labels readable", async () => {
    stubAnswer(
      answerBody({
        detected_language: "ar",
        summary: "الضيف موجود في الاستقبال والحجز مؤكد، لذا الحالة عاجلة ويجب تأمين إقامة بديلة.",
        customer_wording: "أنا أتحقق من حالة حجزك وأبحث عن حل بديل لك فوراً.",
        steps: [
          step({
            instruction: "افتح الحجز في AgentDesk وتأكد من أن شريحة الحالة تقرأ Confirmed.",
          }),
          step({
            instruction: "افتح أداة إيجاد البديل وابحث عن إقامة مكافئة أو أفضل لليالي المتأثرة.",
            step_id: "NW-HTL-014#S05",
            citation: "NW-HTL-014 v2.1 · NW-HTL-014#S05",
            control_name: "Find equivalent stays",
            screenshot: shot(),
          }),
        ],
      }),
    );
    await ask("الفندق ما عنده غرف والضيف واصل");

    await waitFor(() => expect(screen.getByTestId("answer-summary")).toBeTruthy());

    expect(screen.getByTestId("answer-summary").getAttribute("dir")).toBe("rtl");
    expect(screen.getAllByTestId("step-instruction")[0].getAttribute("dir")).toBe("rtl");
    // The English control label is preserved verbatim inside the Arabic step.
    expect(screen.getAllByTestId("step-control")[1].textContent).toBe("Find equivalent stays");
    // The Arabic caption is used for the image in an Arabic answer.
    expect((screen.getByRole("img") as HTMLImageElement).getAttribute("alt")).toContain(
      "أداة إيجاد البديل",
    );
  });

  it("keeps an English answer left-to-right", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("answer-summary")).toBeTruthy());
    expect(screen.getByTestId("answer-summary").getAttribute("dir")).toBe("ltr");
    expect(screen.getAllByTestId("step-instruction")[0].getAttribute("dir")).toBe("ltr");
  });
});

describe("feedback", () => {
  it("records a verdict in this browser without the case text", async () => {
    stubAnswer(answerBody());
    await ask("The hotel walked our guest, booking for Dana Haddad");

    await waitFor(() => expect(screen.getByTestId("feedback")).toBeTruthy());
    expect(readFeedback()).toHaveLength(0);

    await userEvent.click(screen.getByTestId("feedback-not-helpful"));

    const stored = readFeedback();
    expect(stored).toHaveLength(1);
    expect(stored[0].verdict).toBe("not_helpful");
    expect(stored[0].stepIds).toContain("NW-HTL-014#S05");
    expect(JSON.stringify(stored)).not.toContain("Dana");

    expect(screen.getByTestId("feedback-not-helpful").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("feedback-note").textContent).toContain("in this browser only");
  });

  it("says where the feedback goes before it is given", async () => {
    stubAnswer(answerBody());
    await ask();

    await waitFor(() => expect(screen.getByTestId("feedback-note")).toBeTruthy());
    expect(screen.getByTestId("feedback-note").textContent).toContain(
      "The case text is not recorded",
    );
  });
});

describe("progress", () => {
  it("names the stages while a generation is running", async () => {
    // A request that never settles, so the loading state can be observed.
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise<Response>(() => {})),
    );
    await ask();

    await waitFor(() => expect(screen.getByTestId("loading")).toBeTruthy());
    expect(screen.getByTestId("generation-progress").textContent).toContain("local model");
    expect(screen.getByTestId("search-button").hasAttribute("disabled")).toBe(true);
  });

  it("switches to the search mode without losing the typed case", async () => {
    render(<AgentConsole />);
    await userEvent.click(screen.getByTestId("case-input"));
    await userEvent.paste("hotel has no room");

    await userEvent.click(screen.getByTestId("mode-search"));

    expect(screen.getByTestId("mode-search").getAttribute("aria-pressed")).toBe("true");
    expect((screen.getByTestId("case-input") as HTMLTextAreaElement).value).toBe(
      "hotel has no room",
    );
    expect(screen.getByTestId("search-button").textContent).toContain("Find procedures");
    // The case-facts panel belongs to the answer flow only.
    expect(screen.queryByTestId("case-context-form")).toBeNull();
  });
});

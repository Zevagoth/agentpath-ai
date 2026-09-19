/**
 * Types mirroring the AgentPath API contract (`GET /openapi.json`).
 *
 * Kept hand-written rather than generated: the surface is small, and a hand
 * written copy makes an API change fail at compile time in a readable place.
 */

export type Language = "ar" | "en" | "mixed" | "unknown";

export type EvidenceLevel = "strong" | "partial" | "insufficient";

export type ProductLine =
  | "hotels"
  | "flights"
  | "cancellations"
  | "refunds"
  | "escalations";

export interface Correction {
  before: string;
  after: string;
}

export interface ConceptTag {
  concept_id: string;
  canonical: string;
  matched_surface: string;
  language: Language;
}

export interface QueryUnderstanding {
  original: string;
  normalized: string;
  language: Language;
  search_text: string;
  expansions: string[];
  corrections: Correction[];
  concepts: ConceptTag[];
  scenario_tags: string[];
  detected_product_line: string | null;
  requested_product_line: string | null;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  snippet: string;
  score: number;
  lexical_score: number | null;
  semantic_score: number | null;
}

export interface ProcedureCard {
  rank: number;
  procedure_id: string;
  title: string;
  product_line: string | null;
  version: string | null;
  effective_date: string | null;
  score: number;
  scenario_tags: string[];
  matched_sections: number;
  citations: Citation[];
}

export interface Evidence {
  level: EvidenceLevel;
  reason: string;
  abstained: boolean;
  message: string | null;
}

export interface SearchResponse {
  mode: string;
  query: QueryUnderstanding;
  evidence: Evidence;
  procedures: ProcedureCard[];
  conflicts: string[];
  timings_ms: Record<string, number>;
  diagnostics: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  product_line?: ProductLine | null;
  limit?: number;
}

/** A failure the user needs to understand, not a stack trace. */
export interface ApiError {
  kind:
    | "unreachable"
    | "engine_unavailable"
    | "engine_failed"
    | "invalid"
    /** The deployment has no answer model wired up at all. Search still works. */
    | "generation_unavailable"
    | "unknown";
  message: string;
  detail?: string;
}

/* -------------------------------------------------------------------------- *
 * POST /api/answer
 * -------------------------------------------------------------------------- */

/**
 * What happened to one answer attempt.
 *
 * The three failure modes are deliberately distinct and the UI must not merge
 * them. `abstained` means the knowledge base has no answer. `generation_failed`
 * means the evidence was there and the model server was not. `empty_query` means
 * nothing was asked. All three arrive as HTTP 200, so the UI branches on this
 * field and not on a status code.
 */
export type AnswerStatus =
  | "answered"
  | "clarification_needed"
  | "abstained"
  | "empty_query"
  | "generation_failed";

/**
 * Who wrote a step's wording.
 *
 * `model` was generated for this question. `prerequisite` is the procedure's own
 * sentence, inserted by the server because a cited step declares it depends on
 * that step - the model never chose it, so it is shown verbatim.
 */
export type StepSource = "model" | "prerequisite";

export interface LocalizedDescription {
  en: string;
  ar: string;
}

export interface HighlightBox {
  control_name: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ScreenshotRef {
  screenshot_id: string;
  url: string;
  media_type: string;
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  procedure_id: string;
  procedure_version: string;
  step_order: number;
  screen_name: string;
  control_name: string;
  description: LocalizedDescription;
  watermark: string;
  language: string;
  highlight: HighlightBox | null;
}

export interface AnswerStep {
  /** The model's wording, unless `source` is `prerequisite`. */
  instruction: string;
  procedure_id: string;
  procedure_version: string;
  step_id: string;
  /** Human readable, e.g. `NW-HTL-014 v2.1 · NW-HTL-014#S05`. */
  citation: string;
  source_section: string;
  control_name: string | null;
  conditions: string[];
  screenshot: ScreenshotRef | null;
  source: StepSource;
  /** Steps this one depends on. Explains why a prerequisite step is present. */
  requires: string[];
}

/** What the supplied case facts actually did. */
export interface ContextUsed {
  provided: string[];
  applied: Record<string, string>;
  /** Required information still worth asking the agent for. */
  unanswered: string[];
  /**
   * Required information this system refuses to ask for, mapped to why.
   * Reported rather than dropped, so a declined requirement stays visible.
   */
  not_collected: Record<string, string>;
  completed_steps: string[];
  product_line: string | null;
}

export interface AnswerResponse {
  status: AnswerStatus;
  query: string;
  detected_language: Language;
  /** For any non-answer status, this is the message to show. */
  summary: string;
  steps: AnswerStep[];
  conditions: string[];
  escalation: string | null;
  customer_wording: string | null;
  /** At most one, and only when the evidence does not settle the case. */
  clarifying_questions: string[];
  conflicts: string[];
  sources: string[];
  screenshots: ScreenshotRef[];
  evidence: Evidence;
  model_called: boolean;
  failure_detail: string | null;
  dropped_citations: string[];
  /** Steps the server added because a cited step declared it depends on them. */
  inserted_prerequisites: string[];
  context_used: ContextUsed;
  timings_ms: Record<string, number>;
  retrieval_metadata: Record<string, unknown>;
}

export type BookingStatus =
  | "confirmed"
  | "pending"
  | "cancelled"
  | "modified"
  | "no_show"
  | "completed";

export type SupplierStatus =
  | "not_contacted"
  | "unreachable"
  | "confirmed_available"
  | "cannot_accommodate"
  | "no_record_of_booking"
  | "disputed";

export type PaymentStatus =
  | "not_paid"
  | "prepaid"
  | "pay_at_property"
  | "refund_requested"
  | "refund_issued"
  | "chargeback";

/**
 * Non-sensitive case facts, mirroring `agentpath.case_context.CaseContext`.
 *
 * There is no name, booking reference, phone number or payment detail here, and
 * that is the contract rather than an omission: the API declares `extra="forbid"`,
 * so sending one is rejected with HTTP 422 and the field never exists to be
 * logged. Counts are allowed because a count identifies nobody.
 */
export interface CaseContextInput {
  product_type?: string | null;
  booking_status?: BookingStatus | null;
  /** ISO date, `YYYY-MM-DD`. A date, not a timestamp. */
  service_date?: string | null;
  supplier_status?: SupplierStatus | null;
  payment_status?: PaymentStatus | null;
  customer_arrived?: boolean | null;
  party_size?: number | null;
  rooms_booked?: number | null;
  /** Citation tokens such as `NW-HTL-014#S03`. Anything else is ignored. */
  actions_already_taken?: string[];
  error_code?: string | null;
  market?: string | null;
  language?: "ar" | "en" | null;
}

export interface AnswerRequest {
  query: string;
  product_line?: ProductLine | null;
  limit?: number;
  case_context?: CaseContextInput | null;
}

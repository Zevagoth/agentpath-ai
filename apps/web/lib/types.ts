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
  kind: "unreachable" | "engine_unavailable" | "engine_failed" | "invalid" | "unknown";
  message: string;
  detail?: string;
}

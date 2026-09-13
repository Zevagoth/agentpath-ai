# Stage 2A — grounded answer vertical slice (NW-HTL-014)

Stage 2 is split into three gated parts. This document covers **2A only**.

| Part | Scope | Gate |
| --- | --- | --- |
| **2A** | One complete vertical slice for `NW-HTL-014`: query → retrieval → evidence gate → local answer → cited steps → instructional screenshots → rendered UI | Acceptance criteria below, then stop and report |
| 2B | Expand to 12+ procedures, held-out evaluation split, full metric set | Not started until 2A is accepted |
| 2C | Containerise api/web, GitHub Actions (triggers disabled) | Not started until 2B is accepted |

Stage 1 is closed and must keep working. `POST /api/search` is a frozen contract:
additive response fields only.

## Standing constraints

- Branch `stage2-grounded-assistant`. Local commits only. **No push without explicit permission.**
- Everything binds to `127.0.0.1`. No LAN exposure.
- All content synthetic: fictional "Northwind Travel" / "AgentDesk". Every generated
  screenshot must visibly carry `FICTIONAL — SYNTHETIC DEMO DATA`.
- No cloud AI. Generation runs on local Ollama.
- No `sudo`. No driver, firewall, Docker-daemon, kernel or system-package changes.
- No model download above 5 GB without reporting the exact size and asking first.
- No OCR. The screenshot generator's own DOM text is the source of truth for
  screenshot text; a vision model may be used only as an optional cross-check,
  never to produce data the system relies on.
- The reranker stays out of this stage. No further CUDA-in-Docker attempts.
- Retrieval accuracy from the 5-procedure development corpus is never presented as
  production accuracy.

## Task list

### T0 — Verify and branch
Confirm `HEAD`, clean worktree, and running services. Record baseline test counts
from `uv run pytest -m "not integration"` and `cd apps/web && npm run test` before
touching anything. Create `stage2-grounded-assistant` off `main`.

### T1 — Gate 1: a usable browser
Detect an installed chromium-family executable. Prefer it over another 173 MB
Playwright download; the previous download stalled at ~8 MB of 173.7 MiB.

- Support `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`, consumed by both
  `apps/web/playwright.config.ts` and the screenshot generator.
- No machine-specific path in any committed file.
- Add a diagnostic that prints which browser will be used and why.
- If no usable browser exists and fixing it needs `sudo`, stop and report the exact
  blocker.

### T2 — Gate 2: deterministic synthetic screenshots
Three screens already referenced by `NW-HTL-014.yaml`:
`agentdesk-booking-detail-status` (step 1), `agentdesk-supplier-log` (step 4),
`agentdesk-relocation-finder` (step 5). Only these three steps genuinely need
visual navigation.

- Committed HTML + CSS source, no external font/image/API/network dependency.
- Fixed 1440×900 viewport, `deviceScaleFactor` pinned, self-contained font stack.
- Visible watermark. Realistic but entirely fictional console content.
- Highlight box and arrow placed clear of the text they refer to.
- Generator emits the PNG **and** a sidecar JSON containing the page's extracted
  DOM text plus the structured description.
- Determinism is asserted as: identical dimensions, identical DOM text, and a
  perceptual hash within a stated tolerance. **Not** byte-identical PNG.

### T3 — Visual-step data model
Extend without breaking the existing 5 documents. The 31-chunk baseline was
re-recorded at **33 chunks** once the step id entered the rendered text:
NW-CXL-032 6, NW-CXL-045 6, NW-HTL-014 8, NW-HTL-021 7, NW-REF-018 6.

- `Step` in `services/ingestion/agentpath_ingestion/schema.py` is
  `extra="forbid"`, so `step_id` must be added to the model, not just the YAML.
- Visual-step record fields: `procedure_id`, `procedure_version`, `step_id`,
  `step_order`, `screen_name`, `control_name`, `instruction`, `image_path`,
  `image_description`, `dom_text`, `language`, highlight bounding box.
- `ocr_text` is carried in the schema for compatibility but is not populated and
  is not depended on.
- Changing rendered Markdown changes `RenderedProcedure.sha256`, which makes
  `_has_changed` delete and re-upload the document. Expected; it moved the
  baseline from 31 to 33 chunks, re-recorded above.
- Enrichment happens at ingestion time only. Nothing in this path may run during a
  user search or answer request.

### T4 — RAGFlow chunk-with-image support
`RAGFlowClient` has no chunk-creation method today. Add `add_chunk` and
`update_chunk` against
`POST|PATCH /api/v1/datasets/{ds}/documents/{doc}/chunks[/{chunk_id}]`.

Verified in the pinned source (`.external/ragflow/api/apps/restful_apis/chunk_api.py`):

- `image_base64` is accepted on both routes (l.1138 add, l.1289 update).
- Validation at l.82: must be a non-empty string of raw standard base64.
  A `data:image/png;base64,` prefix fails `validate=True`.
- Storage: MinIO bucket = `dataset_id`, object key = `chunk_id`;
  `img_id = "{dataset_id}-{chunk_id}"`, `doc_type_kwd = "image"`.
- `chunk_id = xxhash64(content + document_id)` — deterministic on content.
- **Hazard:** `store_chunk_image` concatenates onto an existing object and
  re-saves as JPEG. Re-posting identical content for the same document stacks
  images rather than replacing. Must be handled.
- **Untested:** whether re-parsing a document wipes manually added chunks. Test
  this before relying on it; it collides with `_replace` and `--force`.

Prove it with an integration test using a small synthetic PNG. If it does not work
on our deployment, fall back to serving images from the repo and record why.

### T5 — Ingest the visual steps
Attach one image-bearing chunk per visual step after document parsing completes.
Chunk content is the canonical step text plus screen, control and description, so
the chunk is retrievable on its own terms. Record the chunk ids in the build
manifest for idempotence.

### T6 — Screenshot serving route
`GET /api/v1/documents/images/<id>` needs a RAGFlow credential, so the browser
cannot fetch it. Serve screenshots from the AgentPath API instead.

- Resolve by id against a known set. Never join request input onto a filesystem
  path.
- Never expose an absolute path such as `/home/...` in any response.
- Path-traversal tests are mandatory.
- `paths.py` gains a `screenshots_dir()` helper alongside the existing ones.

### T7 — Model selection: `qwen3.5:4b`, on evidence
Start with `qwen3.5:4b` (3.4 GB, q4_K_M, 256K context available but not used).
Adopt it only if it earns it.

- Check what is installed first; report exact sizes before any pull.
- Runtime settings: `num_ctx=8192`, `think=false`, low temperature, structured
  JSON output. Do not pass the full 256K context. Do not pass evidence the answer
  does not need.
- Acceptance harness measuring: Arabic accuracy, English accuracy, JSON schema
  compliance rate, citation validity, hallucination/unsupported-claim count,
  latency, VRAM use.
- If it fails these criteria, **stop and present the numbers** before proposing
  `qwen3.5:9b` (6.6 GB, above the download gate).
- Model id stays configurable via `OLLAMA_CHAT_MODEL`.
- The embedding model does not change.

### T8 — Grounded answer service
- Evidence gate runs first. On `INSUFFICIENT`, return the exact existing message —
  `No confirmed procedure was found in the available knowledge base.` — **without
  calling Ollama**. Expose whether the model was called so tests can prove it.
- Blank query is a third case: nothing was searched, so it is not an abstention
  about the knowledge base.
- Retrieved content is untrusted evidence, never instructions. Instructions
  embedded in retrieved documents are ignored.
- The model may use only the supplied evidence. Never invent refund amounts,
  compensation, deadlines, approval requirements, policy exceptions, buttons or
  escalation paths.
- Every factual instruction cites a procedure and a step or section. Citations are
  validated against the evidence set; invented citations are repaired or the step
  is dropped. If nothing valid survives, abstain.
- Partial or ambiguous evidence produces a focused clarifying question, not a guess.
- Conflicting active procedures are surfaced with versions, never silently resolved.
- Answer language matches the user; English UI control labels stay in English.
- Hidden prompts, raw reasoning, credentials and infrastructure detail are never
  exposed.
- Generation failure is a distinct state from abstention.

### T9 — `case_context` contract (design only)
Optional, extensible, forward-looking. **No real booking system, no customer data.**

Optional non-sensitive fields: `product_type`, `booking_status`, `service_date`,
`supplier_status`, `payment_status`, `customer_arrived`, `actions_already_taken`,
`error_code`, `market`, `language`.

- Used to narrow procedures, pick the right decision branch, and name the
  information still missing.
- Response carries `context_used` so the effect is visible.
- It must never override procedure text or invent policy.
- Never store customer names, real booking references, phone numbers, payment data
  or any personal information.
- Any future real integration requires an official API, read-only scope, company
  approval, authentication, authorisation, audit logging and data minimisation.
  Documented as a requirement, not implemented.
- Tested with fabricated data only. Must not delay the vertical slice.

### T10 — `POST /api/answer`
New route registered in `create_app` beside the existing handlers, obtaining
services from `request.app.state.context` like `/api/search` does.

Response contains at least: `status`, `detected_language`, `summary`, `steps`,
`conditions`, `do`, `dont`, `escalation`, `customer_wording`,
`clarifying_questions`, `sources`, `timings`, `retrieval_metadata`,
`context_used`. Each step carries `instruction`, `procedure_id`,
`procedure_version`, `step_id`, a human-readable citation, and an optional
screenshot reference.

Timings measured separately: query processing, retrieval, evidence validation,
prompt construction, model generation, total.

Targets: search p95 stays under 2 s; full local answer p95 preferably under 10 s
warm; unsupported queries abstain without waiting for the model. Report the real
numbers either way. Non-streaming first; SSE only after everything passes.

### T11 — UI
Extend the existing console; search must keep working.

Answer summary card, ordered steps, the relevant screenshot beside its step,
click-to-enlarge, procedure id / version / effective date / step id / source
passage, Do and Don't, escalation, suggested customer wording with a copy button,
clarification state, abstention state, a separate service-error state, loading and
generation-progress states, response-time detail, and Helpful / Not helpful
feedback with an optional reason.

- Feedback is stored locally and does not record full customer queries by default.
- Arabic RTL and English LTR behaviour preserved.
- Replace `dangerouslySetInnerHTML` in `ProcedureCardView.tsx:59` with a parser
  that emits safe nodes for the intended emphasis markup only.

### T12 — Tests
Fake Ollama responses; Ollama error and timeout handling; structured-output
validation; invented citations; missing evidence; abstention without an LLM call;
Arabic and English answers; prompt injection in both the query and the retrieved
content; conflicting versions; screenshot metadata; screenshot route security and
path traversal; deterministic screenshot generation; `/api/search` backward
compatibility; UI answer, screenshot, clarification, abstention and error states;
live integration against the local stack; Playwright behaviour if T1 succeeded.

Unit tests must not download models or browsers. Test counts in documentation must
be generated or mechanically verifiable so they cannot drift again.

### T13 — Report, docs, commit
Vertical-slice report with real example responses and measured timings. Update
README, architecture, evaluation, security, troubleshooting, environment example
and execution plan. Every documented command must work; no fake Makefile targets
and no broken links. Commit as a coherent local checkpoint, then stop.

## Acceptance criteria

1. An English hotel-overbooking query retrieves `NW-HTL-014`.
2. The Arabic equivalent retrieves the same procedure.
3. `overboking at the hotle` still retrieves the correct procedure.
4. The generated answer contains only supported steps.
5. Every answer step has a valid citation.
6. Relevant screenshots appear beside the correct steps.
7. The screenshot carries the fictional-data watermark.
8. The unsupported password-reset query abstains.
9. That query does not call Ollama.
10. A prompt-injection-style query cannot override the grounding rules.
11. A simulated Ollama outage is distinguishable from "no procedure found".
12. `/api/search` remains backward-compatible.
13. Existing tests still pass.
14. New backend and frontend tests pass.
15. The Next.js production build and TypeScript check pass.
16. A real Playwright browser test passes, if T1 found a usable browser.
17. No service restart loops; resource use stays reasonable.

Then stop and report before any 2B work.

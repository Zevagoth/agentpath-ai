# AgentPath AI - implementation plan and progress

> All knowledge-base content in this repository is **fictional and synthetic**.
> See [SECURITY.md](../SECURITY.md).

## Product goal

An agent-facing assistant that turns a free-text customer-service case (Arabic or
English) into the correct internal procedure, using hybrid lexical + semantic
retrieval, grounded answers with citations, the original procedural screenshots,
and explicit abstention when the knowledge base has no answer.

## Architecture as built

| Concern | Choice | Note |
| --- | --- | --- |
| Document/RAG engine | RAGFlow v0.27.2 (Docker) | Parsing, chunking, hybrid retrieval, citation highlights |
| Doc engine | Elasticsearch | BM25 term scoring beside kNN |
| Embeddings | `BAAI/bge-m3` via a TEI container, CPU | Same model as planned; TEI instead of Ollama because ufw blocks bridge-to-host traffic, so the model service lives inside RAGFlow's network |
| Reranking | `BAAI/bge-reranker-v2-m3` via a TEI container | Registered and code-live, **disabled** behind `AGENTPATH_ENABLE_RERANK`; see [TROUBLESHOOTING](TROUBLESHOOTING.md) |
| Answer model | `qwen3.5:4b` via host Ollama | Called from the API over loopback, so it needs no container networking. 34/34 layers on the GPU; `qwen2.5:7b-instruct` was a placeholder default and was never used |
| API | FastAPI, Python 3.12 via uv | |
| Web | Next.js 15 + React 19, plain CSS | RTL/LTR per field |
| Tests | Pytest, Vitest, Playwright | 614 backend, 2 integration, 20 web unit, 5 live UI, 5 browser specs |

Replaceability: application code depends on the `Retriever` port in
`agentpath.retrieval.ports`. `RAGFlowRetriever` is the only production
implementation.

## Session 1 scope

Search and evaluation only. **No answer generation and no image understanding**
until retrieval is working and measured. That condition is now met.

## Progress

### Task 1 - Environment verification and repository skeleton `[x]`

- `scripts/env_check.py` with hard gates (disk, RAM, Docker/Compose, `vm.max_map_count`,
  ports, GPU headroom, Ollama LAN exposure); 28 tests
- uv workspace on CPython 3.12; `.gitignore`, `.env.example`, `Makefile`
- Findings that shaped everything after: port 8080 occupied → RAGFlow UI on 8081;
  only CPython 3.14 installed system-wide → workspace pins 3.12; `vm.max_map_count`
  already sufficient → no sudo needed anywhere

### Task 2 - RAGFlow v0.27.2 stack `[x]`

- `scripts/bootstrap_ragflow.sh` clones at the pinned tag, renders `docker/.env`
  from upstream + `infra/ragflow/env.agentpath`, generates secrets into `.env.local`
- Guards refuse to start on a shipped default password or a non-loopback port
- `scripts/ragflow_health.sh` reports per-service health and audits exposure

### Task 3 - Embedding and reranking path `[x]`

- Two TEI sidecars on RAGFlow's network; `scripts/configure_ragflow_models.py`
  provisions the operator account, API key, providers and tenant defaults
- Verified: 1024-dim embeddings; cross-lingual cosine **0.760** for the same
  meaning in Arabic and English against **0.523** for an unrelated phrase; the
  reranker scores the relevant passage 0.62 and the others 0.00
- Reranking left off: 20-22 s per query on CPU, and an unresolved CUDA
  initialization failure inside Docker on GPU

### Task 4 - Synthetic procedures `[x]`

- Five Northwind Travel procedures with bilingual titles, synonyms, conditions,
  numbered steps carrying AgentDesk screen and control names, decision branches,
  do/do-not, exceptions, escalation with approver, customer wording, version
- `data/synthetic/lexicon.yaml`: typos, abbreviations, bilingual concepts
- Guards: no real brand name anywhere under `data/`, no currency amounts

### Task 5 - Ingestion into RAGFlow `[x]`

- `agentpath-ingest build|ingest|verify`, idempotent via a content hash in
  document metadata, parse polling with RAGFlow's own progress messages
- Verified: 5 documents, 31 chunks, 32 s (Stage 2A T3 re-recorded this at 33
  chunks once step ids entered the rendered text)

### Task 6 - Retriever port, RAGFlowRetriever, query understanding `[x]`

- Script-based language detection, Arabic normalization, lexicon-driven correction
  and expansion, scenario tags, product-line classification
- Chunk-to-procedure aggregation; a procedure scores as its best passage
- One abstention rule in `agentpath.evidence`, shared by API, CLI and benchmark
- 36 tests against recorded RAGFlow payloads

### Task 7 - Search API `[x]`

- `POST /api/search`, `GET /health`; structured JSON logging with request ids
- Engine outage is 503 and never resembles "no procedure found"
- 20 tests with a fake retriever

### Task 8 - Benchmark `[x]`

- Three modes over 30 queries (26 supported, 4 unsupported), scored separately
- Result: keyword baseline 85% R@1 → raw engine 92% → full path **100%**, with
  false abstention 8% → 12% → **0%**, p95 **169 ms**
- Abstention threshold derived from the observed distribution, with the thin
  0.034 gap documented as a limitation
- 50 tests

### Task 9 - Web console `[x]`

- Case field with automatic RTL/LTR, product-line filter, example chips
- Distinct loading, error, abstention and results states
- Top 3 procedures with rank, ID, version, effective date, score, matched-section
  count, and source passages with term/vector score breakdown
- Applied corrections and expansions shown back to the agent
- Round-trip and server timings displayed
- 20 unit tests, 5 live tests against the running stack, 5 Playwright specs

### Task 10 - Documentation `[x]`

- README (run instructions, measured results), SECURITY, EVALUATION,
  TROUBLESHOOTING, this plan
- `tests/test_repo_hygiene.py` keeps them honest: no tracked secrets, loopback-only
  ports, every Makefile recipe resolving to a module or file that exists, README
  links resolving, the abstention wording identical in code and docs
- `docs/ENVIRONMENT.md` is generated in a shareable form: requirement satisfaction
  and major tool versions only, with no host identity, paths, hardware models or
  exact capacities. A test enforces the redaction; the full detail goes to the
  gitignored JSON output

## Known gaps at the end of session 1

1. **Reranking is unmeasured on GPU.** Unresolved CUDA initialization failure
   inside Docker; the flag exists and the model is registered.
2. **Playwright browser download stalls on this host** during extraction. The
   specs are committed; the same five assertions run through
   `make test-web-live` against the real stack in jsdom.
3. **The abstention threshold is fitted to the reported query set** (0.034 gap).
4. **Five procedures.** Retrieval quality will move as the corpus grows.
5. **No CI yet.** Test counts are local.

## Stage 2

Split into three gated parts. See `docs/STAGE2A_PLAN.md` for the full task list,
the measured findings and the known limitations.

**2A — the `NW-HTL-014` vertical slice.** In progress on branch
`stage2-grounded-assistant`.

1. ✅ Playwright usable without its own browser download (`make browser-check`).
2. ✅ Deterministic screenshot generation: committed HTML/CSS AgentDesk screens
   rendered by Playwright, watermarked, verified by DOM text plus a perceptual hash
   rather than byte equality.
3. ✅ Visual-step data model with a stable `step_id`, strict on stale captures.
4. ✅ Image-bearing RAGFlow chunks (`image_base64`), idempotent against RAGFlow's
   image-concatenation behaviour.
5. ✅ Screenshot serving route, resolved against a catalogue built at startup so no
   request input reaches a path operation.
6. ✅ Grounded answer generation on **`qwen3.5:4b`** (not `qwen2.5:7b-instruct`,
   which was a placeholder default): evidence rebuilt from the structured source,
   `enum`-constrained citations, server-attached attribution, clarification,
   conflict surfacing, and the exact abstention message without an LLM call.
7. ⬜ `POST /api/answer`.
8. ⬜ `case_context` contract, design only.
9. ⬜ Console: answer card, screenshots beside their steps, do/don't, feedback.

**2B — expansion and evaluation.** Not started. 12+ procedures, 50+ queries with a
frozen held-out split, citation-correctness scoring.

**2C — packaging.** Not started. GitHub Actions with triggers disabled, then
containerise the API and web app.

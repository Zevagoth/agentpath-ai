# AgentPath AI - implementation plan and progress

> All knowledge-base content in this repository is **fictional and synthetic**.
> AgentPath AI is an independent proof of concept with no affiliation to any
> travel company. See [SECURITY.md](../SECURITY.md).

## Product goal

An agent-facing assistant that turns a free-text customer-service case
(Arabic or English) into the correct internal procedure, using hybrid lexical +
semantic retrieval with cross-encoder reranking, grounded answers with
citations, the original procedural screenshots, and explicit abstention when the
knowledge base has no answer.

## Architecture at a glance

| Concern | Choice | Why |
| --- | --- | --- |
| Document/RAG engine | RAGFlow v0.27.2 (Docker) | Parsing, OCR, chunking, hybrid retrieval and citations out of the box |
| Doc engine | Elasticsearch | Real BM25 lexical scoring next to kNN, plus `highlight` for citation snippets |
| Embeddings | `bge-m3` via Ollama | Strong Arabic + English; ~1.2 GB GGUF instead of TEI's ~21 GB load requirement |
| Reranker | `bge-reranker-v2-m3` in a CPU sidecar | Keeps the RTX 4060 free for the Stage 2 chat model; registered in RAGFlow as a `LocalAI` rerank provider |
| Answer model (Stage 2) | `qwen2.5:7b-instruct` via Ollama | Local, Arabic-capable; 3B fallback if measurements demand it |
| API | FastAPI (Python 3.12 via uv) | Typed, async, testable |
| Web | Next.js + TypeScript | RTL-aware agent console |
| Tests | Pytest, Vitest, Playwright | Backend, web units, end-to-end |

Replaceability: application code depends on the `Retriever` port in
`agentpath.retrieval.ports`. `RAGFlowRetriever` is the only production
implementation; the embedding model, reranker and LLM are configuration.

## Session 1 scope

Search and evaluation only. **No answer generation and no image understanding**
until hybrid retrieval is working and measured.

## Progress

Legend: `[x]` done and verified · `[~]` in progress · `[ ]` not started

### Task 1 - Environment verification and repository skeleton `[x]`

- [x] `scripts/env_check.py`: read-only gates (disk, RAM, Docker, Compose,
      `vm.max_map_count`, ports, GPU, Ollama LAN exposure)
- [x] Repository skeleton, `.gitignore`, `.env.example`, `Makefile`
- [x] uv workspace: `agentpath`, `agentpath-ingestion`, `agentpath-eval` on CPython 3.12
- [x] `tests/test_env_check.py` - 28 tests covering gates, parsing and rendering
- [x] Verified on this machine: verdict **READY** (see [ENVIRONMENT.md](ENVIRONMENT.md))

Findings that changed the plan:

- `vm.max_map_count` is already `1048576` - **no sudo step is needed**.
- Port `8080` is occupied on this host, so the RAGFlow UI is published on **8081**.
- Only CPython 3.14 is installed system-wide; the workspace pins **3.12** via uv
  to stay inside the supported wheel range.
- Ollama listens on `127.0.0.1:11434` only. Containers reach it through a
  bridge-only relay (`scripts/ollama_bridge.sh`), never via `0.0.0.0`.
- Free VRAM was 70 MiB during the check (a game was running), so embeddings run
  on CPU for now. This does not block the search slice.

### Task 2 - RAGFlow v0.27.2 stack `[ ]`

- [ ] `scripts/bootstrap_ragflow.sh`: clone at tag into `.external/ragflow`,
      render `docker/.env` from `infra/ragflow/env.template`
- [ ] Generated secrets into `.env.local` (`openssl rand -hex 32`), no shipped defaults
- [ ] All published ports remapped to `127.0.0.1`
- [ ] `scripts/ragflow_health.sh`
- [ ] RAGFlow API key acquired and stored in `.env.local`

### Task 3 - Embedding and reranking path `[ ]`

- [ ] `bge-m3` pulled into Ollama
- [ ] Ollama bridge relay bound to the Docker bridge interface only
- [ ] Reranker sidecar (`bge-reranker-v2-m3`, CPU) on the RAGFlow network
- [ ] Providers registered in RAGFlow, tenant defaults set
- [ ] Verified: embedding dimensions and rerank score ordering

### Task 4 - Five synthetic procedures `[ ]`

- [ ] `data/synthetic/procedures/*.yaml` with schema validation
- [ ] Markdown renderer with the `FICTIONAL - SYNTHETIC DEMO DATA` banner
- [ ] `data/synthetic/lexicon.yaml` (Arabic + English synonyms, scenario rules)
- [ ] Brand-guard test

### Task 5 - Ingestion into RAGFlow `[ ]`

- [ ] `agentpath-ingest build|ingest|status|verify`
- [ ] Dataset with `bge-m3@Ollama`, metadata fields for filtering
- [ ] Idempotent re-ingestion, parse polling

### Task 6 - Retriever port, RAGFlowRetriever, query understanding `[ ]`

- [ ] Typed domain models and the `Retriever` port
- [ ] Hybrid retrieval call (`keyword=true`, vector weight, `rerank_id`, `metadata_condition`)
- [ ] Language detection, Arabic normalization, lexicon expansion, scenario tags
- [ ] Chunk to procedure aggregation, evidence grading

### Task 7 - FastAPI search endpoint `[ ]`

- [ ] `POST /api/search`, `GET /health`
- [ ] Structured logging with per-stage latency, loopback binding

### Task 8 - Lexical baseline and evaluation harness `[ ]`

- [ ] Benchmark-only lexical retriever
- [ ] `evaluation/queries.yaml` (>= 10 typed queries incl. Arabic, typos, unsupported)
- [ ] Recall@1, Recall@3, MRR, mean and p95 latency, abstention correctness
- [ ] Markdown + JSON report

### Task 9 - Minimal web UI and session report `[ ]`

- [ ] Next.js search page with RTL support and evidence badge
- [ ] Vitest units, Playwright end-to-end
- [ ] README, ARCHITECTURE, SECURITY, EVALUATION
- [ ] Session report

## Stage 2 (not this session)

1. Deterministic screenshot pipeline (HTML/CSS templates rendered by Playwright).
2. Expand to 12+ procedures with screenshot steps; ingest images as chunks with
   OCR text, step metadata and structured descriptions, computed at ingestion.
3. Grounded answer generation with citations, conflict handling and the exact
   abstention message.
4. Full agent console: answer card, screenshots beside steps, Do/Don't,
   escalation, feedback buttons.
5. Full evaluation: 50+ queries, citation correctness, latency targets.
6. Containerize `apps/api` and `apps/web` for one-command startup.

# Security and data handling

AgentPath AI is an independent proof of concept. This document states what it
does, what it deliberately does not do, and where the sharp edges are.

## The knowledge base is entirely invented

Every procedure in `data/synthetic/` describes a fictional online travel agency
called **Northwind Travel** and a fictional agent console called **AgentDesk**.

- No real company policy, wording, screenshot, customer record or trademark is
  reproduced anywhere.
- Every source file and every rendered document carries the banner
  `FICTIONAL — SYNTHETIC DEMO DATA`.
- Procedure references (`NW-HTL-014`) and queue names are invented.
- Automated guards enforce this: one test fails if any real travel brand name
  appears anywhere under `data/`, and another fails if a currency amount appears in
  a procedure, so the corpus cannot start implying real compensation limits.

Nothing in this repository was obtained by scraping, probing or accessing any
company system, and no such access is performed at runtime.

## Nothing leaves the machine

| Component | Where it runs |
| --- | --- |
| RAGFlow, Elasticsearch, MySQL, MinIO, Redis | Local Docker containers |
| Embedding model (`bge-m3`) | Local container, CPU |
| Rerank model (`bge-reranker-v2-m3`) | Local container, currently disabled |
| API and web app | Local processes |

No query, document or embedding is sent to a hosted AI service. Two RAGFlow
features that *would* call a chat model are switched off on purpose:

- `keyword` on the retrieval endpoint (LLM keyword extraction),
- `cross_languages` (LLM query translation).

Both are replaced by the deterministic lexicon in `data/synthetic/lexicon.yaml`.
Outbound network access happens only during setup, to pull Docker images and model
weights from their public registries.

## Network exposure

Every published port binds to `127.0.0.1`:

| Port | Service |
| --- | --- |
| 3000 | Web console |
| 8000 | AgentPath API |
| 8081 | RAGFlow UI |
| 9380, 9381, 9382 | RAGFlow API, admin, MCP |
| 1200 | Elasticsearch |
| 3307 | MySQL |
| 9000, 9001 | MinIO |
| 6379 | Redis |
| 7997, 7998 | Reranker, embedder |

Two automated guards keep it that way:

- `scripts/bootstrap_ragflow.sh` refuses to render a compose environment where any
  published port is not prefixed with `127.0.0.1`.
- `scripts/ragflow_health.sh` audits the running containers and fails if anything
  is published on `0.0.0.0` or `::`.

The model services live on RAGFlow's own Docker network and are reachable by
service name from RAGFlow. They are not exposed to the host beyond loopback, and
no firewall rule was added to make anything work — ufw is active on the reference
machine and stays untouched.

### The API has no authentication

`POST /api/search` and `GET /health` are unauthenticated. This is acceptable for a
local proof of concept **only** because the service binds to loopback. Before this
is exposed on any routable interface, it needs an authentication layer, request
rate limiting and an audit trail. Treat "bind to 0.0.0.0" as a change that
requires those first.

## Secrets

- Real values live only in `.env.local`, which is gitignored and written with mode
  `600`.
- `.env.example` contains placeholders. A test asserts that no `.env` file other
  than the example is tracked by git.
- `scripts/bootstrap_ragflow.sh` generates each password with
  `openssl rand -hex 16` and reuses existing values on re-runs so data already
  written to the volumes stays reachable.
- RAGFlow ships identical default passwords for MySQL, MinIO, Redis and
  Elasticsearch. The bootstrap script **refuses to start** if any effective
  password is still a shipped default.
- The RAGFlow operator account and API key are created by
  `scripts/configure_ragflow_models.py` and stored in `.env.local`; neither is ever
  printed to the console.
- No credential is committed, and the git history was checked for the generated
  API key and passwords before the repository was pushed.

### If you fork or share this

1. Do not copy `.env.local`.
2. Run `make ragflow-bootstrap` to generate fresh secrets.
3. Set `REGISTER_ENABLED=0` in `infra/ragflow/env.agentpath` after creating your
   operator account; `make ragflow-health` warns while registration is still open.
4. Change the RAGFlow admin password (`ADMIN_DEFAULT_PASSWORD`).

## Grounding and abstention

The product-level safety property is that the system does not invent procedure.

- Retrieval returns only passages that exist in the indexed corpus.
- Every returned procedure carries its citations; the API schema has no path that
  returns a recommendation without them, and a test enforces it.
- When evidence is insufficient the system answers exactly: *"No confirmed
  procedure was found in the available knowledge base."*
- "Engine unavailable" (HTTP 503) is kept distinct from "nothing found" (HTTP 200
  with an abstention). Conflating them would teach agents to trust an empty result
  when the search never ran.
- Suggested customer wording is labelled in the source documents as carrying no
  policy weight, so generated phrasing is never mistaken for retrieved policy.

Answer generation is not implemented yet. When it is, the same rule applies: cite
every procedural claim, and never state a limit, amount or approval rule that is
not in a retrieved passage.

## Logging

Structured JSON logs record request ids, procedure ids, evidence levels and
per-stage latency. **Query text is not logged by default** (`AGENTPATH_LOG_QUERY_TEXT=false`),
because a real case description would contain personal detail even though this
corpus is synthetic.

## Known gaps

1. No authentication or rate limiting on the API (loopback-only mitigates it).
2. RAGFlow's own UI on port 8081 uses the account created during setup; its
   session handling is RAGFlow's, not ours.
3. No TLS anywhere. Everything is loopback HTTP.
4. Container images and model weights are pulled from public registries at setup
   time; their integrity rests on those registries.
5. `docs/ENVIRONMENT.md` records host details (CPU, GPU, kernel, paths). Review it
   before sharing the repository publicly.

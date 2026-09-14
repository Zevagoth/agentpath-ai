# AgentPath AI

**Describe the case. Find the right path.**

A local-first assistant for customer-service agents who need the correct procedure
out of a large knowledge base. Keyword search fails when an article is titled
"Property unable to accommodate a confirmed guest" and the agent types "hotel has
no room", "overboking at the hotle", or "الفندق ما عنده غرف". AgentPath AI
retrieves by meaning *and* by exact term, in Arabic and English, and grounds every
recommendation in the source procedure.

> **All knowledge-base content in this repository is fictional and synthetic.**
> The procedures describe an invented online travel agency, *Northwind Travel*,
> and an invented agent console, *AgentDesk*. AgentPath AI is an independent proof
> of concept with no affiliation to, and no data from, any real company. See
> [SECURITY.md](SECURITY.md).

## Status

The search slice is working end to end: browser → API → RAGFlow, with query
understanding, hybrid retrieval, evidence grading and abstention. Answer
generation and screenshot/OCR handling are deliberately **not** started until this
is measured, per the plan in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

Measured on the 30-query benchmark ([EVALUATION.md](EVALUATION.md)):

| Mode | Recall@1 | Recall@3 | MRR | Abstention | False abstention | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword baseline | 85% | 92% | 0.885 | 50% | 8% | <1 ms |
| RAGFlow hybrid (raw query) | 92% | 96% | 0.942 | 100% | 12% | 154 ms |
| **AgentPath (full path)** | **100%** | **100%** | **1.000** | **100%** | **0%** | **169 ms** |

## Requirements

- Linux, Docker ≥ 24 with Compose ≥ 2.26.1
- ~30 GB free disk, ≥ 16 GB RAM
- [uv](https://docs.astral.sh/uv/) (manages CPython 3.12 for this repo)
- Node.js ≥ 20
- NVIDIA GPU optional: everything in the search slice runs on CPU

## Running it

```bash
# 0. Verify the machine (read-only, downloads nothing)
make check

# 1. Python workspace and web dependencies
make install
make install-web

# 2. RAGFlow stack: clones v0.27.2, generates secrets, starts containers
#    First run pulls ~4 GB of images and ~2.3 GB of model weights.
make ragflow-bootstrap
make ragflow-up
make ragflow-health

# 3. Register the embedding and rerank models, then verify them
make models

# 4. Build and ingest the synthetic knowledge base
make dataset
make ingest

# 5. Run the API and the console (separate terminals)
make api      # http://127.0.0.1:8000  (OpenAPI at /docs)
make web      # http://127.0.0.1:3000
```

Then open <http://127.0.0.1:3000> and try:

| Query | Expected |
| --- | --- |
| `hotel has no room for the guest` | NW-HTL-014 first |
| `الفندق ما عنده غرف والضيف واصل` | NW-HTL-014 first, field flips to RTL |
| `overboking at the hotle` | corrected to overbooking/hotel, NW-HTL-014 first |
| `how do I reset my email password` | explicit abstention, no procedure |

Everything binds to `127.0.0.1`. Ports: web 3000, API 8000, RAGFlow UI 8081,
RAGFlow API 9380, embedder 7998, reranker 7997.

## Testing

```bash
make test              # backend suite, no stack required
make test-web          # web unit suite, no stack required
make test-all          # both of the above
make test-web-live     # real UI components against the running API
make test-e2e          # Playwright in Chromium (needs: make install-web-browsers)
make eval              # the benchmark; writes evaluation/reports/
```

Counts as of the last run:

| Suite | Command | Tests | Needs the stack |
| --- | --- | --- | --- |
| Backend | `make test` | 614 | no |
| Backend integration | `make test-integration` | 2 | yes |
| Web unit | `make test-web` | 20 | no |
| Live UI | `make test-web-live` | 5 | yes |
| Browser (Playwright) | `make test-e2e` | 5 specs | yes, plus a browser |

Playwright no longer needs its own 174 MB download. Point it at a Chromium already
on the machine and `make browser-check` will report which binary it resolved and
why:

```bash
export PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
make browser-check
```

Unset, the behaviour is unchanged: the downloaded build is used, and
`make install-web-browsers` fetches it.

## How it works

```
apps/web            Next.js console: case input, product filter, RTL/LTR,
                    evidence badge, citations, timings
apps/api            FastAPI: query understanding -> Retriever port -> aggregation
                    -> evidence grading -> abstention
services/ingestion  Renders the synthetic procedures and ingests them into RAGFlow
evaluation          Benchmark: keyword baseline vs raw engine vs full path
data/synthetic      Fictional procedures (YAML source of truth) and the lexicon
infra/ragflow       Compose overrides: model sidecars, loopback-only ports
scripts             Environment gate, stack bootstrap, model wiring, measurement
docs                Architecture, environment, troubleshooting, plan
```

The search pipeline:

1. Detect language by script (Arabic ratio), no model.
2. Normalize: Arabic diacritics, alef/ya/ta-marbuta folding, digits; English case
   and accents.
3. Correct spelling and expand shorthand from `data/synthetic/lexicon.yaml`.
4. Match concepts, tag the scenario, classify the product line.
5. Send the agent's own words first, plus English equivalents, to RAGFlow.
6. Hybrid retrieval: BM25 term scoring blended with vector similarity.
7. Group chunks into procedures; a procedure scores as its best passage.
8. Grade the evidence and abstain when it is insufficient.

Retrieval sits behind the `Retriever` port in `agentpath.retrieval.ports`, so the
engine is replaceable without touching the application.

## Documentation

| Document | Contents |
| --- | --- |
| [EVALUATION.md](EVALUATION.md) | Benchmark method, results, thresholds, limitations |
| [SECURITY.md](SECURITY.md) | Data, secrets, network posture, what this does not do |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Task-by-task progress |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Every problem hit, with what fixed it |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Generated environment report |

## Security posture

Every service binds to `127.0.0.1`. Secrets live only in `.env.local`, which is
gitignored; `.env.example` carries placeholders. The model services run inside
RAGFlow's Docker network and publish nothing beyond loopback. Nothing in this
repository connects to a third-party AI service or to any company system.

**The API has no authentication.** That is acceptable only because nothing listens
on a routable interface. Details in [SECURITY.md](SECURITY.md).

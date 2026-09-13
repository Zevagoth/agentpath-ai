# AgentPath AI

**Describe the case. Find the right path.**

A local-first assistant for customer-service agents who need the correct
procedure out of a large knowledge base. Keyword search fails when an article is
titled "Property unable to accommodate a confirmed guest" and the agent types
"hotel has no room", "overbooking", or the Arabic equivalent. AgentPath AI
retrieves by meaning *and* by exact term, in Arabic and English, and grounds
every recommendation in the source procedure.

> **All knowledge-base content in this repository is fictional and synthetic.**
> The procedures describe an invented online travel agency, *Northwind Travel*,
> and an invented agent console, *AgentDesk*. AgentPath AI is an independent
> proof of concept with no affiliation to, and no data from, any real company.
> See [SECURITY.md](SECURITY.md).

## Status

Session 1 is in progress: hybrid **search and evaluation** only. Answer
generation and image understanding are deliberately deferred until retrieval is
working and measured. Progress lives in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Requirements

- Linux, Docker >= 24 with Compose >= 2.26.1
- ~30 GB free disk, >= 16 GB RAM
- [uv](https://docs.astral.sh/uv/) (manages CPython 3.12 for this repo)
- Node.js >= 20
- [Ollama](https://ollama.com/) for local embeddings
- NVIDIA GPU optional; everything in the search slice runs on CPU

## Quick start

```bash
# 1. Verify the machine (read-only, downloads nothing)
make check

# 2. Create the Python workspace (uv fetches CPython 3.12)
make install

# 3. Run the backend test suite
make test
```

Full setup, including the RAGFlow stack and model registration, is documented in
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and expanded as each task
lands.

## Layout

```
apps/web             Next.js + TypeScript agent console
apps/api             FastAPI: query understanding, retrieval, evidence grading
services/ingestion   Renders synthetic procedures and ingests them into RAGFlow
services/reranker    Optional local rerank sidecar (fallback path)
data/synthetic       Fictional procedures, lexicon, screenshot templates
evaluation           Lexical baseline vs hybrid retrieval benchmark
tests                Pytest suites
infra                Docker Compose configuration for RAGFlow and sidecars
docs                 Architecture, environment, evaluation, troubleshooting
scripts              Environment checks, stack bootstrap, model wiring
```

## Security posture

Every service binds to `127.0.0.1`. Ollama is never exposed beyond the loopback
interface and the Docker bridge. Secrets live only in `.env.local`, which is
gitignored; `.env.example` carries placeholders. Nothing in this repository
connects to a third-party AI service or any company system. Details in
[SECURITY.md](SECURITY.md).

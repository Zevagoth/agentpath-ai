# AgentPath AI - developer entry points.
#
# Every target is safe to re-run. Nothing here deletes data or pushes to a remote.

SHELL := /bin/bash
.DEFAULT_GOAL := help

WORKSPACE := $(CURDIR)
RAGFLOW_DIR := $(WORKSPACE)/.external/ragflow/docker
COMPOSE_OVERRIDE := $(WORKSPACE)/infra/ragflow/docker-compose.override.agentpath.yml
ENV_LOCAL := $(WORKSPACE)/.env.local

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

.PHONY: check
check: ## Verify the machine is ready (read-only; writes docs/ENVIRONMENT.md)
	@python3 scripts/env_check.py --json docs/ENVIRONMENT.json --markdown docs/ENVIRONMENT.md

.PHONY: install
install: ## Create the Python workspace virtualenv (uv-managed CPython 3.12)
	@uv sync --all-packages

.PHONY: install-web
install-web: ## Install web app dependencies
	@cd apps/web && npm install
	@echo "web dependencies installed. Browser tests also need: make install-web-browsers"

.PHONY: install-web-browsers
install-web-browsers: ## Download the Chromium build Playwright uses (~174 MB)
	@cd apps/web && npx playwright install chromium

.PHONY: browser-check
browser-check: ## Report which browser Playwright will use, and prove it launches
	@cd apps/web && node scripts/browser-check.mjs

.PHONY: web-build
web-build: ## Production build of the web app
	@cd apps/web && npm run build

# --------------------------------------------------------------------------- #
# RAGFlow stack
# --------------------------------------------------------------------------- #

.PHONY: ragflow-bootstrap
ragflow-bootstrap: ## Clone RAGFlow at the pinned tag and render docker/.env
	@bash scripts/bootstrap_ragflow.sh

.PHONY: ragflow-up
ragflow-up: ## Start the RAGFlow stack plus the AgentPath reranker sidecar
	@bash scripts/ragflow_compose.sh up -d

.PHONY: ragflow-down
ragflow-down: ## Stop the stack (volumes are preserved)
	@bash scripts/ragflow_compose.sh down

.PHONY: ragflow-logs
ragflow-logs: ## Tail RAGFlow server logs
	@bash scripts/ragflow_compose.sh logs -f ragflow

.PHONY: ragflow-health
ragflow-health: ## Report the health of every stack dependency
	@bash scripts/ragflow_health.sh



.PHONY: models
models: ## Register the embedding + rerank models in RAGFlow and verify them
	@uv run python scripts/configure_ragflow_models.py

.PHONY: models-verify
models-verify: ## Prove the embedding and rerank paths work end to end
	@uv run python scripts/configure_ragflow_models.py --verify-only

# --------------------------------------------------------------------------- #
# Data and retrieval
# --------------------------------------------------------------------------- #

.PHONY: dataset
dataset: ## Render the synthetic procedures to Markdown
	@uv run agentpath-ingest build

.PHONY: ingest
ingest: ## Ingest the rendered procedures into RAGFlow
	@uv run agentpath-ingest ingest

.PHONY: ingest-force
ingest-force: ## Re-upload every document, moving chunk ids to free image keys
	@uv run agentpath-ingest ingest --force

.PHONY: ingest-verify
ingest-verify: ## Show ingestion status and chunk counts
	@uv run agentpath-ingest verify

.PHONY: screenshots
screenshots: ## Render the synthetic AgentDesk screens to PNG + sidecar JSON
	@cd apps/web && node scripts/generate-screenshots.mjs

.PHONY: screenshots-check
screenshots-check: ## Re-render the screens and verify them without writing
	@cd apps/web && node scripts/generate-screenshots.mjs --check

.PHONY: api
api: ## Run the FastAPI service on 127.0.0.1:8000
	@uv run uvicorn agentpath.api.app:app --host 127.0.0.1 --port 8000 --reload

.PHONY: web
web: ## Run the Next.js dev server on 127.0.0.1:3000
	@test -d apps/web/node_modules || (echo "run 'make install-web' first" && exit 1)
	@cd apps/web && npm run dev

.PHONY: search
search: ## One-off search from the terminal: make search Q="hotel has no room"
	@curl -sS -X POST http://127.0.0.1:8000/api/search \
		-H 'Content-Type: application/json' \
		-d '{"query": $(shell printf '%s' '$(Q)' | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}' \
		| python3 -m json.tool

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #

.PHONY: test
test: ## Run backend unit tests (no live stack required)
	@uv run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Run tests that require the live stack
	@uv run pytest -m integration

.PHONY: test-web
test-web: ## Run web unit tests (no stack required)
	@cd apps/web && npm run test

.PHONY: test-web-live
test-web-live: ## Run the real UI components against the running API
	@cd apps/web && npm run test:live

.PHONY: test-e2e
test-e2e: ## Run Playwright browser tests (needs: make install-web-browsers)
	@cd apps/web && npm run test:e2e

.PHONY: test-all
test-all: ## Backend + web unit tests
	@uv run pytest -m "not integration" -q
	@cd apps/web && npm run test

.PHONY: lint
lint: ## Lint Python sources
	@uv run ruff check .
	@uv run ruff format --check .

.PHONY: format
format: ## Format Python sources
	@uv run ruff format .

.PHONY: eval
eval: ## Run the baseline vs hybrid retrieval benchmark
	@uv run agentpath-eval

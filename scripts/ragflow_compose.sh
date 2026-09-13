#!/usr/bin/env bash
#
# Thin wrapper around docker compose for the RAGFlow stack plus AgentPath
# additions. Keeps the file list and project name in one place so every caller
# (Makefile, health check, docs) operates on exactly the same stack.
#
# Usage: scripts/ragflow_compose.sh <docker compose args...>
#   scripts/ragflow_compose.sh up -d
#   scripts/ragflow_compose.sh ps
#   scripts/ragflow_compose.sh logs -f ragflow-cpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/.external/ragflow/docker"
OVERRIDE="${REPO_ROOT}/infra/ragflow/docker-compose.override.agentpath.yml"

if [[ ! -f "${DOCKER_DIR}/.env" ]]; then
  echo "docker/.env is missing. Run: make ragflow-bootstrap" >&2
  exit 1
fi

# --project-name keeps container and volume names stable regardless of the
# directory Compose is invoked from.
exec docker compose \
  --project-name agentpath \
  --project-directory "${DOCKER_DIR}" \
  --env-file "${DOCKER_DIR}/.env" \
  -f "${DOCKER_DIR}/docker-compose.yml" \
  -f "${OVERRIDE}" \
  "$@"

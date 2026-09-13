#!/usr/bin/env bash
#
# Prepare the RAGFlow stack for AgentPath AI.
#
# What this script does, all idempotently:
#   1. Clones RAGFlow at the pinned tag into .external/ragflow (gitignored).
#   2. Generates strong secrets into .env.local, reusing existing values so a
#      re-run never invalidates data already written to the volumes.
#   3. Renders .external/ragflow/docker/.env as
#      "RAGFlow's shipped .env" + "infra/ragflow/env.agentpath with secrets".
#
# What it deliberately does NOT do: start containers, pull images, or touch
# anything outside this repository.
#
# Usage: scripts/bootstrap_ragflow.sh [--force-env]

set -euo pipefail

RAGFLOW_TAG="v0.27.2"
RAGFLOW_REPO="https://github.com/infiniflow/ragflow.git"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/.external/ragflow"
DOCKER_DIR="${EXTERNAL_DIR}/docker"
ENV_LOCAL="${REPO_ROOT}/.env.local"
ENV_OVERRIDE="${REPO_ROOT}/infra/ragflow/env.agentpath"
RENDERED_ENV="${DOCKER_DIR}/.env"

log()  { printf '\033[36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[bootstrap]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[bootstrap] %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. RAGFlow checkout at the pinned tag
# --------------------------------------------------------------------------- #

if [[ ! -d "${EXTERNAL_DIR}/.git" ]]; then
  log "cloning RAGFlow ${RAGFLOW_TAG} into .external/ragflow"
  mkdir -p "${REPO_ROOT}/.external"
  git clone --depth 1 --branch "${RAGFLOW_TAG}" "${RAGFLOW_REPO}" "${EXTERNAL_DIR}"
else
  current="$(git -C "${EXTERNAL_DIR}" describe --tags --always 2>/dev/null || echo unknown)"
  if [[ "${current}" != "${RAGFLOW_TAG}" ]]; then
    warn "existing checkout is at ${current}, expected ${RAGFLOW_TAG}"
    warn "remove .external/ragflow manually if you want a clean re-clone"
  else
    log "RAGFlow checkout already at ${RAGFLOW_TAG}"
  fi
fi

[[ -f "${DOCKER_DIR}/docker-compose.yml" ]] || die "RAGFlow compose file missing at ${DOCKER_DIR}"
[[ -f "${DOCKER_DIR}/.env" || -f "${DOCKER_DIR}/.env.bak" ]] || true

# --------------------------------------------------------------------------- #
# 2. Secrets in .env.local
# --------------------------------------------------------------------------- #

command -v openssl >/dev/null 2>&1 || die "openssl is required to generate secrets"

# Read an existing value from .env.local, if present.
read_local() {
  local key="$1"
  [[ -f "${ENV_LOCAL}" ]] || return 0
  sed -n "s/^${key}=//p" "${ENV_LOCAL}" | tail -1
}

ensure_secret() {
  local key="$1" existing
  existing="$(read_local "${key}")"
  if [[ -n "${existing}" && "${existing}" != REPLACE-ME* ]]; then
    printf '%s' "${existing}"
    return 0
  fi
  # 32 hex characters: strong, and free of shell/YAML metacharacters that would
  # need escaping inside .env and service_conf.
  openssl rand -hex 16
}

if [[ ! -f "${ENV_LOCAL}" ]]; then
  log "creating .env.local from .env.example"
  cp "${REPO_ROOT}/.env.example" "${ENV_LOCAL}"
  chmod 600 "${ENV_LOCAL}"
fi

MYSQL_PASSWORD="$(ensure_secret RAGFLOW_MYSQL_PASSWORD)"
MINIO_PASSWORD="$(ensure_secret RAGFLOW_MINIO_PASSWORD)"
REDIS_PASSWORD="$(ensure_secret RAGFLOW_REDIS_PASSWORD)"
ELASTIC_PASSWORD="$(ensure_secret RAGFLOW_ELASTIC_PASSWORD)"
ADMIN_PASSWORD="$(ensure_secret RAGFLOW_ADMIN_PASSWORD)"

# Write the values back so the next run is stable.
write_local() {
  local key="$1" value="$2"
  if grep -q "^${key}=" "${ENV_LOCAL}"; then
    # Use a delimiter that cannot appear in a hex secret.
    sed -i "s|^${key}=.*|${key}=${value}|" "${ENV_LOCAL}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_LOCAL}"
  fi
}

write_local RAGFLOW_MYSQL_PASSWORD "${MYSQL_PASSWORD}"
write_local RAGFLOW_MINIO_PASSWORD "${MINIO_PASSWORD}"
write_local RAGFLOW_REDIS_PASSWORD "${REDIS_PASSWORD}"
write_local RAGFLOW_ELASTIC_PASSWORD "${ELASTIC_PASSWORD}"
write_local RAGFLOW_ADMIN_PASSWORD "${ADMIN_PASSWORD}"
chmod 600 "${ENV_LOCAL}"
log "secrets present in .env.local (mode 600, gitignored)"

# --------------------------------------------------------------------------- #
# 3. Render docker/.env
# --------------------------------------------------------------------------- #

# The upstream .env is the baseline: it defines every variable the compose model
# interpolates, including services in profiles we do not start.
UPSTREAM_ENV="${DOCKER_DIR}/.env.upstream"
if [[ ! -f "${UPSTREAM_ENV}" ]]; then
  git -C "${EXTERNAL_DIR}" show "${RAGFLOW_TAG}:docker/.env" > "${UPSTREAM_ENV}"
  log "captured upstream docker/.env as .env.upstream"
fi

TZ_VALUE="$(cat /etc/timezone 2>/dev/null || true)"
if [[ -z "${TZ_VALUE}" ]] && [[ -L /etc/localtime ]]; then
  TZ_VALUE="$(readlink -f /etc/localtime | sed 's#.*/zoneinfo/##')"
fi
TZ_VALUE="${TZ_VALUE:-Etc/UTC}"

{
  cat "${UPSTREAM_ENV}"
  printf '\n'
  sed \
    -e "s|__AGENTPATH_TZ__|${TZ_VALUE}|g" \
    -e "s|__AGENTPATH_MYSQL_PASSWORD__|${MYSQL_PASSWORD}|g" \
    -e "s|__AGENTPATH_MINIO_PASSWORD__|${MINIO_PASSWORD}|g" \
    -e "s|__AGENTPATH_REDIS_PASSWORD__|${REDIS_PASSWORD}|g" \
    -e "s|__AGENTPATH_ELASTIC_PASSWORD__|${ELASTIC_PASSWORD}|g" \
    -e "s|__AGENTPATH_ADMIN_PASSWORD__|${ADMIN_PASSWORD}|g" \
    "${ENV_OVERRIDE}"
} > "${RENDERED_ENV}"
chmod 600 "${RENDERED_ENV}"

# Fail loudly rather than starting a stack with RAGFlow's shipped defaults.
#
# The upstream section still carries those defaults, and that is fine: Compose
# applies the last definition of a key, which is ours. So the guard must check
# the *effective* value per key, not merely whether the string appears.
effective_value() {
  local key="$1"
  awk -F= -v key="${key}" '$1 == key { value = substr($0, index($0, "=") + 1) } END { print value }' "${RENDERED_ENV}"
}

for key in MYSQL_PASSWORD MINIO_PASSWORD REDIS_PASSWORD ELASTIC_PASSWORD ADMIN_DEFAULT_PASSWORD; do
  value="$(effective_value "${key}")"
  case "${value}" in
    ""|infini_rag_flow|admin|REPLACE-ME*)
      die "effective ${key} is a shipped default or empty; refusing to start the stack"
      ;;
  esac
done

# Every published port must be bound to loopback.
for key in SVR_WEB_HTTP_PORT SVR_WEB_HTTPS_PORT SVR_HTTP_PORT ADMIN_SVR_HTTP_PORT SVR_MCP_PORT \
           GO_HTTP_PORT GO_ADMIN_PORT ES_PORT EXPOSE_MYSQL_PORT MINIO_PORT MINIO_CONSOLE_PORT \
           REDIS_PORT AGENTPATH_EMBEDDER_PORT AGENTPATH_RERANKER_PORT; do
  value="$(effective_value "${key}")"
  if [[ "${value}" != 127.0.0.1:* ]]; then
    die "${key}=${value} would publish beyond loopback; refusing to continue"
  fi
done
# Ignore comment lines: the override file documents the placeholder syntax.
if grep -v '^\s*#' "${RENDERED_ENV}" | grep -q '__AGENTPATH_'; then
  die "rendered .env still contains unsubstituted placeholders"
fi

log "rendered ${RENDERED_ENV}"
log "timezone: ${TZ_VALUE}"
log ""
log "next: make ragflow-up   (pulls ~4 GB of images on first run)"

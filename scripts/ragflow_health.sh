#!/usr/bin/env bash
#
# Report the health of every dependency AgentPath AI needs, and verify the
# security posture we promised: nothing published beyond loopback.
#
# Exit code 0 when every required dependency is reachable, 1 otherwise.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="${REPO_ROOT}/scripts/ragflow_compose.sh"

RAGFLOW_API="http://127.0.0.1:9380"
RAGFLOW_UI="http://127.0.0.1:8081"
EMBEDDER="http://127.0.0.1:7998"
RERANKER="http://127.0.0.1:7997"
OLLAMA="http://127.0.0.1:11434"

failures=0
ok()   { printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; failures=$((failures + 1)); }
note() { printf '  \033[33m[WARN]\033[0m %s\n' "$*"; }

echo
echo "AgentPath AI - stack health"
echo "==========================================================="

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
echo
echo "containers"
if ! docker info >/dev/null 2>&1; then
  bad "docker daemon unreachable"
else
  while IFS=$'\t' read -r name state health; do
    [[ -z "${name}" ]] && continue
    case "${health:-none}" in
      healthy|none)
        if [[ "${state}" == "running" ]]; then
          ok "${name} (${state}${health:+, ${health}})"
        else
          bad "${name} is ${state}"
        fi
        ;;
      starting) note "${name} still starting (health: starting)" ;;
      *)        bad "${name} state=${state} health=${health}" ;;
    esac
  done < <("${COMPOSE}" ps --format '{{.Name}}\t{{.State}}\t{{.Health}}' 2>/dev/null)
fi

# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #
echo
echo "endpoints"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null; }

code="$(http_code "${RAGFLOW_API}/api/v1/system/healthz")"
case "${code}" in
  200|401|403) ok "RAGFlow API ${RAGFLOW_API} (HTTP ${code})" ;;
  000)         bad "RAGFlow API ${RAGFLOW_API} not answering" ;;
  *)           note "RAGFlow API ${RAGFLOW_API} returned HTTP ${code}" ;;
esac

code="$(http_code "${RAGFLOW_UI}/")"
case "${code}" in
  200|301|302) ok "RAGFlow UI ${RAGFLOW_UI} (HTTP ${code})" ;;
  000)         bad "RAGFlow UI ${RAGFLOW_UI} not answering" ;;
  *)           note "RAGFlow UI ${RAGFLOW_UI} returned HTTP ${code}" ;;
esac

code="$(http_code "${EMBEDDER}/health")"
case "${code}" in
  200) ok "embedder ${EMBEDDER} (bge-m3 loaded)" ;;
  000) bad "embedder ${EMBEDDER} not answering" ;;
  *)   note "embedder ${EMBEDDER} returned HTTP ${code} (model may still be loading)" ;;
esac

code="$(http_code "${RERANKER}/health")"
case "${code}" in
  200) ok "reranker ${RERANKER} (bge-reranker-v2-m3 loaded)" ;;
  000) bad "reranker ${RERANKER} not answering" ;;
  *)   note "reranker ${RERANKER} returned HTTP ${code} (model may still be loading)" ;;
esac

# Ollama is a Stage 2 dependency only, called from the API process over loopback.
if curl -fsS --max-time 5 "${OLLAMA}/api/tags" >/dev/null 2>&1; then
  ok "Ollama ${OLLAMA} (Stage 2 answer generation)"
else
  note "Ollama ${OLLAMA} not answering (only needed for Stage 2)"
fi

# --------------------------------------------------------------------------- #
# Security posture
# --------------------------------------------------------------------------- #
echo
echo "security posture"

# 1. Published container ports must all bind 127.0.0.1.
exposed="$(docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null | grep -v '127\.0\.0\.1' | grep -E '0\.0\.0\.0|\[::\]|:::' || true)"
if [[ -z "${exposed}" ]]; then
  ok "no container port published beyond 127.0.0.1"
else
  bad "containers publishing beyond loopback:"
  printf '        %s\n' "${exposed}"
fi

# 2. Ollama itself must not listen on a routable address.
ollama_listen="$(ss -ltnH 2>/dev/null | awk '{print $4}' | grep ':11434$' || true)"
if [[ -z "${ollama_listen}" ]]; then
  note "Ollama listener not found via ss"
elif grep -qE '^(0\.0\.0\.0|\*|\[::\])' <<< "${ollama_listen}"; then
  bad "Ollama listens on ${ollama_listen} (reachable from the LAN)"
else
  ok "Ollama listens on ${ollama_listen}"
fi

# 3. Model services must be reachable from RAGFlow over the Compose network,
#    not through the host. This is the check that proves no firewall hole is
#    needed for the model path.
for service in agentpath-embedder agentpath-reranker; do
  if docker exec agentpath-ragflow-cpu-1 \
      timeout 8 curl -fsS "http://${service}:80/health" >/dev/null 2>&1; then
    ok "RAGFlow reaches ${service} over the Compose network"
  else
    bad "RAGFlow cannot reach ${service} on its own network"
  fi
done

# 4. Open registration is a first-run convenience only.
env_file="${REPO_ROOT}/.external/ragflow/docker/.env"
if [[ -f "${env_file}" ]]; then
  register="$(awk -F= '$1 == "REGISTER_ENABLED" { v = $2 } END { print v }' "${env_file}")"
  if [[ "${register}" == "1" ]]; then
    note "RAGFlow registration is open (REGISTER_ENABLED=1). Set it to 0 after creating your account."
  else
    ok "RAGFlow registration is closed"
  fi
fi

echo
if (( failures > 0 )); then
  echo "verdict: ${failures} check(s) failed"
  exit 1
fi
echo "verdict: stack healthy"

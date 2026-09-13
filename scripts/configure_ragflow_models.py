#!/usr/bin/env python3
"""Provision the AgentPath AI model path inside RAGFlow, then prove it works.

RAGFlow v0.27.2 ships no bundled embedding models, so the embedding and rerank
providers must be registered before a dataset can be created. Doing that by hand
in the UI is six forms and a screenshot in a README; doing it here makes the
setup reproducible and reviewable.

What it provisions (idempotently):

1. A local operator account, if registration is still open. Credentials are
   generated into ``.env.local`` - never printed, never committed.
2. An API key for the AgentPath services, stored in ``.env.local``.
3. The ``Ollama`` provider pointing at the bridge relay, serving ``bge-m3`` for
   embeddings.
4. The ``HuggingFace`` provider pointing at the reranker sidecar, serving
   ``BAAI/bge-reranker-v2-m3`` for reranking. RAGFlow's HuggingFace rerank
   adapter speaks Text Embeddings Inference's protocol, which is exactly what the
   sidecar serves.
5. Both models as the tenant defaults.

Then it verifies the path end to end: a bilingual embedding pair through Ollama
(reporting cross-lingual cosine similarity, which is the whole reason bge-m3 was
chosen) and a rerank call through the sidecar.

Usage:
    uv run python scripts/configure_ragflow_models.py
    uv run python scripts/configure_ragflow_models.py --verify-only
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_LOCAL = REPO_ROOT / ".env.local"
PUBLIC_KEY = REPO_ROOT / ".external" / "ragflow" / "conf" / "public.pem"

RAGFLOW_URL = os.environ.get("RAGFLOW_BASE_URL", "http://127.0.0.1:9380")
EMBEDDER_URL = os.environ.get("EMBEDDER_URL", "http://127.0.0.1:7998")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://127.0.0.1:7997")

EMBEDDING_MODEL = os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

#: RAGFlow reaches both sidecars by Compose service name on its own network.
#: Not through the host: ufw is active on the development machine and drops
#: traffic from Docker's user-defined bridge to host addresses, so a host-side
#: model daemon would require a firewall rule. Staying inside the network needs
#: no rule and exposes nothing.
EMBEDDER_INTERNAL_URL = "http://agentpath-embedder:80"
RERANKER_INTERNAL_URL = "http://agentpath-reranker:80"

OPERATOR_EMAIL_DEFAULT = "operator@agentpath.local"
OPERATOR_NICKNAME = "AgentPath Operator"


def info(message: str) -> None:
    print(f"\033[36m[models]\033[0m {message}")


def warn(message: str) -> None:
    print(f"\033[33m[models]\033[0m {message}")


def fail(message: str) -> None:
    print(f"\033[31m[models] {message}\033[0m", file=sys.stderr)


# --------------------------------------------------------------------------- #
# .env.local helpers
# --------------------------------------------------------------------------- #


def read_env_local() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_LOCAL.is_file():
        return values
    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def write_env_local(key: str, value: str) -> None:
    """Set ``key`` in .env.local, preserving everything else."""
    lines = ENV_LOCAL.read_text(encoding="utf-8").splitlines() if ENV_LOCAL.is_file() else []
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[index] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    ENV_LOCAL.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV_LOCAL.chmod(0o600)


# --------------------------------------------------------------------------- #
# RAGFlow password encryption
# --------------------------------------------------------------------------- #


def encrypt_password(password: str) -> str:
    """Encrypt a password the way RAGFlow's own clients do.

    RAGFlow expects ``base64(RSA_PKCS1_v1_5(base64(password)))`` using the public
    key shipped in the repository (passphrase "Welcome" - it is a fixed,
    published value in RAGFlow, not a secret of ours). Replicated here so account
    setup can be scripted instead of clicked.
    """
    try:
        from Cryptodome.Cipher import PKCS1_v1_5
        from Cryptodome.PublicKey import RSA
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise SystemExit("pycryptodomex is required: uv sync --all-packages") from exc

    if not PUBLIC_KEY.is_file():
        raise SystemExit(f"RAGFlow public key not found at {PUBLIC_KEY}. Run: make ragflow-bootstrap")

    rsa_key = RSA.importKey(PUBLIC_KEY.read_text(encoding="utf-8"), "Welcome")
    cipher = PKCS1_v1_5.new(rsa_key)
    password_base64 = base64.b64encode(password.encode("utf-8"))
    return base64.b64encode(cipher.encrypt(password_base64)).decode("utf-8")


# --------------------------------------------------------------------------- #
# RAGFlow admin session
# --------------------------------------------------------------------------- #


@dataclass
class ProviderSpec:
    """One provider instance to register."""

    provider_name: str
    instance_name: str
    base_url: str
    model_name: str
    model_type: str
    max_tokens: int
    api_key: str = "agentpath-local"


class RAGFlowAdmin:
    """Authenticated session against RAGFlow's management API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)
        self._auth: str | None = None
        self._tenant_id: str | None = None

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._auth:
            headers["Authorization"] = self._auth
        return headers

    def _call(self, method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any], httpx.Response]:
        response = self._client.request(
            method, f"{self._base_url}{path}", headers=self._headers(), **kwargs
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"code": -1, "message": f"non-JSON response (HTTP {response.status_code})"}
        return payload, response

    # -- account ------------------------------------------------------------ #

    def register(self, email: str, password: str) -> bool:
        """Create the operator account. Returns True when newly created."""
        payload, _ = self._call(
            "POST",
            "/api/v1/users",
            json={
                "nickname": OPERATOR_NICKNAME,
                "email": email,
                "password": encrypt_password(password),
            },
        )
        if payload.get("code") == 0:
            return True
        message = str(payload.get("message", ""))
        if "already registered" in message:
            return False
        if "registration is disabled" in message:
            raise SystemExit(
                "RAGFlow registration is disabled and no operator account exists yet.\n"
                "Set REGISTER_ENABLED=1 in infra/ragflow/env.agentpath, re-run "
                "make ragflow-bootstrap && make ragflow-up, then retry."
            )
        raise SystemExit(f"registration failed: {message}")

    def login(self, email: str, password: str) -> None:
        payload, response = self._call(
            "POST",
            "/api/v1/auth/login",
            json={"email": email, "password": encrypt_password(password)},
        )
        if payload.get("code") != 0:
            raise SystemExit(f"login failed: {payload.get('message')}")
        auth = response.headers.get("Authorization")
        if not auth:
            raise SystemExit("login succeeded but RAGFlow returned no Authorization header")
        self._auth = auth
        data = payload.get("data") or {}
        self._tenant_id = data.get("id")

    @property
    def tenant_id(self) -> str:
        if not self._tenant_id:
            raise SystemExit("not logged in")
        return self._tenant_id

    # -- api key ------------------------------------------------------------ #

    def existing_api_key(self) -> str | None:
        payload, _ = self._call("GET", "/api/v1/system/tokens")
        if payload.get("code") != 0:
            return None
        tokens = payload.get("data") or []
        for token in tokens:
            value = token.get("token")
            if value:
                return str(value)
        return None

    def create_api_key(self) -> str:
        payload, _ = self._call("POST", "/api/v1/system/tokens")
        if payload.get("code") != 0:
            raise SystemExit(f"API key creation failed: {payload.get('message')}")
        token = (payload.get("data") or {}).get("token")
        if not token:
            raise SystemExit(f"API key creation returned no token: {payload}")
        return str(token)

    # -- providers ---------------------------------------------------------- #

    def add_provider(self, provider_name: str) -> None:
        payload, _ = self._call("PUT", "/api/v1/providers", json={"provider_name": provider_name})
        if payload.get("code") == 0:
            return
        message = str(payload.get("message", "")).lower()
        if "exist" in message or "already" in message:
            return
        raise SystemExit(f"could not add provider {provider_name}: {payload.get('message')}")

    def create_instance(self, spec: ProviderSpec) -> None:
        payload, _ = self._call(
            "POST",
            f"/api/v1/providers/{spec.provider_name}/instances",
            json={
                "instance_name": spec.instance_name,
                "api_key": spec.api_key,
                "base_url": spec.base_url,
                "model_info": [
                    {
                        "model_type": [spec.model_type],
                        "model_name": spec.model_name,
                        "max_tokens": spec.max_tokens,
                    }
                ],
            },
        )
        if payload.get("code") == 0:
            return
        message = str(payload.get("message", ""))
        if "exist" in message.lower():
            info(f"{spec.provider_name}/{spec.model_name} already registered")
            return
        raise SystemExit(f"could not register {spec.provider_name}/{spec.model_name}: {message}")

    def list_models(self) -> dict[str, Any]:
        payload, _ = self._call("GET", "/api/v1/users/me/models")
        if payload.get("code") != 0:
            raise SystemExit(f"could not read tenant models: {payload.get('message')}")
        return payload.get("data") or {}

    def set_default_models(self, *, embedding_id: str, rerank_id: str) -> None:
        """Set tenant defaults.

        The endpoint validates presence of every model slot, so current values
        are read first and sent back unchanged where we have no opinion.
        """
        current = self.list_models()
        body = {
            "tenant_id": self.tenant_id,
            "llm_id": current.get("llm_id") or "",
            "embd_id": embedding_id,
            "asr_id": current.get("asr_id") or "",
            "img2txt_id": current.get("img2txt_id") or "",
            "rerank_id": rerank_id,
        }
        payload, _ = self._call("PATCH", "/api/v1/users/me/models", json=body)
        if payload.get("code") != 0:
            raise SystemExit(f"could not set default models: {payload.get('message')}")


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def verify_embeddings() -> bool:
    """Embed an English and an Arabic phrase with the same meaning.

    A high cosine similarity between them is the property AgentPath depends on:
    it is what lets an Arabic query find an English procedure. The unrelated
    phrase is the control - without it, a model that maps everything close
    together would look like a success.
    """
    english = "the hotel has no room for a confirmed guest"
    arabic = "الفندق ما عنده غرف لضيف عنده حجز مؤكد"
    unrelated = "passenger name correction on a flight ticket"

    vectors: list[list[float]] = []
    with httpx.Client(timeout=180.0) as client:
        for text in (english, arabic, unrelated):
            response = client.post(f"{EMBEDDER_URL}/embed", json={"inputs": [text]})
            if response.status_code != 200:
                fail(f"embedding failed (HTTP {response.status_code}): {response.text[:200]}")
                return False
            payload = response.json()
            if not isinstance(payload, list) or not payload or not payload[0]:
                fail(f"embedder returned no vector for {text!r}: {payload!r}")
                return False
            vectors.append(payload[0])

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        fail(f"inconsistent embedding dimensions: {dimensions}")
        return False

    cross_lingual = cosine(vectors[0], vectors[1])
    unrelated_similarity = cosine(vectors[0], vectors[2])

    info(f"embedding model     : {EMBEDDING_MODEL} ({dimensions.pop()} dimensions)")
    info(f"cross-lingual cosine: {cross_lingual:.3f}  (English vs Arabic, same meaning)")
    info(f"unrelated cosine    : {unrelated_similarity:.3f}  (English vs different topic)")

    if cross_lingual <= unrelated_similarity:
        fail("the Arabic phrasing is not closer than an unrelated phrase; embeddings are unusable")
        return False
    if cross_lingual < 0.5:
        warn("cross-lingual similarity is low; Arabic recall may suffer")
    return True


def verify_reranker() -> bool:
    """Rerank three documents and check the relevant one wins."""
    query = "hotel cannot accommodate a guest who already arrived"
    documents = [
        "Passenger name correction requires the ticket number and the passport spelling.",
        "When the property confirms it cannot host a confirmed guest, secure equivalent alternative accommodation.",
        "Refunds settle within the window shown on the refund timeline.",
    ]

    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "texts": documents, "raw_scores": False},
        )
    if response.status_code != 200:
        fail(f"reranker failed (HTTP {response.status_code}): {response.text[:200]}")
        return False

    results = response.json()
    if not isinstance(results, list) or not results:
        fail(f"unexpected reranker response: {results!r}")
        return False

    ordered = sorted(results, key=lambda item: -item.get("score", 0.0))
    info(f"rerank model        : {RERANK_MODEL}")
    for entry in ordered:
        index = entry.get("index", -1)
        snippet = documents[index][:60] if 0 <= index < len(documents) else "?"
        info(f"  score {entry.get('score', 0.0):.4f}  [{index}] {snippet}...")

    if ordered[0].get("index") != 1:
        fail("the reranker did not rank the relevant passage first")
        return False
    return True


def verify_ragflow_defaults(admin: RAGFlowAdmin, *, embedding_id: str, rerank_id: str) -> bool:
    models = admin.list_models()
    actual_embedding = models.get("embd_id") or ""
    actual_rerank = models.get("rerank_id") or ""
    info(f"RAGFlow default embedding: {actual_embedding or '(unset)'}")
    info(f"RAGFlow default rerank   : {actual_rerank or '(unset)'}")
    ok = True
    if embedding_id.split("@")[0] not in actual_embedding:
        fail(f"expected embedding default to include {embedding_id}")
        ok = False
    if rerank_id.split("@")[0] not in actual_rerank:
        fail(f"expected rerank default to include {rerank_id}")
        ok = False
    return ok


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def provision(admin: RAGFlowAdmin, env: dict[str, str]) -> tuple[str, str]:
    email = env.get("RAGFLOW_OPERATOR_EMAIL") or OPERATOR_EMAIL_DEFAULT
    password = env.get("RAGFLOW_OPERATOR_PASSWORD") or secrets.token_urlsafe(24)
    write_env_local("RAGFLOW_OPERATOR_EMAIL", email)
    write_env_local("RAGFLOW_OPERATOR_PASSWORD", password)

    created = admin.register(email, password)
    info(f"operator account {'created' if created else 'already present'}: {email}")

    admin.login(email, password)
    info("logged in")

    api_key = env.get("RAGFLOW_API_KEY", "")
    if not api_key or api_key.startswith("ragflow-REPLACE"):
        api_key = admin.existing_api_key() or admin.create_api_key()
        write_env_local("RAGFLOW_API_KEY", api_key)
        info("API key stored in .env.local")
    else:
        info("API key already present in .env.local")

    embedding_spec = ProviderSpec(
        provider_name="HuggingFace",
        instance_name="agentpath-embedder",
        base_url=EMBEDDER_INTERNAL_URL,
        model_name=EMBEDDING_MODEL,
        model_type="embedding",
        max_tokens=8192,
    )
    rerank_spec = ProviderSpec(
        provider_name="HuggingFace",
        instance_name="agentpath-reranker",
        base_url=RERANKER_INTERNAL_URL,
        model_name=RERANK_MODEL,
        model_type="rerank",
        max_tokens=8192,
    )

    for spec in (embedding_spec, rerank_spec):
        admin.add_provider(spec.provider_name)
        admin.create_instance(spec)
        info(f"registered {spec.provider_name}: {spec.model_name} ({spec.model_type}) at {spec.base_url}")

    # RAGFlow resolves a model reference as "{model}@{instance}@{provider}". The
    # shorter "{model}@{provider}" form implies an instance literally named
    # "default", and RAGFlow refuses to create an instance with that name, so the
    # three-part form is the only one that resolves for a self-hosted provider.
    embedding_id = (
        f"{embedding_spec.model_name}@{embedding_spec.instance_name}@{embedding_spec.provider_name}"
    )
    rerank_id = f"{rerank_spec.model_name}@{rerank_spec.instance_name}@{rerank_spec.provider_name}"
    admin.set_default_models(embedding_id=embedding_id, rerank_id=rerank_id)
    info(f"tenant defaults set: embedding={embedding_id}, rerank={rerank_id}")

    write_env_local("RAGFLOW_EMBEDDING_MODEL", embedding_id)
    write_env_local("RAGFLOW_RERANK_MODEL", rerank_id)
    return embedding_id, rerank_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure and verify the AgentPath model path in RAGFlow.")
    parser.add_argument("--verify-only", action="store_true", help="skip provisioning; only run checks")
    args = parser.parse_args(argv)

    env = read_env_local()
    admin = RAGFlowAdmin(RAGFLOW_URL)

    try:
        if args.verify_only:
            email = env.get("RAGFLOW_OPERATOR_EMAIL")
            password = env.get("RAGFLOW_OPERATOR_PASSWORD")
            if not email or not password:
                fail("no operator credentials in .env.local; run without --verify-only first")
                return 1
            admin.login(email, password)
            embedding_id = env.get(
                "RAGFLOW_EMBEDDING_MODEL", f"{EMBEDDING_MODEL}@agentpath-embedder@HuggingFace"
            )
            rerank_id = env.get(
                "RAGFLOW_RERANK_MODEL", f"{RERANK_MODEL}@agentpath-reranker@HuggingFace"
            )
        else:
            embedding_id, rerank_id = provision(admin, env)

        print()
        info("verification")
        checks = [
            ("embeddings", verify_embeddings()),
            ("reranker", verify_reranker()),
            ("ragflow defaults", verify_ragflow_defaults(admin, embedding_id=embedding_id, rerank_id=rerank_id)),
        ]
    finally:
        admin.close()

    print()
    failed = [name for name, ok in checks if not ok]
    if failed:
        fail(f"failed checks: {', '.join(failed)}")
        return 1
    info("model path verified: embeddings, reranking and RAGFlow defaults are in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

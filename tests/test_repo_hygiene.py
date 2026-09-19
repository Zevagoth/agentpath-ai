"""Guards on the repository itself.

These assert properties that are easy to break silently and expensive to discover
later: a secret becoming tracked, a Makefile target pointing at something that does
not exist, or a documented port drifting away from the loopback-only promise.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def test_no_env_file_other_than_the_example_is_tracked() -> None:
    tracked = tracked_files()
    env_files = [path for path in tracked if Path(path).name.startswith(".env") and Path(path).name != ".env.example"]
    assert env_files == [], f"secret-bearing env files are tracked: {env_files}"


def test_gitignore_excludes_the_local_env_and_external_checkout() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env.local", ".external/"):
        assert pattern in ignored, pattern


def test_example_env_contains_no_plausible_real_secret() -> None:
    """Placeholders only: nothing that looks like a generated credential."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    # RAGFlow API keys look like ragflow-<32+ base64ish chars>; generated
    # passwords are 32 hex characters.
    assert not re.search(r"ragflow-[A-Za-z0-9]{20,}", text)
    for line in text.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        assert not re.fullmatch(r"[a-f0-9]{32}", value.strip()), line


# --------------------------------------------------------------------------- #
# Loopback-only promise
# --------------------------------------------------------------------------- #


def test_ragflow_env_publishes_only_on_loopback() -> None:
    text = (REPO_ROOT / "infra" / "ragflow" / "env.agentpath").read_text(encoding="utf-8")
    port_keys = [
        "SVR_WEB_HTTP_PORT",
        "SVR_HTTP_PORT",
        "ADMIN_SVR_HTTP_PORT",
        "ES_PORT",
        "EXPOSE_MYSQL_PORT",
        "MINIO_PORT",
        "REDIS_PORT",
        "AGENTPATH_EMBEDDER_PORT",
        "AGENTPATH_RERANKER_PORT",
    ]
    for key in port_keys:
        match = re.search(rf"^{key}=(.+)$", text, re.MULTILINE)
        assert match, f"{key} missing from env.agentpath"
        assert match.group(1).startswith("127.0.0.1:"), f"{key}={match.group(1)}"


def test_api_defaults_to_loopback() -> None:
    from agentpath.config import Settings

    settings = Settings(RAGFLOW_API_KEY="test")
    assert settings.api_host == "127.0.0.1"


def test_web_dev_server_binds_loopback() -> None:
    import json

    package = json.loads((REPO_ROOT / "apps" / "web" / "package.json").read_text(encoding="utf-8"))
    for script in ("dev", "start"):
        assert "127.0.0.1" in package["scripts"][script], script


# --------------------------------------------------------------------------- #
# Makefile targets must point at things that exist
# --------------------------------------------------------------------------- #


def _makefile_recipes() -> dict[str, list[str]]:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        target = re.match(r"^([a-zA-Z][\w-]*):", line)
        if target:
            current = target.group(1)
            recipes[current] = []
        elif current and line.startswith("\t"):
            recipes[current].append(line.strip())
    return recipes


@pytest.mark.parametrize(
    ("target", "expected_path"),
    [
        ("check", "scripts/env_check.py"),
        ("ragflow-bootstrap", "scripts/bootstrap_ragflow.sh"),
        ("ragflow-health", "scripts/ragflow_health.sh"),
        ("models", "scripts/configure_ragflow_models.py"),
    ],
)
def test_makefile_scripts_exist(target: str, expected_path: str) -> None:
    recipes = _makefile_recipes()
    assert target in recipes, f"Makefile target {target} is missing"
    assert any(expected_path in line for line in recipes[target])
    assert (REPO_ROOT / expected_path).is_file(), expected_path


def test_makefile_console_scripts_resolve_to_real_entry_points() -> None:
    """`make ingest` and `make eval` must not name modules that do not exist."""
    from importlib import import_module

    for module_path, function in [
        ("agentpath_ingestion.cli", "main"),
        ("agentpath_eval.runner", "main"),
    ]:
        module = import_module(module_path)
        assert callable(getattr(module, function))


def test_every_makefile_recipe_references_something_that_exists() -> None:
    """Catch the whole class of "target points at a module or file that is gone".

    `make search` previously invoked `python -m agentpath.cli`, which never
    existed. A per-target test would not have caught it, so this scans every
    recipe for the three shapes we use - `python -m module`, `python path.py`, and
    `bash script.sh` - and checks each one resolves.
    """
    import importlib.util

    problems: list[str] = []
    for target, lines in _makefile_recipes().items():
        for line in lines:
            for module in re.findall(r"python3?\s+-m\s+([\w.]+)", line):
                if importlib.util.find_spec(module) is None:
                    problems.append(f"{target}: python -m {module} does not resolve")
            for script in re.findall(r"(?:bash|sh)\s+(scripts/[\w./-]+)", line):
                if not (REPO_ROOT / script).is_file():
                    problems.append(f"{target}: {script} is missing")
            for script in re.findall(r"python3?\s+(scripts/[\w./-]+\.py)", line):
                if not (REPO_ROOT / script).is_file():
                    problems.append(f"{target}: {script} is missing")
    assert problems == [], "\n".join(problems)


def test_makefile_targets_are_all_documented() -> None:
    """Every target carries a `##` description, which is what `make help` prints."""
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    undocumented = [
        match.group(1)
        for match in re.finditer(r"^([a-zA-Z][\w-]*):(?![=])(.*)$", text, re.MULTILINE)
        if "##" not in match.group(2)
    ]
    assert undocumented == [], f"targets without a help description: {undocumented}"


def test_api_module_referenced_by_makefile_imports() -> None:
    from agentpath.api.app import app

    assert app is not None


def test_web_app_entry_points_exist() -> None:
    web = REPO_ROOT / "apps" / "web"
    for path in (
        "package.json",
        "app/page.tsx",
        "app/layout.tsx",
        # The console absorbed the former SearchConsole as its search mode, so
        # there is one component with the case form rather than two.
        "components/AgentConsole.tsx",
        "components/AnswerView.tsx",
    ):
        assert (web / path).is_file(), path


def test_the_web_app_never_injects_html() -> None:
    """No `dangerouslySetInnerHTML` anywhere in the console.

    RAGFlow returns matched terms wrapped in `<em>`, and that markup used to be
    handed to the DOM directly. The snippet is assembled from *indexed document
    content*, which is the category this system treats as untrusted everywhere
    else - T8 rebuilds every generated step from the committed YAML precisely so
    that text from the index cannot reach the model. `lib/highlight.ts` parses the
    markup into React text nodes instead, and this keeps the sink from coming back.
    """
    components = (REPO_ROOT / "apps" / "web").rglob("*.tsx")
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in components
        if "node_modules" not in path.parts
        and ".next" not in path.parts
        and "dangerouslySetInnerHTML" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# Documentation that must stay in step with the code
# --------------------------------------------------------------------------- #


def test_referenced_documents_exist() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in re.findall(r"\]\(([A-Za-z0-9_./-]+\.md)\)", readme):
        assert (REPO_ROOT / name).is_file(), f"README links to missing {name}"


def test_security_document_states_the_missing_auth_layer() -> None:
    """The most important caveat must not quietly disappear."""
    text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "no authentication" in text.lower()


def test_abstention_wording_is_identical_in_code_and_docs() -> None:
    from agentpath.evidence import ABSTENTION_MESSAGE

    for document in ("SECURITY.md", "EVALUATION.md"):
        text = (REPO_ROOT / document).read_text(encoding="utf-8")
        if "No confirmed procedure" in text:
            assert ABSTENTION_MESSAGE.rstrip(".") in text, document

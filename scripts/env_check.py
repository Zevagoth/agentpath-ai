#!/usr/bin/env python3
"""AgentPath AI environment verification.

Read-only inspection of the development machine. Nothing is downloaded,
installed, or started by this script.

It answers one question: is this machine ready to run the AgentPath AI stack
(RAGFlow + Elasticsearch + local model runtimes) before we spend bandwidth and
disk on Docker images and model weights?

Usage::

    python3 scripts/env_check.py                       # human readable table
    python3 scripts/env_check.py --json out.json       # machine readable
    python3 scripts/env_check.py --markdown docs/ENVIRONMENT.md

Exit codes:
    0  all gates pass (warnings allowed)
    1  at least one gate failed; the stack must not be started yet
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Requirements. Keep these in one place so docs and gates cannot drift apart.
# --------------------------------------------------------------------------- #

MIN_FREE_DISK_GB = 30.0
"""RAGFlow suggests 50 GB. Our measured need is ~12 GB (images + weights);
30 GB is the floor at which we refuse to start pulling."""

MIN_TOTAL_RAM_GB = 16.0
MIN_AVAILABLE_RAM_GB = 10.0
MIN_VM_MAX_MAP_COUNT = 262144
MIN_COMPOSE_VERSION = (2, 26, 1)
MIN_DOCKER_VERSION = (24, 0, 0)

#: Ports AgentPath AI publishes, all bound to 127.0.0.1 only.
#: ``owner`` is informational; ``required`` ports must be free.
PORT_PLAN: dict[int, dict[str, Any]] = {
    3000: {"owner": "apps/web (Next.js dev server)", "required": True},
    8000: {"owner": "apps/api (FastAPI)", "required": True},
    8081: {"owner": "RAGFlow web UI (nginx; 8080 was busy on this host)", "required": True},
    9380: {"owner": "RAGFlow HTTP API", "required": True},
    9381: {"owner": "RAGFlow admin server", "required": True},
    7997: {"owner": "AgentPath reranker sidecar (TEI, bge-reranker-v2-m3)", "required": True},
    7998: {"owner": "AgentPath embedder sidecar (TEI, bge-m3)", "required": True},
    1200: {"owner": "Elasticsearch (RAGFlow doc engine)", "required": True},
    3307: {"owner": "MySQL (RAGFlow metadata; host-side remap of 3306)", "required": True},
    6379: {"owner": "Redis (RAGFlow)", "required": True},
    4222: {"owner": "NATS (RAGFlow)", "required": True},
    9000: {"owner": "MinIO API (RAGFlow object storage)", "required": True},
    9001: {"owner": "MinIO console", "required": True},
    11434: {"owner": "Ollama (expected to be busy: the running daemon)", "required": False},
}

OLLAMA_URL = "http://127.0.0.1:11434"

Status = str  # "pass" | "warn" | "fail"


@dataclass
class Check:
    """A single gate result."""

    id: str
    status: Status
    detail: str
    remedy: str | None = None


@dataclass
class Report:
    """Everything env_check learned about the machine."""

    generated_at: str
    os: dict[str, Any] = field(default_factory=dict)
    cpu: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    disks: list[dict[str, Any]] = field(default_factory=list)
    gpu: dict[str, Any] | None = None
    tools: dict[str, str | None] = field(default_factory=dict)
    kernel_params: dict[str, Any] = field(default_factory=dict)
    ports: list[dict[str, Any]] = field(default_factory=list)
    ollama: dict[str, Any] = field(default_factory=dict)
    docker: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def run(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a command, returning stripped stdout or ``None`` on any failure."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """Extract the first dotted version number from arbitrary tool output.

    >>> parse_version("Docker Compose version v2.26.1")
    (2, 26, 1)
    >>> parse_version("no version here")
    """
    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    parts = [int(p) for p in match.groups() if p is not None]
    return tuple(parts)


def format_version(version: tuple[int, ...] | None) -> str:
    return ".".join(str(p) for p in version) if version else "unknown"


def port_is_busy(port: int, host: str = "127.0.0.1") -> bool:
    """True when *port* cannot be bound on *host* (i.e. something owns it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def read_sysctl(name: str) -> int | None:
    path = "/proc/sys/" + name.replace(".", "/")
    try:
        with open(path, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def bytes_to_gb(value: float) -> float:
    return round(value / (1024**3), 1)


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #


def collect_os() -> dict[str, Any]:
    pretty = None
    try:
        with open("/etc/os-release", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    return {
        "platform": sys.platform,
        "distribution": pretty or platform.system(),
        "kernel": platform.release(),
        "arch": platform.machine(),
    }


def collect_cpu() -> dict[str, Any]:
    model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"model": model, "logical_cores": os.cpu_count()}


def collect_memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    values[key] = int(parts[0])  # kB
    except OSError:
        return {}
    return {
        "total_gb": round(values.get("MemTotal", 0) / (1024**2), 1),
        "available_gb": round(values.get("MemAvailable", 0) / (1024**2), 1),
        "swap_total_gb": round(values.get("SwapTotal", 0) / (1024**2), 1),
    }


def collect_disks(paths: Iterable[str]) -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            stat = os.stat(path)
            key = (stat.st_dev, 0)
            if key in seen:
                continue
            seen.add(key)
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        disks.append(
            {
                "path": path,
                "total_gb": bytes_to_gb(usage.total),
                "free_gb": bytes_to_gb(usage.free),
            }
        )
    return disks


def collect_gpu() -> dict[str, Any] | None:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    first = out.splitlines()[0]
    fields = [f.strip() for f in first.split(",")]
    if len(fields) < 5:
        return None
    try:
        return {
            "name": fields[0],
            "memory_total_mib": int(float(fields[1])),
            "memory_used_mib": int(float(fields[2])),
            "memory_free_mib": int(float(fields[3])),
            "driver_version": fields[4],
        }
    except ValueError:
        return None


def collect_tools() -> dict[str, str | None]:
    return {
        "python3": run(["python3", "--version"]),
        "uv": run(["uv", "--version"]),
        "node": run(["node", "--version"]),
        "npm": run(["npm", "--version"]),
        "docker": run(["docker", "--version"]),
        "docker_compose": run(["docker", "compose", "version"]),
        "git": run(["git", "--version"]),
        "ollama": run(["ollama", "--version"]),
    }


def collect_docker() -> dict[str, Any]:
    info = run(
        [
            "docker",
            "info",
            "--format",
            "{{.ServerVersion}}|{{.Driver}}|{{.DockerRootDir}}",
        ]
    )
    if not info:
        return {"reachable": False}
    parts = info.split("|")
    bridge_ip = None
    routes = run(["ip", "-4", "addr", "show", "docker0"])
    if routes:
        match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", routes)
        if match:
            bridge_ip = match.group(1)
    return {
        "reachable": True,
        "server_version": parts[0] if parts else None,
        "storage_driver": parts[1] if len(parts) > 1 else None,
        "root_dir": parts[2] if len(parts) > 2 else None,
        "bridge_ip": bridge_ip,
    }


def collect_ollama() -> dict[str, Any]:
    """Ask the local Ollama daemon what it has. Never downloads anything."""
    result: dict[str, Any] = {"url": OLLAMA_URL, "reachable": False, "models": []}
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return result
    result["reachable"] = True
    result["models"] = [
        {"name": m.get("name"), "size_gb": round(m.get("size", 0) / (1024**3), 1)}
        for m in payload.get("models", [])
    ]
    result["listen_addresses"] = ollama_listen_addresses()
    return result


def ollama_listen_addresses() -> list[str]:
    """Which addresses the Ollama daemon is bound to (LAN-exposure check)."""
    out = run(["ss", "-ltnH"])
    if not out:
        return []
    addresses = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if local.rsplit(":", 1)[-1] == "11434":
            addresses.append(local)
    return addresses


def collect_ports() -> list[dict[str, Any]]:
    rows = []
    for port, meta in sorted(PORT_PLAN.items()):
        rows.append(
            {
                "port": port,
                "owner": meta["owner"],
                "required_free": meta["required"],
                "busy": port_is_busy(port),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def evaluate(report: Report) -> list[Check]:
    """Turn raw measurements into pass/warn/fail gates with remedies."""
    checks: list[Check] = []

    # Disk -----------------------------------------------------------------
    free = min((d["free_gb"] for d in report.disks), default=0.0)
    checks.append(
        Check(
            id="disk_space",
            status="pass" if free >= MIN_FREE_DISK_GB else "fail",
            detail=f"{free} GB free (minimum {MIN_FREE_DISK_GB} GB across workspace and Docker root)",
            remedy=None if free >= MIN_FREE_DISK_GB else "Free disk space before pulling images or model weights.",
        )
    )

    # RAM ------------------------------------------------------------------
    total_ram = report.memory.get("total_gb", 0.0)
    available_ram = report.memory.get("available_gb", 0.0)
    if total_ram < MIN_TOTAL_RAM_GB:
        ram_status: Status = "fail"
        ram_remedy = "RAGFlow with Elasticsearch needs at least 16 GB of RAM."
    elif available_ram < MIN_AVAILABLE_RAM_GB:
        ram_status = "warn"
        ram_remedy = (
            "Close memory-heavy applications before starting the stack; "
            "Elasticsearch alone is configured for ~4 GB."
        )
    else:
        ram_status = "pass"
        ram_remedy = None
    checks.append(
        Check(
            id="memory",
            status=ram_status,
            detail=f"{total_ram} GB total, {available_ram} GB available",
            remedy=ram_remedy,
        )
    )

    # Docker ---------------------------------------------------------------
    docker_version = parse_version(report.tools.get("docker"))
    docker_ok = bool(docker_version and docker_version >= MIN_DOCKER_VERSION)
    checks.append(
        Check(
            id="docker",
            status="pass" if docker_ok and report.docker.get("reachable") else "fail",
            detail=(
                f"docker {format_version(docker_version)}, "
                f"daemon {'reachable' if report.docker.get('reachable') else 'unreachable'}"
            ),
            remedy=None if docker_ok else "Install or start Docker >= 24.0.0.",
        )
    )

    compose_version = parse_version(report.tools.get("docker_compose"))
    compose_ok = bool(compose_version and compose_version >= MIN_COMPOSE_VERSION)
    checks.append(
        Check(
            id="docker_compose",
            status="pass" if compose_ok else "fail",
            detail=f"docker compose {format_version(compose_version)} (minimum {format_version(MIN_COMPOSE_VERSION)})",
            remedy=None if compose_ok else "Upgrade the Docker Compose plugin to >= 2.26.1.",
        )
    )

    # Kernel ---------------------------------------------------------------
    vmmc = report.kernel_params.get("vm.max_map_count")
    vmmc_ok = bool(vmmc and vmmc >= MIN_VM_MAX_MAP_COUNT)
    checks.append(
        Check(
            id="vm_max_map_count",
            status="pass" if vmmc_ok else "fail",
            detail=f"vm.max_map_count = {vmmc} (Elasticsearch needs >= {MIN_VM_MAX_MAP_COUNT})",
            remedy=None
            if vmmc_ok
            else (
                f"Run: sudo sysctl -w vm.max_map_count={MIN_VM_MAX_MAP_COUNT}  "
                f"(persist with: echo 'vm.max_map_count={MIN_VM_MAX_MAP_COUNT}' | "
                "sudo tee /etc/sysctl.d/99-agentpath.conf)"
            ),
        )
    )

    # Toolchain ------------------------------------------------------------
    checks.append(
        Check(
            id="uv",
            status="pass" if report.tools.get("uv") else "fail",
            detail=report.tools.get("uv") or "uv not found",
            remedy=None if report.tools.get("uv") else "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        )
    )
    node_version = parse_version(report.tools.get("node"))
    checks.append(
        Check(
            id="node",
            status="pass" if node_version and node_version >= (20,) else "fail",
            detail=f"node {format_version(node_version)} (minimum 20)",
            remedy=None if node_version and node_version >= (20,) else "Install Node.js >= 20.",
        )
    )

    # Ports ----------------------------------------------------------------
    conflicts = [p for p in report.ports if p["required_free"] and p["busy"]]
    checks.append(
        Check(
            id="ports",
            status="pass" if not conflicts else "fail",
            detail=(
                "all required ports free"
                if not conflicts
                else "busy: " + ", ".join(f"{p['port']} ({p['owner']})" for p in conflicts)
            ),
            remedy=None if not conflicts else "Stop the conflicting service or remap the port in infra/ragflow/env.template.",
        )
    )

    # Ollama ---------------------------------------------------------------
    listen = report.ollama.get("listen_addresses") or []
    lan_exposed = [a for a in listen if a.startswith(("0.0.0.0", "*", "[::]"))]
    if not report.ollama.get("reachable"):
        ollama_status: Status = "warn"
        ollama_detail = "Ollama daemon not reachable on 127.0.0.1:11434"
        ollama_remedy = "Start Ollama before Task 3 (embedding model registration)."
    elif lan_exposed:
        ollama_status = "fail"
        ollama_detail = f"Ollama is listening on {', '.join(lan_exposed)} (reachable from the LAN)"
        ollama_remedy = "Bind Ollama to 127.0.0.1 only; containers reach it through the Docker bridge relay."
    else:
        ollama_status = "pass"
        ollama_detail = f"reachable, bound to {', '.join(listen) or '127.0.0.1:11434'}, {len(report.ollama.get('models', []))} model(s) present"
        ollama_remedy = None
    checks.append(Check(id="ollama", status=ollama_status, detail=ollama_detail, remedy=ollama_remedy))

    # GPU ------------------------------------------------------------------
    if report.gpu is None:
        checks.append(
            Check(
                id="gpu",
                status="warn",
                detail="no NVIDIA GPU detected; embeddings and generation will run on CPU",
            )
        )
    else:
        free_vram = report.gpu["memory_free_mib"]
        checks.append(
            Check(
                id="gpu",
                status="pass" if free_vram >= 2000 else "warn",
                detail=(
                    f"{report.gpu['name']}, {free_vram} MiB free of {report.gpu['memory_total_mib']} MiB"
                ),
                remedy=None
                if free_vram >= 2000
                else (
                    "Low free VRAM: embeddings will fall back to CPU. Free VRAM before Stage 2 "
                    "(qwen2.5:7b-instruct needs roughly 5.5 GB)."
                ),
            )
        )

    return checks


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

SYMBOLS = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def render_table(report: Report) -> str:
    lines = ["", "AgentPath AI - environment check", "=" * 60]
    lines.append(f"generated: {report.generated_at}")
    lines.append(f"os:        {report.os.get('distribution')} (kernel {report.os.get('kernel')}, {report.os.get('arch')})")
    lines.append(f"cpu:       {report.cpu.get('model')} ({report.cpu.get('logical_cores')} threads)")
    lines.append(
        f"memory:    {report.memory.get('total_gb')} GB total / {report.memory.get('available_gb')} GB available"
    )
    for disk in report.disks:
        lines.append(f"disk:      {disk['path']}: {disk['free_gb']} GB free of {disk['total_gb']} GB")
    if report.gpu:
        lines.append(
            f"gpu:       {report.gpu['name']} - {report.gpu['memory_free_mib']} MiB free "
            f"of {report.gpu['memory_total_mib']} MiB (driver {report.gpu['driver_version']})"
        )
    lines.append("")
    lines.append("tools")
    lines.append("-" * 60)
    for name, value in report.tools.items():
        lines.append(f"  {name:<16} {value or 'not found'}")
    lines.append("")
    lines.append("ports (all AgentPath services publish on 127.0.0.1 only)")
    lines.append("-" * 60)
    for row in report.ports:
        state = "busy" if row["busy"] else "free"
        flag = "" if not row["required_free"] else " *required free*"
        lines.append(f"  {row['port']:<6} {state:<5} {row['owner']}{flag}")
    lines.append("")
    lines.append("gates")
    lines.append("-" * 60)
    for check in report.checks:
        lines.append(f"  [{SYMBOLS[check.status]}] {check.id}: {check.detail}")
        if check.remedy:
            lines.append(f"          -> {check.remedy}")
    lines.append("")
    verdict = "READY" if report.ok else "NOT READY"
    lines.append(f"verdict: {verdict}")
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: Report) -> str:
    rows = [
        "# Environment report",
        "",
        "Generated by `scripts/env_check.py`. Read-only inspection: this script never",
        "downloads, installs, or starts anything.",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- OS: {report.os.get('distribution')} (kernel `{report.os.get('kernel')}`, {report.os.get('arch')})",
        f"- CPU: {report.cpu.get('model')} — {report.cpu.get('logical_cores')} threads",
        f"- Memory: {report.memory.get('total_gb')} GB total, {report.memory.get('available_gb')} GB available, "
        f"{report.memory.get('swap_total_gb')} GB swap",
    ]
    if report.gpu:
        rows.append(
            f"- GPU: {report.gpu['name']} — {report.gpu['memory_free_mib']} MiB free of "
            f"{report.gpu['memory_total_mib']} MiB, driver {report.gpu['driver_version']}"
        )
    for disk in report.disks:
        rows.append(f"- Disk `{disk['path']}`: {disk['free_gb']} GB free of {disk['total_gb']} GB")
    if report.docker.get("reachable"):
        rows.append(
            f"- Docker: {report.docker.get('server_version')} "
            f"(storage `{report.docker.get('storage_driver')}`, root `{report.docker.get('root_dir')}`, "
            f"bridge `{report.docker.get('bridge_ip')}`)"
        )

    rows += ["", "## Toolchain", "", "| Tool | Version |", "| --- | --- |"]
    for name, value in report.tools.items():
        rows.append(f"| `{name}` | {value or '_not found_'} |")

    rows += [
        "",
        "## Port plan",
        "",
        "Every AgentPath AI service publishes on `127.0.0.1` only. Nothing is exposed to the LAN.",
        "",
        "| Port | State | Owner | Required free |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.ports:
        rows.append(
            f"| {row['port']} | {'busy' if row['busy'] else 'free'} | {row['owner']} | "
            f"{'yes' if row['required_free'] else 'no'} |"
        )

    if report.ollama.get("reachable"):
        # Only the count is reported. Which models a developer keeps locally is
        # their business and has no bearing on whether the stack can run.
        model_count = len(report.ollama.get("models", []))
        rows += [
            "",
            "## Ollama",
            "",
            f"- Reachable at `{report.ollama['url']}`",
            f"- Listening on: {', '.join(report.ollama.get('listen_addresses') or ['unknown'])}",
            f"- {model_count} pre-existing model(s), untouched by AgentPath AI",
        ]

    rows += ["", "## Gates", "", "| Gate | Status | Detail | Remedy |", "| --- | --- | --- | --- |"]
    for check in report.checks:
        rows.append(
            f"| `{check.id}` | {SYMBOLS[check.status]} | {check.detail} | {check.remedy or '—'} |"
        )
    rows += ["", f"**Verdict: {'READY' if report.ok else 'NOT READY'}**", ""]
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_report(workspace: str) -> Report:
    report = Report(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    report.os = collect_os()
    report.cpu = collect_cpu()
    report.memory = collect_memory()
    docker = collect_docker()
    report.docker = docker
    disk_paths = [workspace, docker.get("root_dir") or "/var/lib/docker", "/"]
    report.disks = collect_disks(disk_paths)
    report.gpu = collect_gpu()
    report.tools = collect_tools()
    report.kernel_params = {"vm.max_map_count": read_sysctl("vm.max_map_count")}
    report.ports = collect_ports()
    report.ollama = collect_ollama()
    report.checks = evaluate(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the AgentPath AI development environment.")
    parser.add_argument("--json", dest="json_path", help="write the raw report as JSON to this path")
    parser.add_argument("--markdown", dest="markdown_path", help="write a Markdown report to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress the human readable table")
    parser.add_argument(
        "--workspace",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="repository root (used for the disk space check)",
    )
    args = parser.parse_args(argv)

    report = build_report(args.workspace)

    if not args.quiet:
        print(render_table(report))
    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.markdown_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown_path)), exist_ok=True)
        with open(args.markdown_path, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

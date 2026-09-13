"""Tests for scripts/env_check.py.

The environment gate decides whether we are allowed to download multi-gigabyte
images and model weights, so its arithmetic is worth testing directly. The
collectors touch the real machine; the gate logic does not, so the gates are
tested against synthesized reports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CHECK_PATH = REPO_ROOT / "scripts" / "env_check.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("agentpath_env_check", ENV_CHECK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


env_check = _load_module()


def _healthy_report() -> Any:
    """A report that should pass every gate."""
    report = env_check.Report(generated_at="2026-01-01T00:00:00+00:00")
    report.os = {"distribution": "TestOS", "kernel": "1.0", "arch": "x86_64"}
    report.cpu = {"model": "Test CPU", "logical_cores": 20}
    report.memory = {"total_gb": 31.2, "available_gb": 14.0, "swap_total_gb": 31.0}
    report.disks = [{"path": "/", "total_gb": 900.0, "free_gb": 245.0}]
    report.gpu = {
        "name": "Test GPU",
        "memory_total_mib": 8188,
        "memory_used_mib": 1000,
        "memory_free_mib": 7188,
        "driver_version": "1.2.3",
    }
    report.tools = {
        "python3": "Python 3.12.7",
        "uv": "uv 0.11.8",
        "node": "v22.0.0",
        "npm": "10.0.0",
        "docker": "Docker version 29.8.0, build 88096ef005",
        "docker_compose": "Docker Compose version 2.29.0",
        "git": "git version 2.55.0",
        "ollama": "ollama version is 0.34.0",
        "socat": "/usr/bin/socat",
    }
    report.docker = {"reachable": True, "server_version": "29.8.0", "bridge_ip": "172.17.0.1"}
    report.kernel_params = {"vm.max_map_count": 1048576}
    report.ports = [
        {"port": 8000, "owner": "api", "required_free": True, "busy": False},
        {"port": 11434, "owner": "ollama", "required_free": False, "busy": True},
    ]
    report.ollama = {
        "url": "http://127.0.0.1:11434",
        "reachable": True,
        "models": [{"name": "some-model", "size_gb": 16.0}],
        "listen_addresses": ["127.0.0.1:11434"],
    }
    return report


def _gate(checks: list[Any], gate_id: str) -> Any:
    matches = [c for c in checks if c.id == gate_id]
    assert matches, f"gate {gate_id} not evaluated"
    return matches[0]


# --------------------------------------------------------------------------- #
# Version parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Docker Compose version v2.26.1", (2, 26, 1)),
        ("Docker Compose version 5.5.1", (5, 5, 1)),
        ("Docker version 29.8.0, build 88096ef005", (29, 8, 0)),
        ("v26.8.2", (26, 8, 2)),
        ("Python 3.14.7", (3, 14, 7)),
        ("uv 0.11.8 (x86_64-unknown-linux-gnu)", (0, 11, 8)),
        ("ollama version is 0.34.0", (0, 34, 0)),
        ("no version here", None),
        (None, None),
    ],
)
def test_parse_version(text: str | None, expected: tuple[int, ...] | None) -> None:
    assert env_check.parse_version(text) == expected


def test_compose_version_comparison_is_numeric_not_lexicographic() -> None:
    """5.5.1 must beat 2.26.1; string comparison would get this backwards."""
    assert env_check.parse_version("Docker Compose version 5.5.1") >= env_check.MIN_COMPOSE_VERSION
    assert env_check.parse_version("Docker Compose version 2.20.0") < env_check.MIN_COMPOSE_VERSION


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def test_healthy_report_passes_every_gate() -> None:
    report = _healthy_report()
    report.checks = env_check.evaluate(report)
    assert report.ok
    assert {c.status for c in report.checks} == {"pass"}


def test_low_disk_space_fails_and_explains_itself() -> None:
    report = _healthy_report()
    report.disks = [{"path": "/", "total_gb": 900.0, "free_gb": 12.0}]
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "disk_space")
    assert gate.status == "fail"
    assert not report.ok
    assert gate.remedy


def test_disk_gate_uses_the_tightest_filesystem() -> None:
    report = _healthy_report()
    report.disks = [
        {"path": "/home", "total_gb": 900.0, "free_gb": 245.0},
        {"path": "/var/lib/docker", "total_gb": 100.0, "free_gb": 5.0},
    ]
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "disk_space").status == "fail"


def test_low_vm_max_map_count_fails_with_the_exact_sudo_command() -> None:
    report = _healthy_report()
    report.kernel_params = {"vm.max_map_count": 65530}
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "vm_max_map_count")
    assert gate.status == "fail"
    assert "sudo sysctl -w vm.max_map_count=262144" in gate.remedy


def test_busy_required_port_fails_and_names_the_owner() -> None:
    report = _healthy_report()
    report.ports = [{"port": 8000, "owner": "apps/api (FastAPI)", "required_free": True, "busy": True}]
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "ports")
    assert gate.status == "fail"
    assert "8000" in gate.detail and "apps/api" in gate.detail


def test_busy_optional_port_is_not_a_failure() -> None:
    """Ollama's own port is expected to be busy."""
    report = _healthy_report()
    report.ports = [{"port": 11434, "owner": "Ollama", "required_free": False, "busy": True}]
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "ports").status == "pass"


def test_ollama_bound_to_all_interfaces_is_a_security_failure() -> None:
    """LAN exposure of the model runtime is a hard failure, not a warning."""
    report = _healthy_report()
    report.ollama["listen_addresses"] = ["0.0.0.0:11434"]
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "ollama")
    assert gate.status == "fail"
    assert "LAN" in gate.detail
    assert not report.ok


def test_ollama_bound_to_loopback_passes() -> None:
    report = _healthy_report()
    report.ollama["listen_addresses"] = ["127.0.0.1:11434"]
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "ollama").status == "pass"


def test_low_free_vram_warns_but_does_not_block_the_search_slice() -> None:
    report = _healthy_report()
    report.gpu["memory_free_mib"] = 70
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "gpu")
    assert gate.status == "warn"
    assert report.ok, "a busy GPU must not block CPU-based embedding"


def test_insufficient_total_ram_fails() -> None:
    report = _healthy_report()
    report.memory = {"total_gb": 8.0, "available_gb": 6.0, "swap_total_gb": 0.0}
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "memory").status == "fail"


def test_tight_available_ram_warns_only() -> None:
    report = _healthy_report()
    report.memory = {"total_gb": 31.2, "available_gb": 7.0, "swap_total_gb": 31.0}
    report.checks = env_check.evaluate(report)
    gate = _gate(report.checks, "memory")
    assert gate.status == "warn"
    assert report.ok


def test_missing_docker_daemon_fails() -> None:
    report = _healthy_report()
    report.docker = {"reachable": False}
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "docker").status == "fail"


def test_old_compose_plugin_fails() -> None:
    report = _healthy_report()
    report.tools["docker_compose"] = "Docker Compose version v2.20.0"
    report.checks = env_check.evaluate(report)
    assert _gate(report.checks, "docker_compose").status == "fail"


# --------------------------------------------------------------------------- #
# Report shape and rendering
# --------------------------------------------------------------------------- #


def test_report_serializes_with_an_ok_flag() -> None:
    report = _healthy_report()
    report.checks = env_check.evaluate(report)
    data = report.to_dict()
    assert data["ok"] is True
    assert isinstance(data["checks"], list)
    assert {"id", "status", "detail", "remedy"} <= set(data["checks"][0])


def test_markdown_report_states_the_verdict_and_the_loopback_policy() -> None:
    report = _healthy_report()
    report.checks = env_check.evaluate(report)
    markdown = env_check.render_markdown(report)
    assert "**Verdict: READY**" in markdown
    assert "127.0.0.1" in markdown
    assert "| Gate | Status | Detail | Remedy |" in markdown


def test_table_report_lists_every_gate() -> None:
    report = _healthy_report()
    report.checks = env_check.evaluate(report)
    table = env_check.render_table(report)
    for check in report.checks:
        assert check.id in table


def test_port_plan_publishes_nothing_outside_loopback_by_design() -> None:
    """Guard: the documented port plan must stay loopback-only."""
    assert env_check.PORT_PLAN, "port plan must not be empty"
    required = [p for p, meta in env_check.PORT_PLAN.items() if meta["required"]]
    assert 8000 in required and 9380 in required
    # 8080 is occupied on this host; the plan must not claim it.
    assert 8080 not in env_check.PORT_PLAN


def test_port_is_busy_detects_a_bound_socket() -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert env_check.port_is_busy(port) is True
    assert env_check.port_is_busy(port) is False

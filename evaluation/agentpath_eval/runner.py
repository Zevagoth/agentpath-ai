"""Benchmark runner: three modes, one query set, JSON and Markdown output.

Method, stated up front so the numbers can be read correctly:

* Every mode sees the identical query set.
* Each query gets one **warm-up** run whose latency is discarded, then
  ``--repeats`` measured runs. Without that, first-call model loading and
  connection setup would dominate the percentiles.
* Retrieval quality is scored on **supported** queries only.
* Abstention is scored on **unsupported** queries only, and false abstention on
  supported queries is reported next to recall so a cautious threshold cannot
  quietly inflate the result.

Usage:
    uv run agentpath-eval
    uv run agentpath-eval --repeats 5
    uv run agentpath-eval --modes lexical_baseline,ragflow_hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentpath.config import Settings, get_settings
from agentpath.paths import reports_dir
from agentpath_eval.metrics import ModeReport
from agentpath_eval.modes import (
    AgentPathAPIMode,
    LexicalBaselineMode,
    Mode,
    RAGFlowHybridMode,
    in_process_full_mode,
)
from agentpath_eval.queries import BenchmarkQuery, load_queries, split_by_support
from agentpath_eval.report import render_markdown

DEFAULT_MODES = ("lexical_baseline", "ragflow_hybrid", "agentpath_full")


async def run_mode(
    mode: Mode,
    queries: list[BenchmarkQuery],
    *,
    limit: int,
    repeats: int,
) -> ModeReport:
    """Run one mode over the query set with a discarded warm-up per query."""
    report = ModeReport(mode=mode.name, description=mode.description)

    for query in queries:
        # Warm-up: result kept for correctness, latency discarded.
        outcome = await mode.run(query, limit=limit)
        for _ in range(repeats):
            outcome = await mode.run(query, limit=limit)
            report.latency.samples.append(outcome.latency_ms)
        report.outcomes.append(outcome)

    return report


async def build_modes(names: list[str], settings: Settings) -> list[Mode]:
    modes: list[Mode] = []
    for name in names:
        if name == "lexical_baseline":
            modes.append(LexicalBaselineMode())
        elif name == "ragflow_hybrid":
            modes.append(RAGFlowHybridMode(settings))
        elif name == "agentpath_full":
            api_mode = AgentPathAPIMode(f"http://{settings.api_host}:{settings.api_port}")
            try:
                await api_mode.probe()
                modes.append(api_mode)
            except RuntimeError as exc:
                # Falling back keeps the benchmark runnable without a server, and
                # the report says which variant produced the numbers rather than
                # pretending the HTTP path was measured.
                print(f"note: {exc}", file=sys.stderr)
                print("note: measuring the full pipeline in process instead", file=sys.stderr)
                await api_mode.aclose()
                modes.append(in_process_full_mode(settings))
        else:
            raise SystemExit(f"unknown mode {name!r}; choose from {', '.join(DEFAULT_MODES)}")
    return modes


async def run_benchmark(args: argparse.Namespace) -> int:
    settings = get_settings()
    queries = load_queries()
    supported, unsupported = split_by_support(queries)

    print(f"\nquery set : {len(queries)} queries ({len(supported)} supported, {len(unsupported)} unsupported)")
    print(f"repeats   : {repeats_label(args.repeats)} (plus one discarded warm-up per query)")
    print(f"top-k     : {args.limit}")
    print(f"reranking : {'on' if settings.enable_rerank else 'off (AGENTPATH_ENABLE_RERANK)'}\n")

    modes = await build_modes(args.modes.split(","), settings)
    reports: list[ModeReport] = []
    try:
        for mode in modes:
            print(f"running {mode.name} ...", flush=True)
            report = await run_mode(mode, queries, limit=args.limit, repeats=args.repeats)
            reports.append(report)
    finally:
        for mode in modes:
            await mode.aclose()

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "query_set_size": len(queries),
        "supported_queries": len(supported),
        "unsupported_queries": len(unsupported),
        "repeats": args.repeats,
        "top_k": args.limit,
        "warmup_runs_discarded_per_query": 1,
        "reranking_enabled": settings.enable_rerank,
        "embedding_model": settings.ragflow_embedding_model,
        "rerank_model": settings.ragflow_rerank_model if settings.enable_rerank else None,
        "vector_similarity_weight": settings.vector_similarity_weight,
        "similarity_threshold": settings.similarity_threshold,
        "evidence_thresholds": {
            "strong_score": settings.evidence_strong_score,
            "strong_margin": settings.evidence_strong_margin,
            "partial_score": settings.evidence_partial_score,
        },
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    payload = {"metadata": metadata, "modes": [report.to_dict() for report in reports]}

    target = reports_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    json_path = target / f"benchmark-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = render_markdown(metadata, reports)
    markdown_path = target / f"benchmark-{stamp}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_path = target / "latest.md"
    latest_path.write_text(markdown, encoding="utf-8")

    print_console_summary(reports)
    print(f"\nwrote {json_path.relative_to(Path.cwd()) if json_path.is_relative_to(Path.cwd()) else json_path}")
    print(f"wrote {markdown_path.name} and latest.md in {target}")
    return 0


def repeats_label(repeats: int) -> str:
    return f"{repeats} measured run" + ("s" if repeats != 1 else "")


def print_console_summary(reports: list[ModeReport]) -> None:
    print()
    header = (
        f"{'mode':<28} {'R@1':>6} {'R@3':>6} {'MRR':>6} {'abstain':>8} {'false':>7} {'p50':>7} {'p95':>7} {'mean':>7}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report.mode:<28} "
            f"{report.recall_at_1:>6.0%} {report.recall_at_3:>6.0%} {report.mrr:>6.3f} "
            f"{report.abstention_accuracy:>8.0%} {report.false_abstention_rate:>7.0%} "
            f"{report.latency.p50_ms:>6.0f}m {report.latency.p95_ms:>6.0f}m {report.latency.mean_ms:>6.0f}m"
        )
    print("\nR@1/R@3/MRR/false: supported queries only. abstain: unsupported queries only.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentPath AI retrieval benchmark.")
    parser.add_argument("--repeats", type=int, default=3, help="measured runs per query (default 3)")
    parser.add_argument("--limit", type=int, default=3, help="procedures per query (default 3)")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help=f"comma-separated subset of: {', '.join(DEFAULT_MODES)}",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())

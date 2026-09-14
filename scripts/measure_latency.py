#!/usr/bin/env python3
"""Measure retrieval latency straight against RAGFlow.

A focused instrument, not the benchmark: it answers one question - does hybrid
retrieval come back inside the two-second budget, and does the right procedure
land first - without the API or the evaluation harness in the way. That isolation
matters when diagnosing, because it removes our own code from the measurement.

Usage:
    uv run python scripts/measure_latency.py
    uv run python scripts/measure_latency.py --rerank   # include the reranker
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]

#: (label, query, expected procedure). Chosen to cover the failure modes the
#: product exists to fix: exact wording, agent shorthand, Arabic, and typos.
QUERIES: list[tuple[str, str, str]] = [
    ("exact", "Property unable to accommodate a confirmed guest", "NW-HTL-014"),
    ("synonym", "hotel has no room for the guest", "NW-HTL-014"),
    ("synonym", "the hotel cannot find the booking", "NW-HTL-021"),
    ("synonym", "customer booked the same hotel twice", "NW-CXL-045"),
    ("synonym", "where is my refund, it never arrived", "NW-REF-018"),
    ("typo", "overboking at the hotle", "NW-HTL-014"),
    ("arabic", "الفندق ما عنده غرف والضيف واصل", "NW-HTL-014"),
    ("arabic", "الاسترداد متأخر وما وصلني المبلغ", "NW-REF-018"),
    ("arabic", "في حجزين لنفس الضيف ونفس التواريخ", "NW-CXL-045"),
    ("unsupported", "how do I reset my email password", None),
]

REPEATS = 3


@dataclass
class Measurement:
    label: str
    query: str
    expected: str | None
    latencies_ms: list[float]
    top_procedure: str | None
    top_score: float
    top3: list[str]

    @property
    def hit_at_1(self) -> bool | None:
        if self.expected is None:
            return None
        return self.top_procedure == self.expected

    @property
    def hit_at_3(self) -> bool | None:
        if self.expected is None:
            return None
        return self.expected in self.top3

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)


def read_env(name: str, default: str = "") -> str:
    env_file = REPO_ROOT / ".env.local"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure RAGFlow retrieval latency.")
    parser.add_argument("--rerank", action="store_true", help="include the cross-encoder reranker")
    parser.add_argument("--candidates", type=int, default=32, help="rerank candidate count")
    args = parser.parse_args(argv)

    base_url = read_env("RAGFLOW_BASE_URL", "http://127.0.0.1:9380")
    api_key = read_env("RAGFLOW_API_KEY")
    dataset_name = read_env("RAGFLOW_DATASET_NAME", "agentpath-northwind-procedures")
    rerank_model = read_env("RAGFLOW_RERANK_MODEL")

    if not api_key:
        print("RAGFLOW_API_KEY missing from .env.local", file=sys.stderr)
        return 1
    if args.rerank and not rerank_model:
        print("--rerank requested but RAGFLOW_RERANK_MODEL is empty in .env.local", file=sys.stderr)
        return 1

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=180.0, headers=headers) as client:
        response = client.get(f"{base_url}/api/v1/datasets", params={"page": 1, "page_size": 100})
        payload = response.json()
        data = payload.get("data")
        datasets = data.get("data") if isinstance(data, dict) else data
        dataset_id = next((str(d["id"]) for d in (datasets or []) if d.get("name") == dataset_name), None)
        if not dataset_id:
            print(f"dataset {dataset_name!r} not found; run: make ingest", file=sys.stderr)
            return 1

        mode = "hybrid + rerank" if args.rerank else "hybrid (BM25 + vector)"
        print(f"\nmode     : {mode}")
        print(f"dataset  : {dataset_name}")
        print(f"repeats  : {REPEATS} per query\n")

        measurements: list[Measurement] = []
        for label, query, expected in QUERIES:
            body = {
                "question": query,
                "dataset_ids": [dataset_id],
                "page_size": 12,
                "similarity_threshold": 0.15,
                "vector_similarity_weight": 0.6,
                "include_metadata": True,
                "highlight": True,
            }
            if args.rerank:
                body["rerank_id"] = rerank_model
                body["rerank_candidates_count"] = args.candidates

            latencies: list[float] = []
            chunks: list[dict] = []
            for _ in range(REPEATS):
                started = time.perf_counter()
                result = client.post(f"{base_url}/api/v1/retrieval", json=body)
                latencies.append((time.perf_counter() - started) * 1000)
                result.raise_for_status()
                chunks = (result.json().get("data") or {}).get("chunks") or []

            ordered: list[str] = []
            for chunk in chunks:
                procedure = (chunk.get("document_metadata") or {}).get("procedure_id")
                if procedure and procedure not in ordered:
                    ordered.append(procedure)

            measurements.append(
                Measurement(
                    label=label,
                    query=query,
                    expected=expected,
                    latencies_ms=latencies,
                    top_procedure=ordered[0] if ordered else None,
                    top_score=float(chunks[0].get("similarity", 0.0)) if chunks else 0.0,
                    top3=ordered[:3],
                )
            )

    # -- report ------------------------------------------------------------- #
    print(f"{'type':<12} {'median ms':>10} {'top-1':<12} {'score':>6}  {'@1':<4} {'@3':<4} query")
    print("-" * 100)
    for m in measurements:
        at1 = "-" if m.hit_at_1 is None else ("yes" if m.hit_at_1 else "NO")
        at3 = "-" if m.hit_at_3 is None else ("yes" if m.hit_at_3 else "NO")
        print(
            f"{m.label:<12} {m.median_ms:>10.0f} {m.top_procedure or '-':<12} "
            f"{m.top_score:>6.3f}  {at1:<4} {at3:<4} {m.query[:44]}"
        )

    all_latencies = [value for m in measurements for value in m.latencies_ms]
    all_latencies.sort()
    p95 = all_latencies[max(0, int(len(all_latencies) * 0.95) - 1)]
    scored = [m for m in measurements if m.expected]
    recall_1 = sum(1 for m in scored if m.hit_at_1) / len(scored)
    recall_3 = sum(1 for m in scored if m.hit_at_3) / len(scored)

    print("-" * 100)
    print(f"mean latency : {statistics.mean(all_latencies):.0f} ms")
    print(f"median       : {statistics.median(all_latencies):.0f} ms")
    print(f"p95          : {p95:.0f} ms")
    print(f"slowest      : {max(all_latencies):.0f} ms")
    print(f"Recall@1     : {recall_1:.0%}  ({sum(1 for m in scored if m.hit_at_1)}/{len(scored)})")
    print(f"Recall@3     : {recall_3:.0%}  ({sum(1 for m in scored if m.hit_at_3)}/{len(scored)})")

    budget_ms = 2000
    verdict = "within" if p95 <= budget_ms else "OVER"
    print(f"\nlatency budget {budget_ms} ms: p95 is {verdict}")
    print("(These are engine-only numbers. evaluation/ adds query understanding and grading.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

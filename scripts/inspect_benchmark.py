#!/usr/bin/env python3
"""Print the interesting parts of the newest benchmark JSON.

The aggregate table says whether the system is good; this says where it is wrong,
which is the only part that tells you what to fix next.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main() -> int:
    candidates = sorted(glob.glob("evaluation/reports/benchmark-*.json"))
    if not candidates:
        print("no benchmark JSON found; run: make eval", file=sys.stderr)
        return 1
    path = Path(candidates[-1])
    payload = json.loads(path.read_text(encoding="utf-8"))

    meta = payload["metadata"]
    print(f"report   : {path.name}")
    print(f"queries  : {meta['query_set_size']} ({meta['supported_queries']} supported, "
          f"{meta['unsupported_queries']} unsupported)")
    print(f"reranking: {'on' if meta['reranking_enabled'] else 'off'}")
    print(f"evidence : {meta['evidence_thresholds']}")

    for mode in payload["modes"]:
        quality = mode["retrieval_quality_supported_only"]
        print(f"\n=== {mode['mode']} ===")
        print(
            f"  R@1={quality['recall_at_1']:.0%}  R@3={quality['recall_at_3']:.0%}  "
            f"MRR={quality['mrr']:.3f}  false_abstention={quality['false_abstention_rate']:.0%}  "
            f"abstention={mode['abstention_unsupported_only']['abstention_accuracy']:.0%}"
        )
        latency = mode["latency"]
        print(f"  latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms mean={latency['mean_ms']}ms")

        print("  unsupported queries:")
        for outcome in mode["outcomes"]:
            if outcome["supported"]:
                continue
            verdict = "abstained  " if outcome["abstained"] else "ANSWERED   "
            top = ", ".join(outcome["ranked"][:2]) or "-"
            print(
                f"    {verdict} score={outcome['top_score']:<7} {outcome['evidence']:<13} "
                f"{outcome['query'][:44]:<46} -> {top}"
            )

        misses = [o for o in mode["outcomes"] if o["supported"] and not o["hit_at_3"]]
        if misses:
            print("  misses (expected not in top 3):")
            for outcome in misses:
                print(
                    f"    {outcome['query'][:44]:<46} expected={','.join(outcome['expected'])} "
                    f"got={','.join(outcome['ranked']) or '-'}"
                )

        weak = [
            o
            for o in mode["outcomes"]
            if o["supported"] and o["hit_at_3"] and not o["hit_at_1"]
        ]
        if weak:
            print("  right procedure found, but not first:")
            for outcome in weak:
                print(
                    f"    {outcome['query'][:44]:<46} expected={','.join(outcome['expected'])} "
                    f"got={','.join(outcome['ranked'])}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

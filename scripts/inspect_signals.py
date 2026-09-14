#!/usr/bin/env python3
"""Compare retrieval signals between answerable and unanswerable queries.

The benchmark showed the system answering two queries that have no procedure
behind them, both scoring around 0.49 while genuine matches score 0.51 and above.
A 0.02 gap is not something to set a threshold on. This script looks for a
*different* signal that separates the two groups, rather than tuning the number
that already failed to.

Usage: uv run python scripts/inspect_signals.py
"""

from __future__ import annotations

import asyncio
import sys

import httpx

API = "http://127.0.0.1:8000"

CASES: list[tuple[str, str, bool]] = [
    ("answerable", "hotel has no room for the guest", True),
    ("answerable", "الفندق ما عنده غرف والضيف واصل", True),
    ("answerable", "customer is asking about money that should have come back", True),
    ("answerable", "guest is at reception and there is a problem with the room", True),
    ("answerable", "nonref cxl request", True),
    ("unanswerable", "كيف أقدم طلب إجازة سنوية", False),
    ("unanswerable", "passenger name correction on a flight ticket", False),
    ("unanswerable", "how do I reset my email password", False),
    ("unanswerable", "what is the office wifi password", False),
]


async def main() -> int:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            await client.get(f"{API}/health")
        except httpx.HTTPError:
            print(f"API not reachable at {API}; start it with: make api", file=sys.stderr)
            return 1

        print(f"{'group':<13} {'score':>6} {'lex':>6} {'sem':>6} {'concepts':>9} {'evidence':<13} query")
        print("-" * 104)

        for group, query, _answerable in CASES:
            response = await client.post(f"{API}/api/search", json={"query": query})
            body = response.json()
            procedures = body.get("procedures") or []
            concepts = body["query"]["concepts"]

            if not procedures:
                print(
                    f"{group:<13} {'-':>6} {'-':>6} {'-':>6} {len(concepts):>9} "
                    f"{body['evidence']['level']:<13} {query[:40]}"
                )
                continue

            top = procedures[0]
            citation = top["citations"][0]
            print(
                f"{group:<13} {top['score']:>6.3f} "
                f"{(citation['lexical_score'] if citation['lexical_score'] is not None else 0):>6.3f} "
                f"{(citation['semantic_score'] if citation['semantic_score'] is not None else 0):>6.3f} "
                f"{len(concepts):>9} {body['evidence']['level']:<13} {query[:40]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
